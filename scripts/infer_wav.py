#!/usr/bin/env python3
"""Decode file audio rời (không qua manifest/fbank) bằng một hoặc nhiều checkpoint.

Repo chưa có script nào làm việc này: dump_val.py / eval_beam.py đều đọc fbank đã
extract sẵn, còn export.sh gọi scripts/infer_onnx.py mà file đó không tồn tại.

Fbank ở đây phải khớp TUYỆT ĐỐI với prepare_data.py, nếu không encoder nhận phân phối
input khác và output thành rác mà không có lỗi nào báo:
    num_mel_bins=80, high_freq=-400, snip_edges=False, dither=0, samp_freq=16000
Audio đọc bằng soundfile dtype float32 (giống prepare_data.py) — KHÔNG scale lên int16.

Model chỉ được train trên cut 1-20s (remove_short_and_long_utt), nên đưa file dài vào
một lần là chạy ngoài phân phối. Script không tự cắt: hãy cắt bằng ffmpeg trước.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, ".")

import train as T  # noqa: E402
from beam_search import greedy_search_batch, modified_beam_search  # noqa: E402
from icefall.checkpoint import load_checkpoint  # noqa: E402
from icefall.context_graph import ContextGraph  # noqa: E402

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80


def get_args():
    p = T.get_parser()
    p.add_argument("--wav", type=str, nargs="+", required=True)
    p.add_argument("--ckpt", type=str, nargs="+", required=True,
                   help="một hoặc nhiều checkpoint, dạng nhan:duong_dan hoặc duong_dan")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--blank-penalties", type=str, default="0",
                   help="DANH SACH blank_penalty, dau PHAY phan tach cac gia tri — "
                        "'0,1,2' = chay 3 muc bp=0, bp=1, bp=2 trong cung mot lan. "
                        "Thap phan dung dau CHAM: '0.5' la mot gia tri, con '0,5' la HAI "
                        "gia tri (0 va 5). Tru thang vao logit blank truoc softmax "
                        "(beam_search.py: logits[:,0] -= bp) → model noi nhieu hon: "
                        "deletion giam, insertion tang. Prod (Core-AI) dat "
                        "GIPFORMER_BLANK_PENALTY=1.0 tu 2026-08-12.")
    p.add_argument("--hotwords", type=str, default="")
    p.add_argument("--context-score", type=float, default=2.0)
    p.add_argument("--detail", action="store_true",
                   help="in n-best cuối cùng + timestamp từng token cho các mode beam")
    return p.parse_args()


class _BeamCapture:
    """Bắt HypothesisList cuối cùng của modified_beam_search.

    modified_beam_search chỉ trả hypothesis tốt nhất (beam_search.py:190
    `get_most_probable(length_norm=True)`), cả beam nằm trong biến cục bộ `B`.
    Patch tạm chính method đó là cách duy nhất lấy được n-best mà không phải fork
    hàm 200 dòng. Chỉ đọc, khôi phục lại ngay khi ra khỏi with.
    """

    def __init__(self):
        import beam_search as BS
        self.BS = BS
        self.lists = []

    def __enter__(self):
        cls = self.BS.HypothesisList
        self._orig = cls.get_most_probable

        def patched(inner_self, length_norm=False):
            self.lists.append(inner_self)
            return self._orig(inner_self, length_norm)

        cls.get_most_probable = patched
        return self

    def __exit__(self, *exc):
        self.BS.HypothesisList.get_most_probable = self._orig
        return False


def print_detail(hl, sp, hotword_phrases, timestamps, tokens, subsample=4,
                 frame_shift=0.01):
    """n-best + timestamp. timestamp[i] là frame SAU subsampling → ×4×10ms."""
    print(f"      timestamp từng token (frame sau subsampling ×{subsample} = "
          f"{subsample * frame_shift * 1000:.0f}ms):")
    line = []
    for t, tok in zip(timestamps, tokens):
        line.append(f"{t * subsample * frame_shift:5.2f}s {sp.id_to_piece(tok)}")
    for i in range(0, len(line), 6):
        print("        " + " | ".join(line[i:i + 6]))

    hyps = sorted(hl.data.values(), key=lambda h: -h.log_prob / max(len(h.ys), 1))
    print(f"      n-best cuối cùng ({len(hyps)} hypothesis trong beam, "
          f"sắp theo log_prob/len — đúng tiêu chí chọn của hàm):")
    for r, h in enumerate(hyps, 1):
        ys = h.ys[2:]                      # bỏ context_size=2 token khởi tạo
        text = sp.decode(ys)
        norm = float(h.log_prob) / max(len(h.ys), 1)
        hit = [p for p in hotword_phrases if p and p in text]
        mark = ("  ⟵ hotword: " + ", ".join(hit)) if hit else ""
        print(f"        #{r:<2d} logP {float(h.log_prob):9.3f}  /len {norm:7.4f}  "
              f"{len(ys):3d} tok  {text}{mark}")


def load_fbank(path: str, device) -> torch.Tensor:
    """(T, 80) fbank, cùng config với prepare_data.py."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)                      # (C, N)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    try:
        import kaldifeat
        opts = kaldifeat.FbankOptions()
        opts.device = device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.mel_opts.num_bins = NUM_MEL_BINS
        opts.mel_opts.high_freq = -400
        return kaldifeat.Fbank(opts)([wav[0].contiguous().to(device)])[0]
    except ImportError:
        return torchaudio.compliance.kaldi.fbank(
            wav[:1], num_mel_bins=NUM_MEL_BINS, high_freq=-400,
            snip_edges=False, dither=0.0, sample_frequency=SAMPLE_RATE).to(device)


def build_context_graph(path: str, sp, context_score: float):
    """Giống eval_beam.py: BẮT BUỘC lowercase, BPE của repo đã lowercase mọi piece."""
    phrases, ids_list, scores, skipped = [], [], [], 0
    unk = sp.piece_to_id("<unk>")
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        score = 0.0
        if ":" in line:
            head, _, tail = line.rpartition(":")
            if tail.strip().replace(".", "", 1).isdigit():
                line, score = head.strip(), float(tail.strip())
        ids = sp.encode(line.lower())
        if not ids or unk in ids:
            skipped += 1
            continue
        phrases.append(line.lower())
        ids_list.append(ids)
        scores.append(score)
    logging.info(f"hotwords: {len(phrases)} cụm"
                 + (f" (bỏ {skipped} cụm ra <unk>)" if skipped else ""))
    g = ContextGraph(context_score)
    g.build(token_ids=ids_list, phrases=phrases, scores=scores)
    return g, phrases


def main():
    args = get_args()
    params = T.get_params()
    params.update(vars(args))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()
    params.dtype = torch.float32
    params.use_autocast = False

    ctx, hw_phrases = build_context_graph(
        params.hotwords, sp, params.context_score) if params.hotwords else (None, [])

    feats = {}
    for w in params.wav:
        f = load_fbank(w, device)
        feats[w] = f
        logging.info(f"{Path(w).name}: {f.shape[0]} frame = {f.shape[0] * 0.01:.2f}s")

    for spec in params.ckpt:
        name, _, path = spec.rpartition(":") if ":" in spec else ("", "", spec)
        name = name or Path(path).stem
        model = T.get_model(params)
        load_checkpoint(path, model)
        model.to(device).eval()

        print(f"\n{'=' * 78}\n{name}   ({path})\n{'=' * 78}")
        with torch.no_grad():
            for w in params.wav:
                f = feats[w]
                x = f.unsqueeze(0)
                x_lens = torch.tensor([f.shape[0]], device=device)
                enc, enc_lens = model.forward_encoder(x, x_lens)
                print(f"\n--- {Path(w).name} ---")
                bps = [float(x) for x in str(params.blank_penalties).split(",")
                       if x.strip()]

                def _sfx(bp):
                    return "" if bp == 0 else f"+bp{bp:g}"

                for bp in bps:
                    g = sp.decode(greedy_search_batch(
                        model=model, encoder_out=enc, encoder_out_lens=enc_lens,
                        blank_penalty=bp))[0]
                    print(f"  [{'greedy' + _sfx(bp):>16s}] {g}")

                variants = [(f"beam{params.beam}", None)]
                if ctx is not None:
                    variants.append((f"beam{params.beam}+hw", ctx))
                for label, cg in variants:
                    for bp in bps:
                        with _BeamCapture() as cap:
                            res = modified_beam_search(
                                model=model, encoder_out=enc,
                                encoder_out_lens=enc_lens, beam=params.beam,
                                context_graph=cg, blank_penalty=bp,
                                return_timestamps=True)
                        toks, ts = res.hyps[0], res.timestamps[0]
                        print(f"  [{label + _sfx(bp):>16s}] {sp.decode(toks)}")
                        if params.detail:
                            print_detail(cap.lists[0], sp, hw_phrases, ts, toks)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
