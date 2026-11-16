#!/usr/bin/env python3
"""Greedy transducer decode, in timestamp + confidence TỪNG TOKEN, và dò xem một từ
mục tiêu (vd "quai") trượt ở đâu.

Vì sao phải viết mới: `greedy_search_batch` trả `scores` là **logit thô** của token
được chọn, không phải xác suất — logit không so sánh được giữa các frame. Còn
`infer_wav.py --detail` chỉ in timestamp và chỉ cho nhánh beam. Ở đây decode lại đúng
vòng lặp của `greedy_search_batch` (max_sym_per_frame=1) nhưng giữ softmax đầy đủ.

CẢNH BÁO khi đọc kết quả: transducer là autoregressive — xác suất của "quai" tại frame
t phụ thuộc decoder state, tức phụ thuộc những gì ĐÃ phát ra trước đó. Bảng dò dưới đây
đo dọc theo đúng đường greedy thực tế, nên nó trả lời được "trên đường model đã đi,
'quai' có suýt thắng không", chứ KHÔNG trả lời được "nếu ép tiền tố khác thì sao".
Muốn cái sau thì phải teacher-forcing, là việc khác.
"""
import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, ".")

import train as T  # noqa: E402
from icefall.checkpoint import load_checkpoint  # noqa: E402

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
FRAME_SHIFT = 0.01
SUBSAMPLE = 4          # encoder frame = 4 × 10ms = 40ms, khớp infer_wav.py:88


def load_fbank(path, device):
    """(T, 80) — phải khớp TUYỆT ĐỐI prepare_data.py, lệch là encoder ăn rác im lặng."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return torchaudio.compliance.kaldi.fbank(
        wav[:1], num_mel_bins=NUM_MEL_BINS, high_freq=-400,
        snip_edges=False, dither=0.0, sample_frequency=SAMPLE_RATE).to(device)


def greedy_with_probs(model, enc, blank_id, context_size, blank_penalty=0.0):
    """Lặp lại greedy_search_batch cho N=1, nhưng giữ softmax mọi frame.

    Trả (emitted, frames): emitted là các token phát ra; frames là toàn bộ T frame kể cả
    frame ra blank — cần frame blank vì chỗ "quai" bị nuốt thường là frame model chọn
    blank chứ không phải chọn nhầm token khác.
    """
    device = enc.device
    hyp = [-1] * (context_size - 1) + [blank_id]
    dec_in = torch.tensor([hyp], device=device, dtype=torch.int64)
    dec_out = model.joiner.decoder_proj(model.decoder(dec_in, need_pad=False))
    enc_proj = model.joiner.encoder_proj(enc[0])          # (T, C)

    emitted, frames = [], []
    for t in range(enc_proj.shape[0]):
        cur = enc_proj[t].unsqueeze(0).unsqueeze(1).unsqueeze(1)
        logits = model.joiner(cur, dec_out.unsqueeze(1), project_input=False)
        logits = logits.squeeze(1).squeeze(1)[0]           # (vocab,)
        if blank_penalty:
            logits[blank_id] -= blank_penalty
        probs = logits.softmax(dim=-1)
        top_p, top_i = probs.topk(6)
        y = int(probs.argmax())
        frames.append({
            "t": t, "probs": probs.detach().cpu(), "argmax": y,
            "top": [(int(i), float(p)) for p, i in zip(top_p, top_i)],
        })
        if y != blank_id:
            emitted.append({"t": t, "tok": y, "p": float(probs[y]),
                            "p_blank": float(probs[blank_id]),
                            "runner_up": frames[-1]["top"][1]})
            hyp.append(y)
            dec_in = torch.tensor([hyp[-context_size:]], device=device,
                                  dtype=torch.int64)
            dec_out = model.joiner.decoder_proj(model.decoder(dec_in, need_pad=False))
    return emitted, frames


def main():
    p = T.get_parser()
    p.add_argument("--wav", required=True)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--targets", default="",
                   help="các từ cần dò, phân tách bằng dấu phẩy, vd 'quai,quay'")
    p.add_argument("--blank-penalty", type=float, default=0.0)
    p.add_argument("--dump-frames", action="store_true",
                   help="in mọi frame kể cả blank (dài — 25 dòng/giây audio)")
    p.add_argument("--watch", default="",
                   help="cửa sổ thời gian 'lo,hi' (giây) — in xác suất CHÍNH XÁC của từng "
                        "mảnh BPE của mọi --targets ở mọi frame trong cửa sổ. Cần vì bảng "
                        "top-k làm tròn 3 chữ số, mà đúng những mảnh đang truy thì nằm ở "
                        "mức 1e-4 nên hiện ra '0.000'.")
    args = p.parse_args()

    params = T.get_params()
    params.update(vars(args))
    device = torch.device("cpu")          # clip 10s, không tranh GPU với run đang train

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()
    params.dtype = torch.float32
    params.use_autocast = False
    blank_id = params.blank_id

    targets = [w.strip().lower() for w in params.targets.split(",") if w.strip()]

    feat = load_fbank(params.wav, device)
    print(f"{Path(params.wav).name}: {feat.shape[0]} frame fbank = "
          f"{feat.shape[0] * FRAME_SHIFT:.2f}s")

    for spec in params.ckpt:
        name, _, path = spec.rpartition(":") if ":" in spec else ("", "", spec)
        name = name or Path(path).stem
        model = T.get_model(params)
        load_checkpoint(path, model)
        model.to(device).eval()
        ctx_size = model.decoder.context_size

        print(f"\n{'=' * 90}\n{name}   ({path})\n{'=' * 90}")
        with torch.no_grad():
            x = feat.unsqueeze(0)
            x_lens = torch.tensor([feat.shape[0]], device=device)
            enc, _ = model.forward_encoder(x, x_lens)
            emitted, frames = greedy_with_probs(
                model, enc, blank_id, ctx_size, params.blank_penalty)

        step = SUBSAMPLE * FRAME_SHIFT
        print(f"encoder: {enc.shape[1]} frame × {step * 1000:.0f}ms\n")
        print(sp.decode([e["tok"] for e in emitted]))

        print(f"\n{'t(s)':>7} {'token':<14} {'p':>6} {'p_blank':>8}  "
              f"{'á quân':<14} {'p2':>6}")
        print("-" * 62)
        for e in emitted:
            r_tok, r_p = e["runner_up"]
            print(f"{e['t'] * step:7.2f} {sp.id_to_piece(e['tok']):<14} "
                  f"{e['p']:6.3f} {e['p_blank']:8.3f}  "
                  f"{sp.id_to_piece(r_tok):<14} {r_p:6.3f}")

        # ── Dò từ mục tiêu ────────────────────────────────────────────────────
        for w in targets:
            ids = sp.encode(w)
            pieces = [sp.id_to_piece(i) for i in ids]
            if not ids:
                print(f"\n[{w}] không mã hoá được bằng BPE này — bỏ qua.")
                continue
            first = ids[0]
            best = max(frames, key=lambda f: float(f["probs"][first]))
            print(f"\n[{w}] BPE = {' '.join(pieces)}   (dò mảnh đầu "
                  f"'{pieces[0]}', id {first})")
            pk = float(best["probs"][first])
            rank = int((best["probs"] > pk).sum()) + 1
            print(f"  đỉnh tại {best['t'] * step:.2f}s: p={pk:.4f}, hạng {rank}/"
                  f"{len(best['probs'])}, "
                  f"thắng cuộc là '{sp.id_to_piece(best['argmax'])}' "
                  f"p={float(best['probs'][best['argmax']]):.3f}")
            near = [f for f in frames if float(f["probs"][first]) > 0.01]
            print(f"  số frame có p > 0.01: {len(near)}/{len(frames)}")
            for f in sorted(near, key=lambda f: -float(f["probs"][first]))[:5]:
                pv = float(f["probs"][first])
                r = int((f["probs"] > pv).sum()) + 1
                print(f"    {f['t'] * step:6.2f}s p={pv:.4f} hạng {r:<4d} "
                      f"(thắng: {sp.id_to_piece(f['argmax'])} "
                      f"{float(f['probs'][f['argmax']]):.3f})")

        # ── Cửa sổ soi kỹ: xác suất chính xác của mọi mảnh BPE của target ────
        if params.watch and targets:
            lo, hi = (float(v) for v in params.watch.split(","))
            cols = []
            for w in targets:
                for i in sp.encode(w):
                    if (w, i) not in cols:
                        cols.append((w, i))
            hdr = "  ".join(f"{sp.id_to_piece(i):>9}" for _, i in cols)
            print(f"\ncửa sổ {lo:.2f}–{hi:.2f}s — p chính xác từng mảnh "
                  f"({', '.join(sorted({w for w, _ in cols}))})")
            print(f"{'t(s)':>6}  {hdr}  {'p_blank':>9}  thắng")
            print("-" * (24 + 11 * len(cols)))
            for f in frames:
                t = f["t"] * step
                if not (lo <= t <= hi):
                    continue
                vals = "  ".join(f"{float(f['probs'][i]):9.5f}" for _, i in cols)
                print(f"{t:6.2f}  {vals}  {float(f['probs'][blank_id]):9.5f}  "
                      f"{sp.id_to_piece(f['argmax'])}")

        if params.dump_frames:
            print(f"\n{'t(s)':>7}  top-6 mỗi frame")
            print("-" * 90)
            for f in frames:
                items = " ".join(f"{sp.id_to_piece(i)}:{pp:.3f}" for i, pp in f["top"])
                print(f"{f['t'] * step:7.2f}  {items}")

        del model


if __name__ == "__main__":
    main()
