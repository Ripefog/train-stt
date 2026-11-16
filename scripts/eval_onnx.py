#!/usr/bin/env python3
"""Đo WER + recall thuật ngữ của bộ ONNX bằng sherpa-onnx — ĐÚNG runtime của prod.

Vì sao cần script riêng: eval_beam.py chạy PyTorch trên fbank đã extract sẵn, còn prod
chạy sherpa-onnx trên wav thô. Hai đường đó có thể lệch (fbank config, context graph của
sherpa khác icefall). Script này đo trên đường prod thật.

CHỈ decode rồi ghi JSONL (ref + hyp_<mode>). WER/recall tính bằng scripts/metrics.py
ở ngoài — image worker KHÔNG có jiwer lẫn pip, nên không tính được trong container.

Audio đọc từ recording.sources[0].source của manifest val, remap prefix cho khớp host.
"""
import argparse, gzip, json, re, unicodedata
from pathlib import Path

import soundfile as sf
import sherpa_onnx


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


PROVIDER = "cpu"
SUFFIX = "epoch-40"     # hậu tố tên file onnx; prod dùng "epoch-35-avg-6"
BPE_VOCAB = ""         # để trống = lấy {model_dir}/bpe.vocab


def build(md, method, bp, hotwords, score, threads):
    kw = dict(
        encoder=f"{md}/encoder-{SUFFIX}.onnx", decoder=f"{md}/decoder-{SUFFIX}.onnx",
        joiner=f"{md}/joiner-{SUFFIX}.onnx", tokens=f"{md}/tokens.txt",
        num_threads=threads, sample_rate=16000, feature_dim=80,
        decoding_method=method, blank_penalty=bp,
        model_type="zipformer2", provider=PROVIDER, debug=False)
    if method == "modified_beam_search":
        kw["max_active_paths"] = 16
        if hotwords:
            # sherpa-onnx BẮT BUỘC có bpe_vocab khi modeling_unit=bpe — đây đúng là lý do
            # prod comment hotwords ra (model_registry.py:83-94), file đó chưa từng có.
            kw.update(hotwords_file=hotwords, hotwords_score=score,
                      modeling_unit="bpe", bpe_vocab=BPE_VOCAB or f"{md}/bpe.vocab")
    return sherpa_onnx.OfflineRecognizer.from_transducer(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/m")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True,
                    help="thay tiền tố '/data/' trong manifest bằng đường dẫn này")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hotwords", default="")
    ap.add_argument("--context-score", type=float, default=2.0)
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--suffix", default="epoch-40",
                    help='hậu tố tên file onnx, vd "epoch-35-avg-6"')
    ap.add_argument("--bpe-vocab", default="",
                    help="đường dẫn bpe.vocab riêng (bundle prod chưa có file này)")
    a = ap.parse_args()
    global PROVIDER, SUFFIX, BPE_VOCAB
    PROVIDER, SUFFIX, BPE_VOCAB = a.provider, a.suffix, a.bpe_vocab

    cuts = []
    op = gzip.open if a.manifest.endswith(".gz") else open
    with op(a.manifest, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["duration"] < a.min_duration:
                continue
            src = d["recording"]["sources"][0]["source"]
            cuts.append((d["id"], src.replace("/data/", a.audio_root.rstrip("/") + "/", 1),
                         d["supervisions"][0]["text"]))
    print(f"val: {len(cuts)} cut (duration >= {a.min_duration}s)", flush=True)

    specs = [("greedy", "greedy_search", 0.0, False),
             ("beam16", "modified_beam_search", 0.0, False)]
    if a.hotwords:
        specs += [("beam16+hw", "modified_beam_search", 0.0, True),
                  ("beam16+hw+bp1", "modified_beam_search", 1.0, True)]

    wavs = {}
    for cid, p, _ in cuts:
        w, sr = sf.read(p, dtype="float32", always_2d=True)
        wavs[cid] = (w[:, 0], sr)

    rows = {cid: {"id": cid, "ref": ref} for cid, _, ref in cuts}
    # ── Chia khúc audio dài ────────────────────────────────────────────────────
    # Encoder ONNX (trace của icefall, cả bản official lẫn custom) bake Slice của
    # positional encoding theo chiều dài lúc trace — file dài (~>20s) crash Reshape
    # trong sherpa. Mọi bundle đều bị như nhau nên chia đều 20s cho TẤT CẢ các hàng
    # — so sánh giữa các hàng vẫn công bằng (513/10.830 cut bị chia, median 5,9s).
    CHUNK_SEC = 20.0
    for label, method, bp, hw in specs:
        r = build(a.model_dir, method, bp, a.hotwords if hw else "",
                  a.context_score, a.threads)
        # decode theo lô để sherpa dùng được song song
        ids = [c[0] for c in cuts]
        piece_streams, piece_owner = [], []
        for cid in ids:
            w, sr = wavs[cid]
            n_piece = max(1, int(w.shape[0] / (sr * CHUNK_SEC)) + (1 if w.shape[0] % (sr * CHUNK_SEC) else 0))
            step = (w.shape[0] + n_piece - 1) // n_piece
            for k in range(n_piece):
                seg = w[k*step:(k+1)*step]
                if seg.shape[0] == 0:
                    continue
                s = r.create_stream()
                s.accept_waveform(sr, seg)
                piece_streams.append(s)
                piece_owner.append(cid)
        for i in range(0, len(piece_streams), 32):
            r.decode_streams(piece_streams[i:i + 32])
        joined = {cid: [] for cid in ids}
        for owner, s in zip(piece_owner, piece_streams):
            txt = s.result.text.strip()
            if txt:
                joined[owner].append(txt)
        for cid in ids:
            rows[cid]["hyp_" + label] = " ".join(joined[cid])
        print(f"  {label}: xong ({len(piece_streams)} stream / {len(ids)} cut)", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_modes": [x[0] for x in specs]}, ensure_ascii=False) + "\n")
        for c in cuts:
            f.write(json.dumps(rows[c[0]], ensure_ascii=False) + "\n")
    print("ghi", a.out)


if __name__ == "__main__":
    main()
