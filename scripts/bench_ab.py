#!/usr/bin/env python3
"""Benchmark A/B công bằng giữa các bundle quantize.

Nguyên tắc công bằng (khác bench_speed.py — bench tuần tự mỗi model một lượt):
  1. MỘT process, nạp ĐỒNG THỜI các model — cùng binary sherpa-onnx, cùng thread pool.
  2. MỘT config duy nhất cho mọi model (greedy, batch 1; num_threads cố định).
  3. INTERLEAVE theo block: xoay vòng model-block-model-block... trong từng round —
     nhiễu tải máy shared (job của người khác nhảy vào giữa chừng) trải đều lên mọi model.
  4. Nhiều round, lấy MEDIAN — bỏ drift nhiệt độ/cache/nhiễu một chiều.
  5. Chạy dưới `taskset -c <core>` để mọi model đo trên cùng 1 lõi vật lý.

Chạy:
  taskset -c 4 python3 scripts/bench_ab.py            # 1 thread (kernel thuần)
  taskset -c 4 python3 scripts/bench_ab.py --threads 8
"""
import argparse
import gzip
import json
import random
import statistics
import time

import sherpa_onnx
import soundfile as sf

CUTS_N, WARMUP, ROUNDS, BLOCK = 200, 10, 3, 25
random.seed(42)
cuts = []
for ln in gzip.open("fbank_scratch/librispeech_cuts_dev-clean.jsonl.gz", "rt"):
    d = json.loads(ln)
    if d["duration"] >= 1.0:
        cuts.append(
            (d["id"], d["recording"]["sources"][0]["source"].replace("/data/", "raw_dataset/"), d["duration"])
        )
sample = random.sample(cuts, CUTS_N)
wavs = {}
for cid, p, _ in sample:
    w, sr = sf.read(p, dtype="float32", always_2d=True)
    wavs[cid] = (w[:, 0], sr)
audio_total = sum(d for _, _, d in sample)
print(
    f"{CUTS_N} cut / {audio_total:.0f}s audio, {ROUNDS} round interleave block={BLOCK}",
    flush=True,
)

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=1)
args = ap.parse_args()

MODELS = [
    ("fp32", "onnx_v4_avg", "avg-96-100"),
    ("int8", "onnx_v4_avg", "avg-96-100.int8"),
    ("int4", "onnx_v4_int4", "avg-96-100.int4"),
]

recs = {}
for label, md, suf in MODELS:
    recs[label] = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{md}/encoder-{suf}.onnx",
        decoder=f"{md}/decoder-{suf}.onnx",
        joiner=f"{md}/joiner-{suf}.onnx",
        tokens=f"{md}/tokens.txt",
        num_threads=args.threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="zipformer2",
        provider="cpu",
    )

# streams tạo sẵn MỘT LẦN, tái dùng mọi round (feature extraction khỏi nằm vào số đo)
streams = {}
for label, rec in recs.items():
    ss = []
    for cid, _, _ in sample:
        w, sr = wavs[cid]
        s = rec.create_stream()
        s.accept_waveform(sr, w)
        ss.append(s)
    streams[label] = ss

# warmup mỗi model
for label, rec in recs.items():
    for s in streams[label][:WARMUP]:
        rec.decode_streams([s])
print("warmup xong", flush=True)

rounds_wall = {label: [] for label, *_ in MODELS}
for r in range(ROUNDS):
    for label, rec in recs.items():
        t_model = 0.0
        for i in range(WARMUP, CUTS_N, BLOCK):
            block = streams[label][i : i + BLOCK]
            t0 = time.perf_counter()
            for s in block:
                rec.decode_streams([s])
            t_model += time.perf_counter() - t0
        rounds_wall[label].append(t_model)
    detail = " | ".join(f"{l}: {rounds_wall[l][-1]:.1f}s" for l, *_ in MODELS)
    print(f"round {r + 1}/{ROUNDS}: {detail}", flush=True)

print(f"\n=== KẾT QUẢ (greedy, batch 1, {args.threads} thread, median {ROUNDS} round) ===")
med = {}
for label, *_ in MODELS:
    m = statistics.median(rounds_wall[label])
    med[label] = m
    rtf = m / audio_total
    print(
        f"  {label:5} wall {m:6.1f}s  RTF {rtf:6.3f}  ({1 / rtf:5.1f}x realtime)"
        f"  [các round: {', '.join(f'{w:.1f}' for w in rounds_wall[label])}]"
    )
base = med["int8"]
print("\n=== TỐC ĐỘ TƯƠNG ĐỐI (int8 = 1.00) ===")
for label, *_ in MODELS:
    print(f"  {label:5} {med[label] / base:.2f}x thời gian int8")
