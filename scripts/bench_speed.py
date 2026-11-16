#!/usr/bin/env python3
"""Benchmark tốc độ fp32 vs int8 (sherpa-onnx, đúng runtime prod).

3 model × 4 cấu hình:
  - greedy, 1 stream/lần, 8 threads   (độ trễ đơn lẻ — quan trọng nhất cho UAT)
  - beam16, 1 stream/lần, 8 threads   (gần cấu hình prod decode)
  - greedy, 1 stream/lần, 1 thread    (latency 1 lõi)
  - greedy, 32 stream/lô, 8 threads   (throughput hàng loạt)

Đo: warmup 10 cut (bỏ), 200 cut mẫu (seed cố định), tính wall + RTF
(wall/audio — nhỏ hơn = nhanh hơn realtime nhiều lần).
"""
import gzip, json, random, time, statistics
import sherpa_onnx, soundfile as sf

CUTS_N, WARMUP = 200, 10
random.seed(42)
cuts = []
for ln in gzip.open("fbank_scratch/librispeech_cuts_dev-clean.jsonl.gz", "rt"):
    d = json.loads(ln)
    if d["duration"] >= 1.0:
        cuts.append((d["id"], d["recording"]["sources"][0]["source"].replace("/data/", "raw_dataset/"), d["duration"]))
sample = random.sample(cuts, CUTS_N)
wavs = {}
for cid, p, dur in sample:
    w, sr = sf.read(p, dtype="float32", always_2d=True)
    wavs[cid] = (w[:, 0], sr)
audio_total = sum(d for _, _, d in sample)
print(f"{CUTS_N} cut, tổng {audio_total:.0f}s audio, median {statistics.median(d for _,_,d in sample):.1f}s", flush=True)

MODELS = [
    ("fp32-gốc", "onnx_v4_avg", "avg-96-100"),
    ("QAT-int8", "onnx_v4_qat", "epoch-6.int8"),
    ("PTQ-int8", "onnx_v4_avg", "avg-96-100.int8"),
]
CONFIGS = [
    ("greedy b1 t8", "greedy_search", 1, 8),
    ("beam16 b1 t8", "modified_beam_search", 1, 8),
    ("greedy b1 t1", "greedy_search", 1, 1),
    ("greedy b32 t8", "greedy_search", 32, 8),
]

def make_rec(md, suf, method, threads):
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{md}/encoder-{suf}.onnx", decoder=f"{md}/decoder-{suf}.onnx",
        joiner=f"{md}/joiner-{suf}.onnx", tokens=f"{md}/tokens.txt",
        num_threads=threads, sample_rate=16000, feature_dim=80,
        decoding_method=method, model_type="zipformer2", provider="cpu")

results = {}
for mlabel, md, suf in MODELS:
    for clabel, method, batch, threads in CONFIGS:
        rec = make_rec(md, suf, method, threads)
        streams = []
        for cid, _, _ in sample:
            w, sr = wavs[cid]
            s = rec.create_stream(); s.accept_waveform(sr, w); streams.append(s)
        # warmup
        for s in streams[:WARMUP]:
            rec.decode_streams([s])
        t0 = time.perf_counter()
        if batch == 1:
            lat = []
            for s in streams:
                ta = time.perf_counter(); rec.decode_streams([s]); lat.append(time.perf_counter() - ta)
            wall = time.perf_counter() - t0
            p50 = statistics.median(lat); p95 = sorted(lat)[int(0.95*len(lat))]
        else:
            for i in range(0, len(streams), batch):
                rec.decode_streams(streams[i:i+batch])
            wall = time.perf_counter() - t0
            p50 = p95 = float("nan")
        rtf = wall / audio_total
        results[(mlabel, clabel)] = (wall, rtf, p50, p95)
        print(f"{mlabel:10} {clabel:13} wall {wall:7.1f}s  RTF {rtf:6.3f}  ({1/rtf:5.1f}x realtime)  p50 {p50*1000:6.0f}ms  p95 {p95*1000:6.0f}ms", flush=True)

print("\n=== TÓM TẮT: tốc độ tương đối (QAT-int8 / fp32-gốc) ===")
for clabel, *_ in CONFIGS:
    w_fp = results[("fp32-gốc", clabel)][0]
    w_q = results[("QAT-int8", clabel)][0]
    print(f"  {clabel:13}: fp32 {w_fp:6.1f}s | QAT-int8 {w_q:6.1f}s | fp32 chậm hơn {(w_q/w_fp):.2f}x-nhanh... speedup QAT = {w_fp/w_q:.2f}x")
