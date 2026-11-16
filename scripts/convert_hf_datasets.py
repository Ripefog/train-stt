#!/usr/bin/env python3
"""Convert parquet HF datasets → format repo (X.wav + X.txt) cho prepare_data.py.

Output: raw_dataset/train/<ds>/<ds>_<id>.{wav,txt} và raw_dataset/val/<ds>/...
Split 95/5 DETERMINISTIC theo md5("<ds>/<id>") — không phụ thuộc thứ tự chạy,
chạy lại cho cùng kết quả. Prefix <ds>_ trong tên file để metrics phân rã
per-domain từ cut id (eval_beam.py chỉ ghi cut id trong JSONL).

Audio: decode bằng soundfile (WAV/FLAC/OGG/MP3 — libsndfile 1.2 đọc được hết,
đã test trên data thật của repo này), mix đa kênh bằng mean, resample về 16 kHz
bằng scipy.resample_poly nếu cần, ghi PCM_16 mono.

Transcript ghi THÔ (không normalize) — normalize_text nằm trong prepare_data.py,
chạy đúng một lần lúc scan, giữ nguyên kiến trúc repo.

Bỏ qua: text rỗng/toàn khoảng trắng, audio decode lỗi, duration < 0.2s.
Không xoá gì — việc xoá parquet gốc là bước riêng sau khi verify.

Usage:
  python3 scripts/convert_hf_datasets.py --dataset vimd vietmed ...
  python3 scripts/convert_hf_datasets.py --list
"""
import argparse
import glob
import hashlib
import io
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
DS_ROOT = REPO / "datasets"
OUT_ROOT = REPO / "raw_dataset"
SAMPLE_RATE = 16000
VAL_PCT = 5

# ds_name → (glob tương đối dưới datasets/, cột text, cột text-dự-phòng)
DATASETS = {
    "vimd":       ("ViMD_Dataset/data/*.parquet", "text", None),
    "vietmed":    ("VietMed-Label/data/*.parquet", "transcription", None),
    "multimed":   ("MultiMED/vietnamese/*.parquet", "text", None),
    "vietmdd":    ("VietMDD/data/*.parquet", "observed_transcription", "original_text"),
    "gigaspeech": ("gigaspeech2_vie/data/*.parquet", "transcription", None),
    "fosd":       ("FOSD/data/*.parquet", "transcription", None),
    "vlsp2020":   ("vlsp2020/data/*.parquet", "transcription", None),
    "lsvsc":      ("LSVSC/data/*.parquet", "transcription", None),
}

_SAN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(s: str) -> str:
    s = _SAN.sub("_", s).strip("_.-")
    return s or "x"


def _is_val(ds: str, cut_id: str) -> bool:
    h = hashlib.md5(f"{ds}/{cut_id}".encode()).hexdigest()
    return int(h, 16) % 100 < VAL_PCT


def _work(args):
    """1 task = 1 batch nhỏ các row. Trả list (status, duration_s)."""
    ds, rows = args  # rows: list[(cut_id, text, audio_bytes)]
    import scipy.signal as _sig  # import trong worker sau fork

    out = []
    for cut_id, text, ab in rows:
        try:
            data, sr = sf.read(io.BytesIO(ab), dtype="float32", always_2d=True)
        except Exception:
            out.append(("decode_fail", 0.0))
            continue
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        dur = len(data) / sr
        if dur < 0.2:
            out.append(("too_short", dur))
            continue
        if sr != SAMPLE_RATE:
            data = _sig.resample_poly(data, SAMPLE_RATE, sr).astype("float32")
        split = "val" if _is_val(ds, cut_id) else "train"
        d = OUT_ROOT / split / ds
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{ds}_{cut_id}.wav", data, SAMPLE_RATE, subtype="PCM_16")
        (d / f"{ds}_{cut_id}.txt").write_text(text, encoding="utf-8")
        out.append(("ok_val" if split == "val" else "ok", dur))
    return out


def convert(ds: str, workers: int, start_shard: int = 0) -> dict:
    pat, tcol, tfallback = DATASETS[ds]
    files = sorted(f for f in glob.glob(str(DS_ROOT / pat)) if "incomplete" not in f)
    if not files:
        return {"error": f"no parquet: {pat}"}

    import pyarrow.parquet as pq

    seen = set()
    t0 = time.time()
    res = {"ok": 0, "ok_val": 0, "decode_fail": 0, "too_short": 0, "hours": 0.0,
           "files": len(files), "empty_text": 0, "dup": 0}
    # Xử lý TỪNG SHARD: chỉ giữ ~1 shard audio bytes trong RAM cùng lúc.
    # Bài học từ lần chạy đầu: gom toàn bộ task list (56GB ViMD) trước khi submit
    # làm RSS phình tới 50GB và sắp OOM — streaming per-shard là bắt buộc.
    # start_shard: bỏ qua N shard đầu (resume sau crash đĩa đầy). fi vẫn là index
    # toàn cục nên id dup-suffix khớp với run gốc — ghi đè tại chỗ, an toàn.
    with Pool(processes=workers, maxtasksperchild=16) as pool:
        for fi, fp in enumerate(files):
            if fi < start_shard:
                continue
            pf = pq.ParquetFile(fp)
            cols = [c for c in (tcol, tfallback, "audio")
                    if c and c in pf.schema_arrow.names]
            merged, buf = [], []
            for batch in pf.iter_batches(batch_size=32, columns=cols):
                d = batch.to_pydict()
                for r in range(batch.num_rows):
                    text = (d[tcol][r] or "").strip() if tcol in d else ""
                    if not text and tfallback and tfallback in d:
                        text = (d[tfallback][r] or "").strip()
                    if not text:
                        res["empty_text"] += 1
                        continue
                    a = d["audio"][r]
                    ab = a.get("bytes") if isinstance(a, dict) else None
                    if not ab:
                        res["empty_text"] += 1
                        continue
                    stem = Path(a.get("path") or f"r{fi}_{r}").stem
                    cid = _sanitize(stem)
                    if cid in seen:
                        res["dup"] += 1
                        cid = f"{cid}_{fi}_{r}"
                    seen.add(cid)
                    buf.append((cid, text, ab))
                    if len(buf) >= 32:
                        merged.append((ds, buf))
                        buf = []
            if buf:
                merged.append((ds, buf))

            for out in pool.imap_unordered(_work, merged, chunksize=1):
                for status, dur in out:
                    res[status] = res.get(status, 0) + 1
                    if status.startswith("ok"):
                        res["hours"] += dur / 3600
            print(f"  [{ds}] shard {fi+1}/{len(files)} — {res['ok']+res['ok_val']} cut, "
                  f"{res['hours']:.1f}h", flush=True)
    res["n_val"], res["n_train"] = res["ok_val"], res["ok"]
    res["ok"] = res["ok"] + res["ok_val"]
    res["sec"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", choices=list(DATASETS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--start-shard", type=int, default=0,
                    help="bỏ qua N shard đầu (resume; fi vẫn đếm toàn cục)")
    a = ap.parse_args()
    if a.list or not a.dataset:
        for k, (p, t, tf) in DATASETS.items():
            n = len([f for f in glob.glob(str(DS_ROOT / p)) if "incomplete" not in f])
            print(f"{k:<12} {n:>4} shards  text={t}" + (f" (+{tf})" if tf else ""))
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT_ROOT / "_audit_convert.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}

    print(f"{'dataset':<12} {'shards':>6} {'ok':>7} {'val':>6} {'rác':>10} "
          f"{'giờ':>7} {'phút':>5}")
    for ds in a.dataset:
        r = convert(ds, a.workers, start_shard=a.start_shard)
        audit[ds] = r
        if "error" in r:
            print(f"{ds:<12} LỖI: {r['error']}")
            continue
        junk = r.get("empty_text", 0) + r.get("decode_fail", 0) + r.get("too_short", 0)
        print(f"{ds:<12} {r['files']:>6} {r['ok']:>7} {r.get('n_val', 0):>6} "
              f"{junk:>10} {r['hours']:>7.2f} {r['sec']/60:>5.1f}")
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
