#!/usr/bin/env python3
"""Convert VIVOS (tar đã extract) → format repo. Chạy SAU khi extract tar.gz.

VIVOS: {train,test}/prompts.txt (id<TAB>text IN HOA) + waves/<spk>/<id>.wav.
Gộp cả 2 split vào một pool rồi chia 95/5 theo md5 — cùng quy tắc
convert_hf_datasets.py. Text ghi nguyên bản (HOA) — prepare_data lowercase.
"""
import hashlib
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "datasets" / "_vivos_extracted" / "vivos"
OUT_ROOT = REPO / "raw_dataset"
SR = 16000


def main():
    prompts = {}
    for split in ("train", "test"):
        p = SRC / split / "prompts.txt"
        if not p.is_file():
            print(f"thiếu {p}"); sys.exit(1)
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            # separator là KHOẢNG TRẮNG (không phải tab) — tách id ở token đầu
            cid, _, text = ln.partition(" ")
            cid, text = cid.strip(), text.strip()
            if cid and text:
                prompts[cid] = text

    n_ok = n_val = n_miss = 0
    for wav in sorted(SRC.glob("*/waves/*/*.wav")):
        cid = wav.stem
        text = prompts.get(cid)
        if not text:
            n_miss += 1
            continue
        split = ("val" if int(hashlib.md5(f"vivos/{cid}".encode()).hexdigest(), 16)
                 % 100 < 5 else "train")
        d = OUT_ROOT / split / "vivos"
        d.mkdir(parents=True, exist_ok=True)
        data, sr = sf.read(wav, dtype="float32", always_2d=True)
        data = data[:, 0]
        if sr != SR:
            from scipy.signal import resample_poly
            data = resample_poly(data, SR, sr).astype("float32")
        sf.write(d / f"vivos_{cid}.wav", data, SR, subtype="PCM_16")
        (d / f"vivos_{cid}.txt").write_text(text, encoding="utf-8")
        n_val += split == "val"
        n_ok += 1
    print(f"vivos: {n_ok} cut ({n_val} val, {n_miss} thiếu transcript)")


if __name__ == "__main__":
    main()
