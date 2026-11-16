#!/usr/bin/env python3
"""Cắt cuộc gọi OneVoice thành từng turn (agent/customer) bằng energy-VAD.

Vì sao phải viết riêng: OneVoice chỉ có audio FULL cuộc gọi (25–178s, mono) và
danh sách turn KHÔNG có timestamp — chỉ có speaker + text + tts_text. Không cắt
thì không train được (prepare_data lọc cut > 30s).

Nguyên lý: TTS render tuần tự từng turn, giữa các turn có khoảng lặng sạch.
Nên số vùng tiếng tìm được bằng VAD phải khớp số turn:
  - n_vùng == n_turn  → gán tuần tự, biên = biên vùng tiếng.
  - n_vùng >  n_turn  → khoảng lặng BÊN TRONG một turn làm VAD tách dư; gộp dần
                        cặp vùng có khoảng lặng ngắn nhất cho tới khi khớp.
  - n_vùng <  n_turn  → có turn không sinh tiếng (hoặc VAD miss) — không gán
                        nổi tin cậy ⇒ BỎ cả cuộc gọi sang đống test audio dài.
Cắt thành công mới vào train; mọi cuộc gọi bỏ đi được giữ nguyên file làm
bộ test audio dài (theo rule của repo: phải đếm số từ trên audio dài).

Text dùng `tts_text` — đúng những gì TTS thực sự đọc, đã normalize ("alô vin
ai đi"). Guard chất lượng từng cut: 1,0–7,0 từ/giây (tiếng Việt nói thường
2–4 từ/s); cuộc gọi mất >30% cut vì guard ⇒ bỏ cả cuộc gọi (ngại lệch gán).

Output: raw_dataset/{train,val}/onevoice/onevoice_<call>_<tXX>.{wav,txt}
Split 95/5 deterministic theo md5 — cùng quy tắc convert_hf_datasets.py.
"""
import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
ONEVOICE = REPO / "datasets" / "OneVoice"
OUT_ROOT = REPO / "raw_dataset"
FULLCALL_DIR = REPO / "audio_test_onevoice"

SR = 16000
WIN, HOP = int(0.025 * SR), int(0.010 * SR)
THR_DB = 8.0          # ngưỡng tiếng = sàn nhiễu (p10) + 8 dB
MIN_VOICED = 0.25     # vùng tiếng ngắn hơn bị bỏ
CLOSE_GAP = 0.45      # khoảng lặng ngắn hơn bị lấp (nhờ đó tách turn thật)
PAD = 0.06            # đệm 2 bên biên khi cắt
WPS_LO, WPS_HI = 1.0, 7.0   # từ/giây hợp lệ
MAX_BAD_FRAC = 0.30   # bỏ cả call nếu >30% cut rớt guard


def vad_regions(x: np.ndarray, close_gap: float = CLOSE_GAP,
                min_voiced: float = MIN_VOICED):
    """[(start_s, end_s)] vùng tiếng sau khi lấp gap ngắn + bỏ vùng bé."""
    n = 1 + max(0, (len(x) - WIN) // HOP)
    if n < 2:
        return []
    fr = np.lib.stride_tricks.as_strided(
        x, shape=(n, WIN), strides=(x.strides[0] * HOP, x.strides[0]))
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-8)
    voiced = db > (np.percentile(db, 10) + THR_DB)
    regions, i = [], 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        regions.append([i * 0.010, (j - 1) * 0.010 + 0.025])
        i = j
    merged = []
    for r in regions:  # lấp gap
        if merged and r[0] - merged[-1][1] < close_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_voiced]


# Thang (close_gap, min_voiced) thử lần lượt khi VAD ra ÍT vùng hơn số turn.
# Càng xuống thấp càng tách được turn có khoảng lặng hẹp, nhưng cũng dễ tách
# nhịp thở trong 1 turn — guard words/s phía dưới bắt lỗi gán sai.
GAP_LADDER = ((0.45, 0.25), (0.30, 0.20), (0.18, 0.12), (0.10, 0.08))


def merge_to_n(regions, n_target):
    """Gộp cặp vùng có gap ngắn nhất cho tới khi còn n_target vùng."""
    regs = [list(r) for r in regions]
    while len(regs) > n_target:
        gaps = [regs[i + 1][0] - regs[i][1] for i in range(len(regs) - 1)]
        k = int(np.argmin(gaps))
        regs[k][1] = regs[k + 1][1]
        del regs[k + 1]
    return [(a, b) for a, b in regs]


def _is_val(cut_id: str) -> bool:
    return int(hashlib.md5(f"onevoice/{cut_id}".encode()).hexdigest(), 16) % 100 < 5


def _work(call):
    """1 call → (stats, n_cuts_ok)."""
    fname, turns = call
    wav = ONEVOICE / fname  # file_name đã chứa prefix "audio/agent_XXX/..."
    try:
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        x = x[:, 0]
    except Exception as e:
        return {"read_fail": 1}, 0
    if sr != SR:
        return {"wrong_sr": 1}, 0

    texts = [(t.get("tts_text") or "").strip() for t in turns]
    if not texts or not all(texts):
        return {"empty_turn": 1}, 0

    regions = []
    for gap, mv in GAP_LADDER:
        regions = vad_regions(x, close_gap=gap, min_voiced=mv)
        if len(regions) >= len(texts):
            break
    if len(regions) < len(texts):
        return {"fewer_regions": 1}, 0
    if len(regions) > len(texts):
        regions = merge_to_n(regions, len(texts))

    call_id = Path(fname).stem
    n_ok, n_bad = 0, 0
    for i, ((a, b), text) in enumerate(zip(regions, texts)):
        s = max(0, int((a - PAD) * SR))
        e = min(len(x), int((b + PAD) * SR))
        dur = (e - s) / SR
        nw = len(text.split())
        if nw == 0 or not (WPS_LO <= nw / dur <= WPS_HI):
            n_bad += 1
            continue
        cut_id = f"{call_id}_t{i:02d}"
        d = OUT_ROOT / ("val" if _is_val(cut_id) else "train") / "onevoice"
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"onevoice_{cut_id}.wav", x[s:e], SR, subtype="PCM_16")
        (d / f"onevoice_{cut_id}.txt").write_text(text, encoding="utf-8")
        n_ok += 1

    if n_bad > MAX_BAD_FRAC * (n_ok + n_bad):
        # nghi lệch gán — xoá cut đã ghi của call này, giữ nguyên file làm test dài
        for i in range(len(texts)):
            for split in ("train", "val"):
                f = OUT_ROOT / split / "onevoice" / f"onevoice_{call_id}_t{i:02d}.wav"
                f.unlink(missing_ok=True)
                f.with_suffix(".txt").unlink(missing_ok=True)
        FULLCALL_DIR.mkdir(parents=True, exist_ok=True)
        (FULLCALL_DIR / f"{call_id}.txt").write_text(
            "\n".join(texts), encoding="utf-8")
        return {"misassigned": 1}, 0
    return {"call_ok": 1}, n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="chỉ N call đầu (debug)")
    a = ap.parse_args()

    calls = []
    with open(ONEVOICE / "conversations.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            turns = row.get("turns")
            if isinstance(turns, str):
                turns = json.loads(turns)
            calls.append((row["file_name"], turns))
    if a.limit:
        calls = calls[:a.limit]
    print(f"{len(calls)} cuộc gọi")

    stats = {}
    with Pool(a.workers, maxtasksperchild=64) as pool:
        for st, n_ok in pool.imap_unordered(_work, calls, chunksize=8):
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v
            stats["cuts"] = stats.get("cuts", 0) + n_ok
            if (stats.get("call_ok", 0) + stats.get("misassigned", 0)) % 100 == 0:
                print(f"  {stats}", flush=True)
    print("KẾT QUẢ:", stats)


if __name__ == "__main__":
    main()
