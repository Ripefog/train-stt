# prepare_data_new.py
#!/usr/bin/env python3
"""
prepare_data_fast.py — Phiên bản tối ưu tốc độ tối đa cho H100 + multi-CPU

Cải tiến so với bản cũ:
  1. GPU H100: dùng torchaudio trên CUDA để tính Fbank — nhanh hơn CPU ~10-20x
  2. True multiprocessing: N worker process độc lập, mỗi process xử lý 1 shard riêng
  3. torch.set_num_threads(1): tránh thread contention
  4. Batch write: dùng lilcom/numpy array thay vì ghi file lẻ
  5. Prefetch: đọc wav song song với compute bằng ThreadPoolExecutor
  6. Progress: tqdm real-time trên từng shard

Fix:
  - Rebuild CutSet dùng đúng lhotse.features.base.Features import
  - Bỏ silent except để debug rõ lỗi
  - store_array dùng đúng key + value
  - channels field trong Features
  - duration/num_frames tính từ feature thực tế tránh lệch frame
  - supervision duration khớp với cut duration
"""

import argparse
import json
import math
import os
import re
import time
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, current_process
import multiprocessing as mp
import unicodedata
import lhotse
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.features.base import Features
from lhotse.features.io import LilcomChunkyWriter
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SPLIT_DATASETS = {"LSVSC", "VLSP2020", "fleurs"}
VAL_DATASETS   = {"val"}        # dataset 100% val, không liên quan SPLIT_DATASETS


# ── Bảng map từ/ký hiệu → chữ (word_mapping.json) ─────────────────────────────
# Toàn bộ convention nằm trong file JSON, không hardcode ở đây. Xem "_note" trong
# file đó để biết các con số đo được từ metadata.csv của FPT.
_MAPPING_PATHS = [
    os.environ.get("WORD_MAPPING", ""),
    "/app/word_mapping.json",
    # Vị trí hiện tại: config/ ở repo root (parent của scripts/)
    str(Path(__file__).resolve().parent.parent / "config" / "word_mapping.json"),
    # Tương thích vị trí cũ (trước khi dọn vào config/)
    str(Path(__file__).resolve().parent.parent / "word_mapping.json"),
    str(Path(__file__).resolve().parent / "word_mapping.json"),
]


def _load_word_mapping() -> dict:
    for p in _MAPPING_PATHS:
        if p and Path(p).is_file():
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            # tương thích format phẳng cũ: {"fpt": "ép pê tê", ...}
            if not any(k in m for k in ("words", "symbols", "digit_by_digit")):
                m = {"words": m}
            n_w = len([k for k in m.get("words", {}) if not k.startswith("_")])
            print(f"📖 word_mapping: {p} ({n_w} từ)")
            return m
    raise FileNotFoundError(
        "Không tìm thấy word_mapping.json. Đã thử:\n  "
        + "\n  ".join(p for p in _MAPPING_PATHS if p)
        + '\nMount vào container: -v "$(pwd)/config/word_mapping.json:/app/word_mapping.json:ro"'
    )


_MAP = _load_word_mapping()


def _strip_notes(d) -> dict:
    return {k: v for k, v in (d or {}).items() if not k.startswith("_")}


def _alt(keys) -> str:
    """Alternation regex, key dài đứng trước để 'vnđ' được thử trước 'đ'."""
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


_DIGIT_WORDS = _MAP.get(
    "digits", ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
)
_NW = _MAP.get("number_words", {})
_W_TEN   = _NW.get("ten", "mười")
_W_TENS  = _NW.get("tens", "mươi")
_W_HUND  = _NW.get("hundred", "trăm")
_W_THOU  = _NW.get("thousand", "nghìn")
_W_ZERO  = _NW.get("zero_filler", "lẻ")
_W_MOT   = _NW.get("one_after_tens", "mốt")
_W_TU    = _NW.get("four_after_tens", "tư")
_W_LAM   = _NW.get("five_after_ten", "lăm")
_W_DEC   = _NW.get("decimal_point", "phẩy")

_SYMBOLS   = _MAP.get("symbols", {})
_SYM_ANY   = _strip_notes(_SYMBOLS.get("anywhere"))
_SYM_ALNUM = _strip_notes(_SYMBOLS.get("between_alnum"))
_SYM_NUM   = _strip_notes(_SYMBOLS.get("after_number"))
_WORDS     = _strip_notes(_MAP.get("words"))

# tra cứu không phân biệt hoa thường
_SYM_ANY_L   = {k.lower(): v for k, v in _SYM_ANY.items()}
_SYM_ALNUM_L = {k.lower(): v for k, v in _SYM_ALNUM.items()}
_SYM_NUM_L   = {k.lower(): v for k, v in _SYM_NUM.items()}
_WORDS_L     = {k.lower(): v for k, v in _WORDS.items()}

_RE_SYM_ANY = re.compile(_alt(_SYM_ANY), re.IGNORECASE) if _SYM_ANY else None
# cần chữ/số ở CẢ HAI bên: "60/14" khớp, "/ ạ" không
_RE_SYM_ALNUM = (
    re.compile(r"(?<=\w)(" + _alt(_SYM_ALNUM) + r")(?=\w)", re.IGNORECASE)
    if _SYM_ALNUM else None
)
# cần chữ số ngay trước và KHÔNG phải chữ cái ngay sau: "50.000đ" khớp,
# "quận 4 đường" và "50.000 đồng" (đã viết chữ) thì không
_RE_SYM_NUM = (
    re.compile(r"(?<=\d)\s*(" + _alt(_SYM_NUM) + r")(?!\w)", re.IGNORECASE)
    if _SYM_NUM else None
)
# Biên là CHỮ CÁI, không dùng \b: '\b' không tồn tại giữa '5' và 'ghz' (cả hai đều là
# word char) nên "5GHz" sẽ trượt key 'ghz'. '[^\W\d_]' = ký tự chữ (không số, không '_').
_LETTER = r"[^\W\d_]"
_RE_WORDS = (
    re.compile(
        r"(?<!" + _LETTER + r")(" + _alt(_WORDS) + r")(?!" + _LETTER + r")",
        re.IGNORECASE,
    )
    if _WORDS else None
)

_DBD = _MAP.get("digit_by_digit", {})
_DBD_MIN  = int(_DBD.get("min_digits", 7))
_DBD_ZERO = bool(_DBD.get("leading_zero", True))
_RE_DBD   = [re.compile(p) for p in _DBD.get("patterns", [])]

_NUM_RE = re.compile(r"\d[\d.,]*\d|\d")


def _read_digits(s: str) -> str:
    """Đọc từng chữ số: dùng cho hotline, số điện thoại, số có '0' đứng đầu."""
    return " ".join(_DIGIT_WORDS[int(c)] for c in s)


def _read_group3(n: int, pad: bool = False) -> str:
    """0..999 → chữ. pad=True khi nhóm này đứng sau 'nghìn' (1005 → 'không trăm lẻ năm')."""
    tram, chuc, dvi = n // 100, (n // 10) % 10, n % 10
    out = []
    if tram or pad:
        out += [_DIGIT_WORDS[tram], _W_HUND]
    if chuc == 0 and dvi and (tram or pad):
        out.append(_W_ZERO)
    elif chuc == 1:
        out.append(_W_TEN)
    elif chuc >= 2:
        out += [_DIGIT_WORDS[chuc], _W_TENS]
    if dvi:
        if chuc >= 2 and dvi == 1:
            out.append(_W_MOT)
        elif chuc >= 2 and dvi == 4:
            out.append(_W_TU)
        elif chuc >= 1 and dvi == 5:
            out.append(_W_LAM)
        else:
            out.append(_DIGIT_WORDS[dvi])
    return " ".join(out)


def _read_cardinal(n: int) -> str:
    """0..999999. Số dài hơn đã bị _read_number_token đẩy sang đọc từng chữ số."""
    if n == 0:
        return _DIGIT_WORDS[0]
    nghin, rest = divmod(n, 1000)
    if nghin == 0:
        return _read_group3(rest)
    out = _read_group3(nghin) + " " + _W_THOU
    if rest:
        out += " " + _read_group3(rest, pad=True)
    return out


def _read_number_token(tok: str) -> str:
    # dấu phân cách nghìn: 50.000 / 50,000 → 50000
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", tok):
        tok = re.sub(r"[.,]", "", tok)
    # thập phân: 3,5 → "ba phẩy năm"
    m = re.fullmatch(r"(\d+)[.,](\d+)", tok)
    if m:
        return f"{_read_number_token(m.group(1))} {_W_DEC} {_read_digits(m.group(2))}"
    digits = re.sub(r"\D", "", tok)
    if not digits:
        return ""
    if (_DBD_ZERO and digits.startswith("0")) or len(digits) >= _DBD_MIN:
        return _read_digits(digits)
    return _read_cardinal(int(digits))


def apply_word_mapping(text: str) -> str:
    """Áp bảng map từ word_mapping.json, theo đúng thứ tự đã ghi trong "_note" của file."""
    # 0. Dấu phân cách nghìn bằng KHOẢNG TRẮNG: "3 000 đồng", "600 000 đ", "104 495 tỷ".
    #    Phải gộp ở đây vì _NUM_RE không quét qua khoảng trắng, nên "3 000" sẽ bị đọc
    #    thành "ba" + "không không không" thay vì "ba nghìn".
    #    Đo trên manifest: 0,82% cut public dùng dạng này; kiểm 14 mẫu thì cả 14 đều là
    #    số thật (theo sau là đồng/ha/tỷ/triệu/hộ dân), không có danh sách số rời.
    #    Không đụng hotline: "1900 6600" không khớp vì \d{1,3} không ăn được 4 chữ số đầu.
    text = re.sub(
        r"\b(\d{1,3})((?: \d{3})+)\b",
        lambda m: m.group(1) + m.group(2).replace(" ", ""),
        text,
    )
    # 1. hotline / số điện thoại → đọc từng chữ số (trước mọi bước khác, vì các
    #    pattern này nhìn nguyên cụm số kể cả khi bị khoảng trắng chia đôi)
    for rx in _RE_DBD:
        text = rx.sub(lambda m: f" {_read_digits(re.sub(r'[^0-9]', '', m.group(0)))} ", text)
    # 2. đơn vị dính sau số (đ / VNĐ) — phải trước bước đọc số, lúc chữ số còn đó
    if _RE_SYM_NUM:
        text = _RE_SYM_NUM.sub(lambda m: " " + _SYM_NUM_L[m.group(1).lower()], text)
    # 3. ký hiệu giữa chữ/số ('/' → sẹt), rồi ký hiệu mọi vị trí ('%' → phần trăm)
    if _RE_SYM_ALNUM:
        text = _RE_SYM_ALNUM.sub(lambda m: " " + _SYM_ALNUM_L[m.group(1).lower()] + " ", text)
    if _RE_SYM_ANY:
        text = _RE_SYM_ANY.sub(lambda m: " " + _SYM_ANY_L[m.group(0).lower()] + " ", text)
    # 4. map từ — phải trước bước đọc số, nếu không "5g" bị tách thành "năm g"
    if _RE_WORDS:
        text = _RE_WORDS.sub(lambda m: " " + _WORDS_L[m.group(1).lower()] + " ", text)
    # 5. số còn lại → đọc số đếm
    return _NUM_RE.sub(lambda m: f" {_read_number_token(m.group(0))} ", text)


def normalize_text(text: str) -> str:
    """NFC → word_mapping → lowercase → xóa dấu câu.

    NFC phải đi trước: chữ Việt mã hoá tổ hợp (NFD) có dấu là combining mark,
    mà '\\w' không khớp category Mn nên bước xóa dấu câu sẽ biến 'kết' → 'ke t'.
    apply_word_mapping phải đi trước bước xóa dấu câu, để '50.000' và '60/14'
    còn nguyên hình dạng lúc được map.
    """
    text = unicodedata.normalize("NFC", text)
    text = apply_word_mapping(text)
    text = text.lower()
    # Xóa mọi ký tự không phải chữ cái Unicode, chữ số, hoặc khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # \w bao gồm underscore — xóa nốt
    text = text.replace("_", " ")
    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text
TRAIN_RATIO    = 0.7
SAMPLE_RATE    = 16000
NUM_MEL_BINS   = 80
FRAME_SHIFT    = 0.01  # hop_length / sample_rate = 160 / 16000

# ── BAND_LIMIT_SR: mô phỏng băng tần hẹp của val ──────────────────────────────
# Hạ audio xuống BAND_LIMIT_SR Hz (⇒ low-pass ở Nyquist = BAND_LIMIT_SR/2) rồi đưa
# lại SAMPLE_RATE. 0 hoặc rỗng = tắt, giữ nguyên hành vi cũ.
#
# Tại sao cần: val là thoại call-center, nội dung phổ THẬT chỉ tới ~3 kHz (vách
# −70 dB trong khoảng 2→3,4 kHz — codec narrowband, còn hẹp hơn G.711). Sample rate
# 11.025 Hz của nó chỉ là vỏ chứa sau transcode. Trong khi data train có năng lượng
# trải đều cả băng. Đo trên fbank, bin 54–79 (>3 kHz) so với quy chiếu 200–800 Hz:
#
#     val −57,2 dB  |  FPT clean/noise +18…+20 dB  |  public +39…+53 dB
#
# BAND_LIMIT_SR=6000 (Nyquist 3 kHz) kéo cả 8 dataset về +4,7…+8,5 dB. Còn 11025
# (Nyquist 5512) chỉ đóng ~10/47 dB vì nó bỏ nguyên dải 3–5,5 kHz — mà đó là 17
# trong 26 bin lệch — và với FPT 8 kHz thì hơi PHẢN tác dụng (+18,5 → +20,0) vì
# không có gì trên 4 kHz để xoá, chỉ thêm gợn resampler.
#
# CHỈ ÁP CHO TRAIN. Val giữ nguyên đường gốc (11.025 Hz → 16 kHz) vì val là bản đại
# diện cho audio deploy thật — transform nó là đổi thước đo, và buộc inference phải áp
# cùng chuỗi resample. Đo thực: band-limit val làm nó XẤU đi (−63,7 → −58,4 dB) vì
# resampler thêm gợn vào dải cao, cao hơn cả sàn leakage tự nhiên của val.
# Việc chia train/val nằm ở `extract_parallel_gpu` qua `split_name`, không ở worker.
#
# ⚠️  Đổi giá trị này là đổi phân phối input của encoder ⇒ PHẢI extract lại fbank,
#     và fbank cũ/mới không trộn lẫn được. Ghi ra thư mục khác (xem run_new.sh).
# lowpass_filter_width của torchaudio.functional.resample. Mặc định của torchaudio là 6 —
# filter quá ngắn, stopband không sâu, và ĐÓ là toàn bộ phần dư khi khớp val. Đo trên
# viet_bud500, bin 54–79 so với val (−65,1 dB): width 6 → +10,2 dB | 16 → −2,8 |
# 32 → −5,6 | 64 → −6,4 | 128 → −6,3 (bão hoà). Chọn 16 vì khớp val sát nhất; rộng hơn
# thì cắt sâu hơn cả codec của val. Không phải chỗ nên tinh chỉnh thêm — chênh vài dB ở
# đây nhỏ hơn nhiều so với 47 dB ban đầu.
BAND_LIMIT_FILTER_WIDTH = int(os.environ.get("BAND_LIMIT_FILTER_WIDTH", "16") or 16)

_bl_raw = (os.environ.get("BAND_LIMIT_SR", "") or "0").strip()
try:
    BAND_LIMIT_SR = int(float(_bl_raw))
except ValueError:
    raise ValueError(f"BAND_LIMIT_SR không phải số: {_bl_raw!r}")
if BAND_LIMIT_SR:
    if BAND_LIMIT_SR < 2000:
        raise ValueError(f"BAND_LIMIT_SR={BAND_LIMIT_SR} quá thấp (Nyquist < 1 kHz), gần như chắc là gõ sai")
    if BAND_LIMIT_SR >= SAMPLE_RATE:
        raise ValueError(
            f"BAND_LIMIT_SR={BAND_LIMIT_SR} >= SAMPLE_RATE={SAMPLE_RATE} — không cắt được gì, "
            f"chỉ thêm 2 lần resample vô nghĩa"
        )
    print(f"🎚️  BAND_LIMIT_SR={BAND_LIMIT_SR} Hz → low-pass ở {BAND_LIMIT_SR/2:.0f} Hz — "
          f"CHỈ tập TRAIN (val giữ nguyên); trong train áp cho mọi dataset, "
          f"kể cả dataset đã {SAMPLE_RATE} Hz")
    print(f"    lowpass_filter_width={BAND_LIMIT_FILTER_WIDTH}")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: SCAN & BUILD CUTS (CPU, multi-thread)
# ══════════════════════════════════════════════════════════════════════════════

def _build_cut(wav_file: Path):
    txt_file = wav_file.with_suffix(".txt")
    if not txt_file.exists():
        return None
    try:
        text = normalize_text(txt_file.read_text(encoding="utf-8"))
        if not text:
            return None
        recording = Recording.from_file(wav_file)
        supervision = SupervisionSegment(
            id=wav_file.stem,
            recording_id=wav_file.stem,
            start=0.0,
            duration=recording.duration,
            text=text,
        )
        return MonoCut(
            id=wav_file.stem,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
            supervisions=[supervision],
        )
    except Exception as e:
        print(f"  ⚠️ _build_cut failed for {wav_file}: {e}")
        return None


def scan_dataset_parallel(source_dir: Path, num_threads: int = 16,
                          test_mode: bool = False, test_samples: int = 10) -> tuple[list, int]:
    wav_files = sorted(source_dir.glob("**/*.wav"))
    
    # ✅ Giới hạn ngay từ đầu, trước khi submit bất kỳ future nào
    if test_mode:
        wav_files = wav_files[:test_samples]  # lấy đúng số lượng, không cần *3
    
    cuts, skipped = [], 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(_build_cut, f): f for f in wav_files}
        for fut in tqdm(as_completed(futures), total=len(wav_files),
                        desc=f"  Scanning {source_dir.name}", leave=False):
            result = fut.result()
            if result is not None:
                cuts.append(result)
            else:
                skipped += 1
    # không cần break nữa vì đã giới hạn wav_files từ đầu

    return cuts, skipped

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: WORKER — mỗi process xử lý 1 shard độc lập
# ══════════════════════════════════════════════════════════════════════════════

def _worker_extract_shard(args):
    """
    Worker function chạy trong process riêng.
    Dùng GPU nếu có, fallback về CPU.
    """
    shard_idx, cut_dicts, storage_dir, use_gpu, gpu_id, batch_size, band_limit_sr = args

    # ── Setup thread / device ──────────────────────────────────────────────
    torch.set_num_threads(1)  # QUAN TRỌNG: tránh thread explosion

    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # ── Rebuild cuts từ dict (pickling lhotse cuts an toàn hơn) ──────────
    cuts = []
    for d in cut_dicts:
        try:
            cut = MonoCut.from_dict(d)
            cuts.append(cut)
        except Exception as e:
            print(f"  ⚠️ [shard {shard_idx}] from_dict failed: {e}")

    if not cuts:
        return []

    # ── Kaldi fbank — dùng kaldifeat (GPU) nếu có, fallback compliance.kaldi (CPU)
    # Parameters giống hệt inference_gipformer_pt.py
    try:
        import kaldifeat
        opts = kaldifeat.FbankOptions()
        opts.device = device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.mel_opts.num_bins = NUM_MEL_BINS
        opts.mel_opts.high_freq = -400
        fbank_extractor = kaldifeat.Fbank(opts)
        use_kaldifeat = True
    except ImportError:
        use_kaldifeat = False
        _FBANK_KWARGS = dict(
            num_mel_bins=NUM_MEL_BINS,
            high_freq=-400,
            snip_edges=False,
            dither=0.0,
            sample_frequency=SAMPLE_RATE,
        )

    shard_name = f"shard_{shard_idx:05d}"
    feat_path  = storage_dir / f"{shard_name}.lca"
    manifest_rows = []

    with LilcomChunkyWriter(feat_path) as writer:
        for i in range(0, len(cuts), batch_size):
            batch_cuts = cuts[i : i + batch_size]
            for cut in batch_cuts:
                try:
                    # Đọc audio (dùng soundfile — torchaudio.load() ở torchaudio>=2.10
                    # yêu cầu package torchcodec, chưa cài trong image này)
                    wav_path = cut.recording.sources[0].source
                    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
                    waveform = torch.from_numpy(data.T)

                    # Resample. Với BAND_LIMIT_SR, đi native → BAND_LIMIT_SR → SAMPLE_RATE:
                    # bước 1 là chỗ low-pass thật (anti-alias filter của resampler cắt ở
                    # Nyquist mới), bước 2 chỉ đưa lại 16k mà KHÔNG phục hồi được dải đã mất.
                    # Phải nằm NGOÀI nhánh `sr != SAMPLE_RATE`: viet_bud500, infore2_audiobooks,
                    # LSVSC, VLSP2020 vốn đã 16k nên không bao giờ vào nhánh đó — 4 dataset
                    # lớn nhất sẽ bị bỏ qua mà không có lỗi nào báo.
                    # band_limit_sr do process cha truyền vào (0 cho val) — KHÔNG đọc
                    # BAND_LIMIT_SR toàn cục ở đây, nếu không val cũng bị cắt băng.
                    if band_limit_sr:
                        _w = BAND_LIMIT_FILTER_WIDTH
                        if sr != band_limit_sr:
                            waveform = torchaudio.functional.resample(
                                waveform, sr, band_limit_sr, lowpass_filter_width=_w)
                        waveform = torchaudio.functional.resample(
                            waveform, band_limit_sr, SAMPLE_RATE, lowpass_filter_width=_w)
                    elif sr != SAMPLE_RATE:
                        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

                    # Tính Kaldi fbank: kaldifeat trên GPU nếu có, fallback CPU
                    with torch.no_grad():
                        if use_kaldifeat:
                            feats = fbank_extractor(
                                [waveform[0].contiguous().to(device)]
                            )[0].cpu().numpy()                      # (T, n_mels)
                        else:
                            feats = torchaudio.compliance.kaldi.fbank(
                                waveform[:1], **_FBANK_KWARGS
                            ).numpy()                               # (T, n_mels)

                    # FIX: tính duration từ num_frames thực tế để tránh lệch
                    actual_num_frames = feats.shape[0]
                    actual_duration = round(actual_num_frames * FRAME_SHIFT, ndigits=8)

                    # Lưu vào LilcomChunky store — store_array trả về Array object
                    array_obj = writer.store_array(key=cut.id, value=feats)
                    feat_key  = array_obj.storage_key   # ← string "offset,size1,size2,..."
                    feat_path_actual = array_obj.storage_path  # lấy path thực từ object

                    manifest_rows.append({
                        "cut_id":       cut.id,
                        "feat_key":     feat_key,            # string "579568,58265,..."
                        "feat_path":    feat_path_actual,    # path từ Array object
                        "num_frames":   actual_num_frames,
                        "num_features": feats.shape[1],
                        "duration":     actual_duration,
                        "start":        cut.start,
                        "channel":      cut.channel,
                        "recording":    cut.recording.to_dict(),
                        "sup_id":       cut.supervisions[0].id if cut.supervisions else cut.id,
                        "sup_text":     cut.supervisions[0].text if cut.supervisions else "",
                        "sup_speaker":  cut.supervisions[0].speaker if cut.supervisions else None,
                        "sup_language": cut.supervisions[0].language if cut.supervisions else None,
                    })
                except Exception as e:
                    print(f"  ⚠️ [shard {shard_idx}] feature extract failed for {cut.id}: {e}")

    return manifest_rows


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3: MAIN — phân shard + dispatch workers
# ══════════════════════════════════════════════════════════════════════════════

def extract_parallel_gpu(
    cuts_list: list,
    split_name: str,
    output_path: Path,
    num_workers: int = 8,
    num_gpus: int = 1,
    batch_size: int = 32,
):
    """
    Chia cuts thành num_workers shard, mỗi shard chạy trên 1 process.
    GPU được round-robin giữa các worker.
    """
    total = len(cuts_list)
    if total == 0:
        return CutSet.from_cuts([])

    storage_dir = output_path / f"feats_{split_name}"
    storage_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available() and num_gpus > 0
    device_label = f"GPU (cuda x{num_gpus})" if use_gpu else "CPU"
    print(f"\n⚙️  [{split_name.upper()}] {total} cuts → {num_workers} workers trên {device_label}")

    # Serialize cuts thành dict để truyền qua multiprocessing
    cuts_dicts = [c.to_dict() for c in tqdm(cuts_list, desc="  Serializing cuts", leave=False)]

    # Chia shard
    shard_size = math.ceil(total / num_workers)
    shards = [cuts_dicts[i : i + shard_size] for i in range(0, total, shard_size)]

    # BAND_LIMIT_SR chỉ áp cho TRAIN. Val phải giữ đúng đường xử lý gốc
    # (11.025 Hz → 16 kHz), vì val là bản đại diện cho audio deploy thật: đổi nó là
    # đổi luôn thước đo, và bắt inference phải áp cùng chuỗi resample mới khớp.
    _split_bl = BAND_LIMIT_SR if split_name == "train" else 0
    if BAND_LIMIT_SR and split_name != "train":
        print(f"   ⏭️  [{split_name}] BỎ QUA band limit — giữ nguyên đường resample gốc")
    worker_args = [
        (shard_idx, shard, str(storage_dir), use_gpu, shard_idx % max(num_gpus, 1),
         batch_size, _split_bl)
        for shard_idx, shard in enumerate(shards)
    ]

    t0 = time.time()
    all_rows = []

    with Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for rows in tqdm(
            pool.imap_unordered(_worker_extract_shard, worker_args),
            total=len(shards),
            desc=f"  [{split_name}] shards",
        ):
            all_rows.extend(rows)

    elapsed = time.time() - t0
    print(f"  ✅ Xong {total} cuts trong {elapsed:.1f}s ({total/elapsed:.0f} cuts/s)")
    print(f"  📦 Số rows thu được: {len(all_rows)}")

    # ── Rebuild CutSet từ manifest rows ───────────────────────────────────
    rebuilt_cuts = []
    rebuild_errors = 0
    for row in tqdm(all_rows, desc="  Rebuilding CutSet", leave=False):
        try:
            recording = Recording.from_dict(row["recording"])

            # FIX: supervision duration phải khớp với cut duration (từ feature)
            supervision = SupervisionSegment(
                id=row["sup_id"],
                recording_id=recording.id,
                start=row["start"],
                duration=row["duration"],   # ← khớp với actual_duration
                text=row["sup_text"],
                speaker=row["sup_speaker"],
                language=row["sup_language"],
            )

            # FIX: channels + duration từ feature thực tế
            feats = Features(
                type="kaldi-fbank",
                num_frames=row["num_frames"],
                num_features=row["num_features"],
                frame_shift=FRAME_SHIFT,
                sampling_rate=SAMPLE_RATE,
                start=0.0,
                duration=row["duration"],   # ← actual_duration
                storage_type="lilcom_chunky",
                storage_path=row["feat_path"],
                storage_key=row["feat_key"],
                channels=row["channel"],    # ← FIX: phải khớp với cut.channel
            )

            cut = MonoCut(
                id=row["cut_id"],
                start=row["start"],
                duration=row["duration"],   # ← actual_duration
                channel=row["channel"],
                recording=recording,
                supervisions=[supervision],
                features=feats,
            )
            rebuilt_cuts.append(cut)
        except Exception as e:
            rebuild_errors += 1
            print(f"  ⚠️ Rebuild failed for {row.get('cut_id', '?')}: {e}")

    if rebuild_errors:
        print(f"  ⚠️ Rebuild errors: {rebuild_errors}/{len(all_rows)}")

    return CutSet.from_cuts(rebuilt_cuts)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4: ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _backup_output(output_path: Path, n_train_cuts: int) -> None:
    """Copy toàn bộ output sang <output_dir>_save ngay sau khi extract xong.

    Extract 2.3k giờ mất ~15h nên phải có bản lưu. Ba điều bắt buộc:

    1. Phải COPY THẬT, không hard link. `to_file()` và `LilcomChunkyWriter` đều mở
       file bằng open(path,'wb') → truncate TẠI CHỖ, giữ nguyên inode. Hard link
       dùng chung inode nên run sau sẽ phá luôn bản lưu (đã test xác nhận).
    2. KHÔNG ghi đè bản lưu lớn hơn. Đây chính là kịch bản đã làm mất 195h fbank:
       một run TEST_MODE với 3000 cut ghi đè lên bản đầy đủ. Nên so số cut trước.
    3. Kiểm dung lượng trống trước khi copy, để không lấp đầy ổ giữa lúc train.

    Tắt bằng SAVE_FBANK=false.
    """
    if os.environ.get("SAVE_FBANK", "true").lower() != "true":
        print("⏭️  SAVE_FBANK=false — bỏ qua backup")
        return

    import shutil

    save_path = output_path.parent / f"{output_path.name}_save"
    manifest = "librispeech_cuts_train-clean-100.jsonl.gz"

    # (2) bản lưu hiện có nhiều cut hơn thì KHÔNG đè
    old_manifest = save_path / manifest
    if old_manifest.is_file():
        try:
            import gzip
            with gzip.open(old_manifest, "rt", encoding="utf-8") as f:
                n_old = sum(1 for _ in f)
        except Exception as e:
            print(f"⚠️  Không đọc được manifest của bản lưu ({e}) — bỏ qua backup cho an toàn")
            return
        if n_old > n_train_cuts:
            print(f"🛑 GIỮ NGUYÊN bản lưu: {save_path} đang có {n_old} cut > {n_train_cuts} cut vừa tạo.")
            print(f"   Muốn đè thì tự xoá {save_path} rồi chạy lại.")
            return
        print(f"♻️  Bản lưu cũ có {n_old} cut ≤ {n_train_cuts} — sẽ thay thế")
        shutil.rmtree(save_path)

    # (3) đủ chỗ chưa
    src_bytes = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    free = shutil.disk_usage(output_path.parent).free
    print(f"\n💾 Backup → {save_path}  ({src_bytes/1e9:.1f} GB, ổ còn trống {free/1e9:.1f} GB)")
    if free < src_bytes * 1.05:
        print(f"🛑 KHÔNG ĐỦ CHỖ — cần {src_bytes*1.05/1e9:.1f} GB, bỏ qua backup")
        return

    t0 = time.time()
    shutil.copytree(output_path, save_path)
    dst_bytes = sum(f.stat().st_size for f in save_path.rglob("*") if f.is_file())
    ok = dst_bytes == src_bytes
    print(f"{'✅' if ok else '🛑'} Backup {'xong' if ok else 'LỆCH BYTE'} trong {time.time()-t0:.0f}s "
          f"— {dst_bytes/1e9:.1f} GB / {src_bytes/1e9:.1f} GB")


def prepare_data(
    data_dir: str,
    output_dir: str,
    num_workers: int = 8,
    num_gpus: int = 1,
    batch_size: int = 32,
    scan_threads: int = 16,
    test_mode: bool = False,
    test_samples: int = 10,
):
    data_path   = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if test_mode:
        print(f"\n⚠️  TEST MODE: giới hạn tối đa {test_samples} file mỗi dataset")

    # ── Scan ──────────────────────────────────────────────────────────────
    sub_dirs = sorted([p for p in data_path.iterdir() if p.is_dir()])
    if not sub_dirs:
        sub_dirs_map = {"__root__": (data_path, "train_only")}
    else:
        def _split_type(name):
            if name in SPLIT_DATASETS:
                return "split"
            if name in VAL_DATASETS:
                return "val_only"
            return "train_only"

        sub_dirs_map = {d.name: (d, _split_type(d.name)) for d in sub_dirs}

    print(f"📊 Split 70/30 : {[k for k,(_,t) in sub_dirs_map.items() if t=='split']}")
    print(f"📊 100% Val    : {[k for k,(_,t) in sub_dirs_map.items() if t=='val_only']}")
    print(f"📊 100% Train  : {[k for k,(_,t) in sub_dirs_map.items() if t=='train_only']}")

    all_train_cuts, all_val_cuts = [], []

    for name, (src_dir, split_type) in sub_dirs_map.items():
        print(f"\n🔍 [{name}] scanning {src_dir}, test_mode={test_mode}")
        t0 = time.time()
        cuts, skipped = scan_dataset_parallel(
            src_dir, num_threads=scan_threads,
            test_mode=test_mode, test_samples=test_samples,
        )
        print(f"   {len(cuts)} hợp lệ, bỏ qua {skipped} — {time.time()-t0:.1f}s")

        if not cuts:
            continue

        if split_type == "split":
            cuts_sorted = sorted(cuts, key=lambda c: c.id)
            n_train = int(len(cuts_sorted) * TRAIN_RATIO)
            all_train_cuts.extend(cuts_sorted[:n_train])
            all_val_cuts.extend(cuts_sorted[n_train:])
            print(f"   → Train: {n_train} | Val: {len(cuts_sorted)-n_train}")
        elif split_type == "val_only":
            all_val_cuts.extend(cuts)
            print(f"   → Val: {len(cuts)}")
        else:
            all_train_cuts.extend(cuts)
            print(f"   → Train: {len(cuts)}")

    print(f"\n✅ Tổng: train={len(all_train_cuts)}, val={len(all_val_cuts)}")
    if not all_train_cuts:
        raise RuntimeError("Không có dữ liệu train!")

    # ── Extract features ──────────────────────────────────────────────────
    train_cut_set = extract_parallel_gpu(
        all_train_cuts, "train", output_path,
        num_workers=num_workers, num_gpus=num_gpus, batch_size=batch_size,
    )

    if all_val_cuts:
        val_cut_set = extract_parallel_gpu(
            all_val_cuts, "val", output_path,
            num_workers=num_workers, num_gpus=num_gpus, batch_size=batch_size,
        )
    else:
        print("\n⚠️  Không có val set — dùng 5% train")
        n_val = max(1, int(len(train_cut_set) * 0.05))
        val_cut_set   = train_cut_set.subset(first=n_val)
        train_cut_set = train_cut_set.subset(last=len(train_cut_set) - n_val)

    # ── Lưu manifests ─────────────────────────────────────────────────────
    print("\n💾 Lưu manifests...")
    train_cut_set.to_file(output_path / "librispeech_cuts_train-clean-100.jsonl.gz")
    val_cut_set.to_file(output_path / "librispeech_cuts_dev-clean.jsonl.gz")
    val_cut_set.to_file(output_path / "librispeech_cuts_dev-other.jsonl.gz")
    val_cut_set.to_file(output_path / "librispeech_cuts_test-clean.jsonl.gz")
    val_cut_set.to_file(output_path / "librispeech_cuts_test-other.jsonl.gz")

    # Dấu vết cấu hình feature: fbank full-band và fbank band-limited nhìn giống nhau
    # hoàn toàn từ bên ngoài (cùng tên file, cùng số cut, kích thước xấp xỉ). Trộn lẫn
    # hai bản khi train là làm hại convergence mà không có lỗi nào báo.
    with open(output_path / "FEATURE_CONFIG.txt", "w", encoding="utf-8") as f:
        f.write(f"SAMPLE_RATE={SAMPLE_RATE}\n")
        f.write(f"NUM_MEL_BINS={NUM_MEL_BINS}\n")
        f.write(f"BAND_LIMIT_SR={BAND_LIMIT_SR}\n")
        f.write(f"band_limit_hz={BAND_LIMIT_SR/2 if BAND_LIMIT_SR else 'none (full-band)'}\n")
        f.write(f"band_limit_applied_to={'train only (val giu nguyen 11025->16k)' if BAND_LIMIT_SR else 'none'}\n")
        f.write(f"BAND_LIMIT_FILTER_WIDTH={BAND_LIMIT_FILTER_WIDTH}\n")
        f.write(f"train_cuts={len(train_cut_set)}\nval_cuts={len(val_cut_set)}\n")
        f.write(f"created={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n🎉 Hoàn thành!")
    print(f"   Train: {len(train_cut_set)} cuts")
    print(f"   Val  : {len(val_cut_set)} cuts")
    print(f"   Output: {output_path}")

    _backup_output(output_path, len(train_cut_set))

    if "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"]:
        try:
            import wandb
            if wandb.run is None:
                wandb.init(project=os.environ.get("WANDB_PROJECT", "data-prep"))
            wandb.log({
                "data_prep/train_cuts": len(train_cut_set),
                "data_prep/val_cuts":   len(val_cut_set),
            })
            wandb.finish()
        except Exception as e:
            print(f"⚠️ Wandb logging failed: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # QUAN TRỌNG cho CUDA + multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",     required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--num-workers",  type=int, default=8,
                        help="Số process song song (khuyến nghị = số CPU cores / 2)")
    parser.add_argument("--num-gpus",     type=int, default=1,
                        help="Số GPU dùng (H100=1, nếu multi-GPU tăng lên)")
    parser.add_argument("--batch-size",   type=int, default=1024,
                        help="Số file xử lý cùng lúc trong 1 GPU batch")
    parser.add_argument("--scan-threads", type=int, default=32,
                        help="Số thread để scan metadata wav files")
    parser.add_argument("--test-mode", action="store_true", default=False,
                        help="Giới hạn số file mỗi dataset để test nhanh pipeline")
    parser.add_argument("--test-samples", type=int, default=10,
                        help="Số file tối đa mỗi dataset khi test_mode=true")
    args = parser.parse_args()

    t0 = time.time()
    prepare_data(
        data_dir     = args.data_dir,
        output_dir   = args.output_dir,
        num_workers  = args.num_workers,
        num_gpus     = args.num_gpus,
        batch_size   = args.batch_size,
        scan_threads = args.scan_threads,
        test_mode    = args.test_mode,
        test_samples = args.test_samples,
    )
    print(f"\n🚀 Tổng thời gian: {(time.time()-t0)/3600:.2f} giờ")