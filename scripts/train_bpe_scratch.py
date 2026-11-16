#!/usr/bin/env python3
"""Train BPE MỚI (2000 piece) trên transcript đã normalize của raw_dataset.

Từ đầu = không còn ràng buộc "đổi BPE hỏng checkpoint" (CLAUDE.md). BPE cũ bằm
nát thuật ngữ hơn từ thường 47% (`pách két` → 7 mảnh); nay train trên chính text
domain nên thuật ngữ ăn piece gọn hơn ngay từ đầu.

Ba việc:
  1. Gom mọi .txt của raw_dataset/{train,val} — đủ cả val vì BPE là thuộc tính
     của model, không được để val có token <unk> ngoài ý muốn.
  2. Normalize bằng ĐÚNG normalize_text của prepare_data.py (NFC → word_mapping
     → đọc số → lowercase → bỏ dấu câu) — BPE phải thấy đúng chuỗi sẽ được encode
     lúc train. word_mapping.json nằm ở repo root nên import được.
  3. sentencepiece BPE, vocab 2000, character_coverage 1.0 (tiếng Việt đủ bộ
     unicode), user_defined_symbols ["<blk>", "<sos/eos>"] — train.py tra
     piece_to_id của đúng 2 tên này.

Cổng kiểm sau khi train: <blk>/<sos/eos> tồn tại, encode→decode roundtrip,
in 30 piece chứa "ép"/"qua"/"phai". Chỉ khi pass mới copy vào lang_bpe_scratch/
(entrypoint sẽ tự sinh tokens.txt từ bpe.model).
"""
import argparse
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# prepare_data.py import lhotse/torchaudio/tqdm ở cấp module — venv này không có.
# Stub chúng để LẤY ĐÚNG normalize_text của prepare_data (không copy code, khỏi
# lệch pipeline). Các stub không bao giờ được dùng: normalize_text chỉ chạy re/unicodedata.
for _m, _attrs in {
    "lhotse": ["CutSet", "MonoCut", "Recording", "SupervisionSegment"],
    "lhotse.features": ["base"],
    "lhotse.features.base": ["Features"],
    "lhotse.features.io": ["LilcomChunkyWriter"],
    "torchaudio": ["functional"],
    "torchaudio.functional": ["resample"],
    "tqdm": ["tqdm"],
}.items():
    mod = sys.modules.get(_m) or types.ModuleType(_m)
    for _a in _attrs:
        if not hasattr(mod, _a):
            if _a in sys.modules:      # "base"/"io" là submodules, không phải attr
                setattr(mod, _a, sys.modules[_a])
            else:
                setattr(mod, _a, type(_a, (), {}))
    sys.modules[_m] = mod
sys.modules["lhotse.features"].base = sys.modules["lhotse.features.base"]

from prepare_data import normalize_text  # noqa: E402  (đúng pipeline, không copy code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(REPO / "raw_dataset"))
    ap.add_argument("--outdir", default=str(REPO / "lang_bpe_scratch"))
    ap.add_argument("--vocab", type=int, default=2000)
    a = ap.parse_args()

    import sentencepiece as spm

    texts = []
    n = 0
    for split in ("train", "val"):
        for f in sorted(Path(a.raw, split).glob("*/*.txt")):
            texts.append(normalize_text(f.read_text(encoding="utf-8")))
            n += 1
            if n % 50000 == 0:
                print(f"  đọc {n} transcript", flush=True)
    texts = [t for t in texts if t]
    print(f"{len(texts)} transcript → train BPE vocab={a.vocab}")

    # SentencePiece đọc từ FILE (input=list file), không nhận list chuỗi trực tiếp
    # ở mọi phiên bản — ghi ra file tạm cho chắc.
    inp = Path(a.raw) / "_bpe_input.txt"
    inp.write_text("\n".join(texts), encoding="utf-8")

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # ĐÚNG convention của icefall (local/train_bpe_model.py): <blk>=0, <sos/eos>=1,
    # <unk>=2, KHÔNG có <s>/</s> (bos_id=eos_id=-1), model_type=unigram.
    # Critical vì export_onnx.py hardcode BLANK_ID=0 và sherpa-onnx cũng giả định
    # blank ở id 0 — lệch id là hỏng toàn bộ đường deploy.
    spm.SentencePieceTrainer.train(
        input=str(inp),
        vocab_size=a.vocab,
        model_type="unigram",
        character_coverage=1.0,
        normalization_rule_name="identity",
        user_defined_symbols=["<blk>", "<sos/eos>"],
        unk_id=2,
        bos_id=-1,
        eos_id=-1,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
        model_prefix=str(out / "bpe"),
    )
    inp.unlink()

    # ── Cổng kiểm ──────────────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.load(str(out / "bpe.model"))
    blk, sos = sp.piece_to_id("<blk>"), sp.piece_to_id("<sos/eos>")
    assert blk == 0 and sos == 1, (
        f"SAI convention icefall: <blk>={blk} (phải 0), <sos/eos>={sos} (phải 1) "
        f"— export_onnx.py và sherpa-onnx sẽ hỏng"
    )
    assert sp.unk_id() == 2, f"unk_id={sp.unk_id()} (phải 2)"
    for probe in ("ép pi ti", "quai phai", "mô đem", "bệnh nhân"):
        pieces = sp.encode(probe, out_type=str)
        back = sp.decode(sp.encode(probe))
        print(f"  {probe!r}: {' '.join(pieces)}  → roundtrip {'OK' if back == probe else 'SAI: ' + back!r}")
    terms = [p for p in [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
             if any(s in p for s in ("ép", "qua", "phai", "đem"))]
    print(f"  piece chứa mảnh thuật ngữ: {' '.join(terms[:30])}")
    print(f"✅ {out}/bpe.model — vocab {sp.get_piece_size()}, <blk>={blk}, <sos/eos>={sos}")


if __name__ == "__main__":
    main()
