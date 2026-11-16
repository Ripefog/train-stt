#!/usr/bin/env python3
"""In kiến trúc zipformer theo từng lớp + số param của từng lớp.

CHỈ dựng model trong bộ nhớ, không đọc data, không cần GPU. Chạy được cả khi không có
checkpoint — nhưng truyền --ckpt thì nó kiểm luôn key có khớp không (bắt sớm lỗi lệch
vocab/kiến trúc, thứ mà load_state_dict báo rất khó đọc).

Dùng ĐÚNG train.get_model() nên số param khớp tuyệt đối với lúc train.

--trainable-modules / --freeze-encoder mô phỏng lại logic freeze ở train.py:1710-1718 để
xem TRƯỚC một phạm vi sẽ train bao nhiêu param, khỏi phải chạy thật mới biết.

Chạy trong container (xem lệnh ở cuối file).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

import train as T  # noqa: E402  (train.py đã được copy vào cwd)


def get_args():
    p = T.get_parser()
    p.add_argument("--ckpt", type=str, default="",
                   help="tuỳ chọn: nạp thử checkpoint để kiểm key có khớp")
    p.add_argument("--depth", type=int, default=3,
                   help="độ sâu cây module in ra (0 = chỉ top-level)")
    p.add_argument("--vocab-size", type=int, default=0,
                   help="dùng khi không có bpe.model; 0 = đọc từ --bpe-model")
    p.add_argument("--show-params", action="store_true",
                   help="in từng tensor param kèm shape")
    p.add_argument("--min-params", type=int, default=0,
                   help="bỏ qua module có ít hơn N param cho gọn")
    return p.parse_args()


def fmt(n):
    return f"{n:,}".replace(",", ".")


def main():
    args = get_args()
    params = T.get_params()
    params.update(vars(args))

    if params.vocab_size and params.vocab_size > 0 and not Path(params.bpe_model).is_file():
        params.blank_id = 0
        params.sos_id = params.eos_id = 1
        src = f"--vocab-size {params.vocab_size}"
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(params.bpe_model)
        params.blank_id = sp.piece_to_id("<blk>")
        params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
        params.vocab_size = sp.get_piece_size()
        src = params.bpe_model
    params.dtype = torch.float32
    params.use_autocast = False

    model = T.get_model(params)
    total = sum(p.numel() for p in model.parameters())

    print("=" * 100)
    print(f"vocab_size = {params.vocab_size}  (từ {src})   "
          f"use_transducer={params.use_transducer}  use_ctc={params.use_ctc}")
    print(f"TỔNG PARAM: {fmt(total)}")
    print("=" * 100)

    # ── Top-level: cái này là bảng để chọn TRAINABLE_MODULES ────────────────
    print("\n### Top-level module (đây là các prefix hợp lệ cho TRAINABLE_MODULES)\n")
    print(f"{'module':<24} {'param':>14} {'%':>7}")
    print("-" * 48)
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        print(f"{name:<24} {fmt(n):>14} {100 * n / total:>6.2f}%")

    # ── Cây module ──────────────────────────────────────────────────────────
    print(f"\n### Cây module (depth <= {params.depth}"
          + (f", bỏ module < {fmt(params.min_params)} param" if params.min_params else "")
          + ")\n")
    for name, mod in model.named_modules():
        if name == "":
            continue
        depth = name.count(".")
        if depth > params.depth:
            continue
        n = sum(p.numel() for p in mod.parameters())
        if n < params.min_params:
            continue
        leaf = not any(True for _ in mod.children())
        short = name.split(".")[-1]
        pad = "    " * depth
        tag = mod.__class__.__name__
        print(f"{pad}{short:<{max(1, 26 - len(pad))}} {tag:<26} "
              f"{fmt(n):>13}  {100 * n / total:>5.2f}%" + ("  ←leaf" if leaf else ""))

    if params.show_params:
        print("\n### Từng tensor param\n")
        for name, p in model.named_parameters():
            print(f"  {name:<62} {str(tuple(p.shape)):<20} {fmt(p.numel()):>13}")

    # ── Mô phỏng freeze: xem TRƯỚC một phạm vi train bao nhiêu param ─────────
    prefixes = []
    if params.trainable_modules.strip():
        prefixes = [m.strip() for m in params.trainable_modules.split(",") if m.strip()]
    elif params.freeze_encoder:
        prefixes = ["ctc_output", "joiner", "decoder",
                    "simple_am_proj", "simple_lm_proj", "attention_decoder"]
    print("\n" + "=" * 100)
    if prefixes:
        keep = [(n, p.numel()) for n, p in model.named_parameters()
                if any(n == x or n.startswith(x + ".") for x in prefixes)]
        tr = sum(n for _, n in keep)
        print(f"FREEZE mô phỏng — prefix {prefixes}")
        print(f"  trainable: {fmt(tr)} ({100 * tr / total:.3f}%)   "
              f"frozen: {fmt(total - tr)}")
        for n, c in keep:
            print(f"    {n:<60} {fmt(c):>13}")
        if not keep:
            print("  ⚠️  KHÔNG param nào khớp — prefix gõ sai. train.py sẽ train TOÀN BỘ "
                  "vì `if _trainable_prefixes:` là danh sách rỗng.")
    else:
        print(f"FREEZE mô phỏng — không có prefix ⇒ FULL FINETUNE {fmt(total)} (100%)")
    print("=" * 100)

    if params.ckpt:
        print(f"\n### Kiểm checkpoint {params.ckpt}")
        sd = torch.load(params.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  key trong checkpoint : {len(sd)}")
        print(f"  MISSING (model cần mà ckpt không có) : {len(missing)}")
        for k in missing[:10]:
            print(f"      {k}")
        print(f"  UNEXPECTED (ckpt có mà model không cần): {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"      {k}")
        if not missing and not unexpected:
            print("  ✅ khớp hoàn toàn")


if __name__ == "__main__":
    main()

# Chạy:
#   R=/home/anlnm/lmq/train_zipformer_5080
#   docker run --rm \
#     -v /mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/lang_bpe_500_gip:/app/icefall/egs/librispeech/ASR/zipformer/data/lang_bpe_500:ro \
#     -v "$R/scripts:/app/scripts:ro" -v "$R/word_mapping.json:/app/word_mapping.json:ro" \
#     -v "$R/callbacks:/ck:ro" \
#     --entrypoint bash zipformer-training:latest -c '
#   cd /app/icefall/egs/librispeech/ASR/zipformer && cp -f /app/scripts/train.py /app/scripts/model.py .
#   python3 /app/scripts/print_arch.py --use-transducer 1 --use-ctc 1 --depth 3 \
#     --trainable-modules decoder --ckpt /ck/epoch-35.pt'
