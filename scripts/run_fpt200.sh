#!/bin/bash
# Finetune CHỈ trên 200h FPT — wrapper mỏng của run_new.sh với ENV_FILE=.env_fpt200.
#
# Vì sao là wrapper chứ không phải script riêng: run_new.sh đã có 5 preflight quan
# trọng (trùng path với container đang chạy, FBANK_DIR không được đè fbank chính,
# CALLBACKS_DIR không được đè ./callbacks, checkpoint để resume, RAM/GPU) và LUÔN
# dùng `-d --name` không `--rm` để log còn lại sau khi container thoát. Copy 250 dòng
# đó ra file thứ hai là chắc chắn lệch nhau về sau.
#
# Toàn bộ khác biệt nằm trong .env_fpt200: DATA_DIR / FBANK_DIR / CALLBACKS_DIR riêng.
# DATA_DIR override được nhờ run_new.sh:32 dùng "${DATA_DIR:-...}".
#
# Usage: ./run_fpt200.sh [-y|--yes] [-n|--dry-run]

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FPT200=".env_fpt200"
[ -f "$ENV_FPT200" ] || { echo "❌ Không tìm thấy $ENV_FPT200" >&2; exit 1; }

echo "══════════════════════════════════════════════════════════════════════════"
echo " Finetune 200h FPT  —  env: $ENV_FPT200"
echo "══════════════════════════════════════════════════════════════════════════"

# Hai cảnh báo của run_new.sh được hiệu chỉnh cho corpus 2.340h; ở 200h chúng sai
# một bậc độ lớn. Nói trước để không ai bỏ dở vì tưởng phải chờ 7 tiếng.
if ! grep -qE '^SKIP_PREPROCESSING=true' "$ENV_FPT200"; then
    echo "ℹ️  Sắp EXTRACT fbank mới. run_new.sh sẽ cảnh báo \"cần ~77 GB / 7 tiếng\" —"
    echo "    con số đó cho 2.340h. Thực tế ở 200h: ~7 GB, ~1,2h (8,1% của corpus)."
    echo "    Extract xong thì đặt SKIP_PREPROCESSING=true để lần quét lr sau không làm lại."
fi

# Bẫy đã ghi trong CLAUDE.md: START_EPOCH=N ghi ra epoch-N.pt và ĐÈ file cùng tên.
# Quét lr mà quên đổi CALLBACKS_DIR là mất kết quả lr trước mà không có cảnh báo nào.
_cb="$(grep -E '^CALLBACKS_DIR=' "$ENV_FPT200" | tail -1 | cut -d= -f2-)"
_lr="$(grep -E '^BASE_LR=' "$ENV_FPT200" | tail -1 | cut -d= -f2-)"
echo "ℹ️  BASE_LR=$_lr  →  $_cb"
if [ -n "$_cb" ] && compgen -G "$_cb/epoch-3[6-9].pt" >/dev/null 2>&1; then
    echo "⚠️  $_cb ĐÃ CÓ epoch-36..39.pt của một lần chạy trước."
    echo "⚠️     START_EPOCH sẽ GHI ĐÈ chúng. Đang quét lr thì đổi CALLBACKS_DIR trước."
fi

# Nhắc mốc đo, vì đây là khung đo MỚI (word_mapping đổi ép pê tê → ép pi ti):
echo "ℹ️  Khung đo MỚI: 37/549 cut val đổi reference. KHÔNG so với số nào trong"
echo "    CLAUDE.md — dùng dòng EVAL_FIRST của chính run này làm mốc."
echo "    Chốt kết luận bằng: scripts/eval_beam.py --beams 16  (KHÔNG chốt bằng greedy)."
echo

exec env ENV_FILE="$ENV_FPT200" ./run_new.sh "$@"
