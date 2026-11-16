#!/bin/bash
# Zipformer ASR Training entrypoint
# Môi trường: icefall/egs/librispeech/ASR/zipformer

set -e

# Đánh dấu thời gian bắt đầu trọn bộ pipeline
PIPELINE_START_TIME=$(date +%s)

WORKDIR="/app/icefall/egs/librispeech/ASR/zipformer"
cd "$WORKDIR"

DATA_DIR="/data"              # mount từ host: $(pwd)/DATA
EXP_DIR="/app/callbacks"      # mount từ host: $(pwd)/callbacks
FBANK_DIR="data/fbank"        # mount từ host: $(pwd)/fbank_data
BPE_DIR="data/lang_bpe_500"   # mount từ host: $(pwd)/lang_bpe_500
BPE_MODEL="$BPE_DIR/bpe.model"
TOKENS_TXT="$BPE_DIR/tokens.txt"

# ── Load .env (nếu được mount vào container) ──────────────────────────────────
# run.sh mount .env qua --env-file, nên các biến đã có sẵn trong env.
# Đoạn này chỉ để fallback nếu chạy container thủ công.
if [ -f "/app/.env" ]; then
    set -a; source /app/.env; set +a
fi

# ── WandB ─────────────────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-zipformer-training}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python3 -c "import wandb; print(wandb.util.generate_id())")}"
    export WANDB_RESUME="allow"
fi

echo "=============================================="
echo "🤖 Zipformer Training Pipeline"
echo "=============================================="
echo "  Start time: $(date)"
echo "  WandB project: ${WANDB_PROJECT}"
echo "  WandB logging: $([ -n \"$WANDB_API_KEY\" ] && echo 'enabled' || echo 'disabled - no WANDB_API_KEY')"
echo "  WandB run name: ${WANDB_RUN_NAME}"
echo "=============================================="
echo ""

# ── 1. Chuẩn bị thư mục ──────────────────────────────────────────────────────
mkdir -p "$BPE_DIR" "$FBANK_DIR" "$EXP_DIR"

# Symlink exp → /app/callbacks (để train.py dùng --exp-dir exp)
if [ ! -L exp ] && [ -d exp ]; then
    rm -rf exp
fi
if [ ! -e exp ]; then
    ln -s "$EXP_DIR" exp
fi

# ── 2. Tải BPE model ─────────────────────────────────────────────────────────
if [ ! -f "$BPE_MODEL" ]; then
    echo "📥 Downloading BPE model..."
    python3 -c "
import requests
url = '${BPE_URL:-https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h/resolve/main/bpe.model}'
with open('$BPE_MODEL', 'wb') as f:
    f.write(requests.get(url).content)
print('✅ BPE model downloaded')
"
else
    echo "✅ BPE model already exists, skipping download"
fi

# ── 3. Lowercase BPE tokens và tạo tokens.txt ────────────────────────────────
# QUAN TRỌNG: KHÔNG ghi đè bpe.model — lowercase trong RAM, export thẳng ra tokens.txt
if [ ! -f "$TOKENS_TXT" ]; then
    echo "🔡 Lowercasing BPE tokens and generating tokens.txt..."
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 -c "
import re
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

MODEL_PATH = '$BPE_MODEL'

model = sp_pb2.ModelProto()
with open(MODEL_PATH, 'rb') as f:
    raw = f.read()
model.ParseFromString(raw)

seen = {}
changed = 0
duplicates = 0
foreign_count = 0
unused_counter = 0

ALLOWED_PATTERN = re.compile(r'^[a-záàãạảăắằẵặẳâấầẫậẩéèẽẹẻêếềễệểíìĩịỉóòõọỏôốồỗộổơớờỡợởúùũụủưứừữựửýỳỹỵỷđ0-9.,!?_\-\'\"\  ▁]+\$')

token_list = []
for idx, piece in enumerate(model.pieces):
    original = piece.piece
    if original.startswith('<') and original.endswith('>'):
        seen[original] = idx
        token_list.append(original)
        continue
    lowered = original.lower()

    if not ALLOWED_PATTERN.match(lowered):
        placeholder = f'<unused_{unused_counter}>'
        unused_counter += 1
        seen[placeholder] = idx
        token_list.append(placeholder)
        foreign_count += 1
        continue

    if lowered in seen:
        placeholder = f'<unused_{unused_counter}>'
        unused_counter += 1
        seen[placeholder] = idx
        token_list.append(placeholder)
        duplicates += 1
    else:
        if lowered != original:
            changed += 1
        seen[lowered] = idx
        token_list.append(lowered)

with open('$TOKENS_TXT', 'w', encoding='utf-8') as f:
    for i, tok in enumerate(token_list):
        f.write(f'{tok} {i}\n')

print(f'✅ Lowercased {changed} tokens')
print(f'✅ Replaced {foreign_count} foreign + {duplicates} duplicate tokens with <unused_N>')
print(f'✅ tokens.txt generated with {len(token_list)} tokens')

print('\n--- 10 token đầu tiên ---')
for t in token_list[:10]:
    print(t)
print('\n--- 10 token cuối cùng ---')
for t in token_list[-10:]:
    print(t)
print('--------------------------')
"
else
    echo "✅ tokens.txt already exists, skipping"
fi

# ── 4. Chuẩn bị dữ liệu (Fbank features) ────────────────────────────────────
# Khi SKIP_PREPROCESSING=true:
#   - Fbank manifests + feats đã được mount từ host vào $FBANK_DIR (/app/fbank_data)
#   - Bỏ qua toàn bộ bước extract features (tốn ~vài phút đến hàng giờ)
# Khi SKIP_PREPROCESSING=false (default):
#   - Đọc wav+txt từ /data, trích xuất Fbank 80-dim, lưu manifest jsonl.gz
MANIFEST_TRAIN="$FBANK_DIR/librispeech_cuts_train-clean-100.jsonl.gz"

if [ "${SKIP_PREPROCESSING:-false}" = "true" ]; then
    if [ -f "$MANIFEST_TRAIN" ]; then
        echo "⏭️  Skip preprocessing — dùng fbank manifests có sẵn tại: $FBANK_DIR"
        echo "   $(ls $FBANK_DIR/*.jsonl.gz 2>/dev/null | wc -l) manifest files found"
    else
        echo "❌ SKIP_PREPROCESSING=true nhưng không tìm thấy manifest tại: $FBANK_DIR"
        echo "   Hãy mount thư mục fbank đã chuẩn bị sẵn:"
        echo "   Chạy run.sh không có --skip-preprocess lần đầu để tạo data/"
        exit 1
    fi
else
    echo "⚙️  Preparing data from $DATA_DIR..."
    TEST_ARGS=""
    if [ "${TEST_MODE:-false}" = "true" ]; then
        TEST_ARGS="--test-mode --test-samples ${TEST_SAMPLES:-10}"
        echo "   ⚠️  TEST MODE: tối đa ${TEST_SAMPLES:-10} file/dataset"
    fi
    python3 /app/scripts/prepare_data.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$FBANK_DIR" \
        --num-workers "${NUM_WORKERS:-8}" \
        --num-gpus 1 \
        $TEST_ARGS
fi

# ── 5. Train ─────────────────────────────────────────────────────────────────
# Ưu tiên CUDA_VISIBLE_DEVICES để tránh torch detect sai số GPU
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
fi

# Icefall mặc định:
#   + Tự động save checkpoint vào cuối MỖI epoch
#   + Tự động chạy validation vào cuối MỖI epoch
#   + Chỉ cần thiết lập KEEP_LAST_K để xoay vòng dung lượng ổ cứng.

# ── Resume: theo đúng START_EPOCH trong .env (KHÔNG auto-detect) ──────────────
# START_EPOCH=N nghĩa là load exp/epoch-{N-1}.pt và train từ epoch N.
_RESUME_CKPT="$EXP_DIR/epoch-$(( ${START_EPOCH:-1} - 1 )).pt"
if [ "${START_EPOCH:-1}" -gt 1 ]; then
    if [ -f "$_RESUME_CKPT" ]; then
        echo "  🔄 Resume theo START_EPOCH=${START_EPOCH} → load $(basename "$_RESUME_CKPT")"
    else
        echo "  ❌ START_EPOCH=${START_EPOCH} nhưng không tìm thấy $_RESUME_CKPT"
        echo "     Các checkpoint đang có:"
        ls -1 "$EXP_DIR"/epoch-*.pt 2>/dev/null | sed 's/^/       /' || echo "       (không có epoch-*.pt nào)"
        exit 1
    fi
else
    echo "  🆕 START_EPOCH=1 → train từ đầu, không load checkpoint"
fi


echo ""
echo "=============================================="
echo "🚀 Zipformer Training Pipeline"
echo "=============================================="
echo "  GPUs (count):    $NUM_GPUS"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'(not set, dùng tất cả)'}"
python3 -c "
import torch
print(f'  torch.cuda.device_count() = {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  cuda:{i} → {torch.cuda.get_device_name(i)}')
"
echo "  Num epochs:      ${NUM_EPOCHS:-30}"
echo "  Start epoch:     ${START_EPOCH:-1}"
echo "  Max duration:    ${MAX_DURATION:-300}s (dynamic batching)"
echo "  Base LR:         ${BASE_LR:-0.045}"
echo "  Keep last K ckpt:${KEEP_LAST_K:-5}"
echo "  Exp dir:         $(readlink -f exp)"
echo "=============================================="
echo ""
# export PYTHONWARNINGS="ignore" cấm hiển thị các cảnh báo làm rác log
export PYTHONWARNINGS="ignore"

# Giảm memory fragmentation trong long-running process
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728
# Giúp debug CUDA error thay vì silent segfault (bỏ comment nếu cần diagnose)
# export CUDA_LAUNCH_BLOCKING=1

# ── WandB TensorBoard Sync ───────────────────────────────────────────────────
# ── Xử lý file train.py Custom ──────────────────────────────────────────────────
# Sao chép file train.py từ host (mounted ở /app/scripts) đè lên file train.py gốc của Icefall
if [ -f "/app/scripts/train.py" ]; then
    echo "  Lấy file train.py tùy chỉnh từ /app/scripts/train.py đè lên mặc định..."
    cp -f /app/scripts/train.py ./train.py
fi

if [ -f "/app/scripts/model.py" ]; then
    echo "  Lấy file model.py tùy chỉnh từ /app/scripts/model.py đè lên mặc định..."
    cp -f /app/scripts/model.py ./model.py
fi

# ── WandB Login ──────────────────────────────────────────────────────────────
if [ -n "$WANDB_API_KEY" ]; then
    echo "  WandB: Enabling TensorBoard auto-sync..."
    wandb login "$WANDB_API_KEY"
fi

python3 train.py \
    --world-size "$NUM_GPUS" \
    --num-epochs "${NUM_EPOCHS:-30}" \
    --start-epoch "${START_EPOCH:-1}" \
    --use-fp16 "${USE_FP16:-1}" \
    --qat "${QAT:-False}" \
    --qat-calib-epochs "${QAT_CALIB_EPOCHS:-1}" \
    --qat-bits "${QAT_BITS:-8}" \
    --exp-dir exp \
    --bpe-model "$BPE_MODEL" \
    --num-workers "${NUM_WORKERS:-2}" \
    --max-duration "${MAX_DURATION:-300}" \
    --base-lr "${BASE_LR:-0.045}" \
    --lr-batches "${LR_BATCHES:-7500}" \
    --lr-epochs "${LR_EPOCHS:-3.5}" \
    --freeze-encoder "${FREEZE_ENCODER:-False}" \
    --trainable-modules "${TRAINABLE_MODULES:-}" \
    --emb-term-mask "${EMB_TERM_MASK:-}" \
    --on-the-fly-feats 0 \
    --full-libri 0 \
    --keep-last-k "${KEEP_LAST_K:-5}" \
    --save-every-n "${SAVE_EVERY_N:-9999999}" \
    --print-diagnostics False \
    --enable-musan False \
    --use-transducer "${USE_TRANSDUCER:-True}" \
    --use-ctc "${USE_CTC:-True}"

echo ""
echo "=============================================="

# Tính toán tổng thời gian Pipeline
PIPELINE_END_TIME=$(date +%s)
TOTAL_TIME=$((PIPELINE_END_TIME - PIPELINE_START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "✅ Training complete!"
echo "   Checkpoints: $(pwd)/exp  (→ host: callbacks/)"
echo "   End time: $(date)"
echo "   Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=============================================="
