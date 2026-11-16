#!/bin/bash
# Zipformer ASR Training — chạy Docker container
# Tất cả config nằm trong file .env
# Usage: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load .env ─────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo "❌ Không tìm thấy file .env"
    echo "   Chạy: cp .env.example .env rồi chỉnh lại"
    exit 1
fi
set -a; source .env; set +a

# # ── Kiểm tra DATA/ ────────────────────────────────────────────────────────────
# if [ ! -d "DATA" ]; then
#     echo "❌ Không tìm thấy thư mục DATA/"
#     exit 1
# fi
# WAV_COUNT=$(find DATA -name "*.wav" 2>/dev/null | wc -l)
# echo "📂 DATA/: $WAV_COUNT file .wav"

# ── Build image ───────────────────────────────────────────────────────────────
# if [ "${BUILD:-false}" = "true" ] || [ -z "$(docker images -q "${IMAGE_NAME:-zipformer-training:latest}" 2>/dev/null)" ]; then
#     echo "🔨 Building Docker image: ${IMAGE_NAME:-zipformer-training:latest} ..."
#     docker build -t "${IMAGE_NAME:-zipformer-training:latest}" .
# fi
_image="${IMAGE_NAME:-zipformer-training:latest}"
if [ "${BUILD:-false}" = "true" ] || [ -z "$(docker images -q "$_image" 2>/dev/null)" ]; then
    echo "🔨 Building Docker image: $_image ..."
    docker build -t "$_image" .
fi

# ── Tạo thư mục output ────────────────────────────────────────────────────────
# mkdir -p callbacks fbank_data lang_bpe_500
mkdir -p callbacks

# ── Screen session ────────────────────────────────────────────────────────────
if [ "${USE_SCREEN:-false}" = "true" ] && [ -z "$STY" ]; then
    SESSION_NAME="zipformer_train_$(date +%Y%m%d_%H%M%S)"
    echo "🖥️  Khởi động screen session: ${SESSION_NAME}"
    echo "  Theo dõi: screen -r ${SESSION_NAME}"
    echo "  Tách ra:  Ctrl+A rồi D"
    exec screen -S "${SESSION_NAME}" bash -c "cd '$SCRIPT_DIR' && USE_SCREEN=false ./run.sh; exec bash"
fi

# ── Sinh run name (dùng chung cho container và WandB) ─────────────────────────
RUN_NAME="zipformer_train_$(date +%Y%m%d_%H%M%S)"

# ── In config ─────────────────────────────────────────────────────────────────
echo ""
echo "🚀 Khởi động container Zipformer..."
echo "   Image:            ${IMAGE_NAME:-zipformer-training:latest}"
echo "   DATA:             /mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/formated_dataset → /data (ro)"
echo "   Callbacks:        $(pwd)/callbacks → /app/callbacks"
echo "   Fbank data:       /mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/fbank_data → data/fbank"
echo "   BPE model:        /mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/lang_bpe_500_gip → data/lang_bpe_500"
echo "   Detach:           ${DETACH:-true}"
echo "   Skip preprocess:  ${SKIP_PREPROCESSING:-false}"
echo "   Num epochs:       ${NUM_EPOCHS:-30}  (start: ${START_EPOCH:-1})"
echo "   Base LR:          ${BASE_LR:-0.045}  Batch size: ${MAX_DURATION:-300}s"
echo "   Keep last K:      ${KEEP_LAST_K:-5}"
echo "   WandB project:    ${WANDB_PROJECT:-}  run: ${RUN_NAME}"
echo ""

# ── Docker run ────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --gpus all
    --shm-size=32g
    -v "/mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/formated_dataset:/data:ro"
    -v "$(pwd)/callbacks:/app/callbacks"
    -v "/mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/fbank_data:/app/icefall/egs/librispeech/ASR/zipformer/data/fbank"
    -v "/mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/lang_bpe_500_gip:/app/icefall/egs/librispeech/ASR/zipformer/data/lang_bpe_500"
    -v "$(pwd)/scripts:/app/scripts:ro"
    -v "$(pwd)/config/word_mapping.json:/app/word_mapping.json:ro"
    --env-file "$(pwd)/.env"
    -e WANDB_RUN_NAME="${RUN_NAME}"
)

if [ "${DETACH:-true}" = "true" ]; then
    docker run -d \
        --name "${RUN_NAME}" \
        --restart unless-stopped \
        "${COMMON_ARGS[@]}" \
        "${IMAGE_NAME:-zipformer-training:latest}"

    echo "✅ Container: ${RUN_NAME}"
    echo "   Xem log:  docker logs -f ${RUN_NAME}"
else
    docker run --rm -it \
        "${COMMON_ARGS[@]}" \
        "${IMAGE_NAME:-zipformer-training:latest}"
fi