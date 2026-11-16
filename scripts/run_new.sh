#!/bin/bash
# Zipformer ASR — chạy run THỨ HAI song song, CÁCH LY hoàn toàn với run đang chạy.
# Config đọc từ .env_new (không phải .env).
#
# Khác run.sh ở 4 điểm cốt lõi:
#   1. fbank dir + callbacks dir lấy từ .env_new → không đè dữ liệu của run đang chạy
#   2. PREFLIGHT: từ chối chạy nếu path va vào bất kỳ container nào đang chạy
#   3. LUÔN `-d --name`, không bao giờ `--rm` (run.sh với DETACH=false dùng --rm và
#      xoá sạch docker logs + exit code khi container thoát — đã mất bằng chứng crash 08-05)
#   4. `--restart no`: container crash thì DỪNG, không tự chạy lại. Với
#      SKIP_PREPROCESSING=false, auto-restart nghĩa là extract lại từ đầu và đè fbank.
#
# Usage: ./run_new.sh [-y]     (-y = bỏ qua xác nhận)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="${ENV_FILE:-.env_new}"     # override được để test: ENV_FILE=x ./run_new.sh -n
ASSUME_YES=false
DRY_RUN=false
for _arg in "$@"; do
    case "$_arg" in
        -y|--yes)     ASSUME_YES=true ;;
        -n|--dry-run) DRY_RUN=true ;;
        *) echo "Usage: $0 [-y|--yes] [-n|--dry-run]" >&2; exit 2 ;;
    esac
done

# DATA_DIR override được từ env file (run_fpt200.sh dùng để trỏ vào data chỉ có FPT).
# Mặc định giữ nguyên corpus 2.340h nên run_new.sh không đổi hành vi.
DATA_DIR="${DATA_DIR:-/mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/formated_dataset}"
# BPE_DIR override được từ env file (run from-scratch dùng lang_bpe_scratch tự train).
BPE_DIR="${BPE_DIR:-/mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/lang_bpe_500_gip}"
MAIN_FBANK="/mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/fbank_data"

die() { echo "❌ $*" >&2; exit 1; }
warn() { echo "⚠️  $*"; }

# ── Load .env_new ─────────────────────────────────────────────────────────────
[ -f "$ENV_FILE" ] || die "Không tìm thấy $ENV_FILE (xem .env_new.example hoặc copy từ .env rồi chỉnh)"
set -a; source "$ENV_FILE"; set +a

: "${FBANK_DIR:?$ENV_FILE phải khai báo FBANK_DIR (thư mục fbank RIÊNG cho run này)}"
: "${CALLBACKS_DIR:?$ENV_FILE phải khai báo CALLBACKS_DIR (thư mục checkpoint RIÊNG)}"
RUN_TAG="${RUN_TAG:-zipformer_new}"
SHM_SIZE="${SHM_SIZE:-8g}"

# realpath -m: chuẩn hoá được cả path chưa tồn tại, nên KHÔNG cần mkdir trước.
# mkdir để sau toàn bộ preflight — path gõ sai không được tạo thư mục rác, nhất là
# khi nó rơi vào bên trong data đang được run chính dùng.
FBANK_DIR="$(realpath -m "$FBANK_DIR")"
CALLBACKS_DIR="$(realpath -m "$CALLBACKS_DIR")"

# ── PREFLIGHT 1: không được trùng path của run chính ──────────────────────────
# SKIP_PREPROCESSING=true thì fbank CHỈ ĐƯỢC ĐỌC, nên dùng chung với run đang chạy là
# an toàn — mount :ro để kernel bảo đảm chứ không dựa vào thiện chí. Nhờ vậy probe
# BAND_CLAMP chạy được ngay trên fbank full-band hiện có, không phải chờ 7h extract.
# Ngược lại SKIP_PREPROCESSING=false thì prepare_data GHI vào đó → tuyệt đối không chung.
if [ "${SKIP_PREPROCESSING:-false}" = "true" ]; then
    FBANK_SHARED=true; FBANK_MOUNT_OPT=":ro"
else
    FBANK_SHARED=false; FBANK_MOUNT_OPT=""
    [ "$FBANK_DIR" = "$(realpath "$MAIN_FBANK")" ] && \
        die "SKIP_PREPROCESSING=false + FBANK_DIR = fbank_data của run chính → sẽ GHI ĐÈ feature đang được đọc."
fi
[ "$CALLBACKS_DIR" = "$(realpath ./callbacks)" ] && \
    die "CALLBACKS_DIR trỏ đúng vào ./callbacks của run chính — sẽ ĐÈ epoch-*.pt đang được ghi."

# ── PREFLIGHT 2: không được va vào mount của BẤT KỲ container đang chạy ───────
# Bắt cả trường hợp cha/con, không chỉ trùng khít: mount fbank_data rồi ta chọn
# fbank_data/sub thì vẫn là cùng dữ liệu.
_overlaps() {                      # $1, $2 = 2 path tuyệt đối
    [ "$1" = "$2" ] && return 0
    case "$1/" in "$2"/*) return 0 ;; esac
    case "$2/" in "$1"/*) return 0 ;; esac
    return 1
}

echo "🔎 Preflight: quét mount của các container đang chạy..."
_clash=""
for _cid in $(docker ps -q); do
    _cname="$(docker inspect "$_cid" --format '{{.Name}}' 2>/dev/null | sed 's|^/||')"
    while read -r _src; do
        [ -z "$_src" ] && continue
        [ -e "$_src" ] || continue
        _src="$(realpath "$_src" 2>/dev/null || echo "$_src")"
        # fbank chỉ kiểm khi nó sẽ bị GHI; đọc chung (:ro) thì không phải xung đột
        _check_list="$CALLBACKS_DIR"
        [ "$FBANK_SHARED" = "true" ] || _check_list="$FBANK_DIR $CALLBACKS_DIR"
        for _mine in $_check_list; do
            if _overlaps "$_mine" "$_src"; then
                _clash="${_clash}   ${_cname}: ${_src}  ⟷  ${_mine}"$'\n'
            fi
        done
    done < <(docker inspect "$_cid" --format '{{range .Mounts}}{{println .Source}}{{end}}' 2>/dev/null)
done
if [ -n "$_clash" ]; then
    echo "$_clash"
    die "Path trùng với container đang chạy — DỪNG để không phá run đó."
fi
echo "   ✅ không trùng mount nào"

# Preflight path đã xong → giờ mới được tạo thư mục
mkdir -p "$FBANK_DIR" "$CALLBACKS_DIR"

# ── PREFLIGHT 3: checkpoint để resume ────────────────────────────────────────
_start="${START_EPOCH:-1}"
if [ "$_start" -gt 1 ]; then
    _ckpt="$CALLBACKS_DIR/epoch-$(( _start - 1 )).pt"
    if [ ! -f "$_ckpt" ]; then
        echo "❌ START_EPOCH=$_start cần $(basename "$_ckpt") trong $CALLBACKS_DIR nhưng không có."
        echo "   entrypoint.sh:198 sẽ fail. Copy checkpoint gốc vào rồi chạy lại:"
        echo "     cp ./callbacks/epoch-$(( _start - 1 )).pt $CALLBACKS_DIR/"
        exit 1
    fi
fi

# ── PREFLIGHT 4: manifest vs SKIP_PREPROCESSING ──────────────────────────────
_manifest="$FBANK_DIR/librispeech_cuts_train-clean-100.jsonl.gz"
if [ "${SKIP_PREPROCESSING:-false}" = "true" ]; then
    [ -f "$_manifest" ] || die "SKIP_PREPROCESSING=true nhưng $FBANK_DIR chưa có manifest. Đặt false để extract, hoặc trỏ FBANK_DIR sang thư mục đã có."
    echo "   ✅ dùng lại manifest có sẵn ($(du -sh "$FBANK_DIR" | cut -f1))"
else
    if [ -f "$_manifest" ]; then
        warn "SKIP_PREPROCESSING=false và $FBANK_DIR ĐÃ CÓ manifest — sẽ bị GHI ĐÈ."
    fi
    _free_gb=$(df -BG --output=avail "$FBANK_DIR" | tail -1 | tr -dc '0-9')
    echo "   ℹ️  extract mới: cần ~77 GB, ổ còn ${_free_gb} GB"
    [ "$_free_gb" -lt 90 ] && warn "chỗ trống mỏng (<90 GB)"
    warn "extract đọc lại TOÀN BỘ wav từ ổ cơ. Ổ đang bão hoà vì run chính (disk-bound) → cả hai sẽ chậm đi rõ rệt."
    # Chặn cái bẫy tốn 7 tiếng: extract lại mà prepare_data.py chưa có thay đổi nào
    # thì fbank mới GIỐNG HỆT fbank cũ, chỉ khác đường dẫn.
    if ! grep -q 'BAND_LIMIT' scripts/prepare_data.py 2>/dev/null; then
        warn "scripts/prepare_data.py KHÔNG có xử lý BAND_LIMIT nào → fbank mới sẽ giống hệt"
        warn "   $MAIN_FBANK, chỉ khác đường dẫn. 7 tiếng extract không đổi lấy gì cả."
        warn "   Muốn fbank khác thì phải patch prepare_data.py trước."
    fi
fi

# ── PREFLIGHT 5: RAM ─────────────────────────────────────────────────────────
# Chạy CHO MỌI trường hợp, không chỉ khi extract. Ngày 2026-08-06 chính chỗ này thiếu:
# check RAM từng nằm trong nhánh SKIP_PREPROCESSING=false nên đường clamp không bao giờ
# đi qua, và run chính bị OOM killer giết sau 10 tiếng train (docker event `oom` rồi
# `die exitCode=137`, lúc 19:17). Đo được: MỘT process training ≈ 19 GB RSS. Host 62 GB
# đã cõng nghechuanai/smart-agent/kube nên không đủ cho 2 run.
# shm-size là hạn mức chứ không phải cấp trước, NHƯNG trang tmpfs đã dùng thì tính vào
# RAM và không bị đẩy ra swap — nên shm to vẫn góp phần đẩy tới OOM.
_TRAIN_RSS_GB=19
_ram_gb=$(free -g | awk '/^Mem:/{print $7}')
_swap_used=$(free -g | awk '/^Swap:/{print $3}')
_n_train=$(docker ps --format '{{.Names}}' | grep -cE 'zipformer|train' || true)
echo "   ℹ️  RAM khả dụng ${_ram_gb} GB, swap đã dùng ${_swap_used} GB, "\
"${_n_train} container training đang chạy"
if [ "$_n_train" -gt 0 ]; then
    warn "ĐÃ CÓ ${_n_train} run training. Mỗi run ≈ ${_TRAIN_RSS_GB} GB RSS —"
    warn "   chạy song song trên host này đã gây OOM một lần (2026-08-06, mất 6h compute)."
    warn "   Và vì ổ cơ đã bão hoà, song song KHÔNG tăng tổng thông lượng: 2 run × ~11%"
    warn "   GPU util = đúng bằng 1 run × 23%. Chỉ được thêm thí nghiệm, không thêm tốc độ."
fi
if [ "$_ram_gb" -lt "$_TRAIN_RSS_GB" ] && [ "${FORCE_LOW_RAM:-false}" != "true" ]; then
    die "RAM khả dụng ${_ram_gb} GB < ${_TRAIN_RSS_GB} GB mà một run training cần → gần
    chắc chắn OOM, và kernel sẽ giết process TO NHẤT (có thể là run đang chạy dở, không
    phải run mới này). Dừng bớt việc rồi chạy lại, hoặc FORCE_LOW_RAM=true nếu bạn chắc."
fi

# ── PREFLIGHT 6: GPU ─────────────────────────────────────────────────────────
_mygpu="${CUDA_VISIBLE_DEVICES:-}"
for _cid in $(docker ps -q); do
    _theirs="$(docker inspect "$_cid" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
               | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
    if [ -n "$_mygpu" ] && [ "$_theirs" = "$_mygpu" ]; then
        warn "container $(docker inspect "$_cid" --format '{{.Name}}' | sed 's|^/||') cũng dùng GPU $_mygpu → tranh VRAM, dễ CUDA OOM. Cân nhắc GPU khác hoặc giảm MAX_DURATION."
    fi
done
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/   GPU /'

# ── Build image (mặc định KHÔNG build) ───────────────────────────────────────
_image="${IMAGE_NAME:-zipformer-training:latest}"
if [ "${BUILD:-false}" = "true" ]; then
    echo "🔨 Build $_image ..."
    docker build -t "$_image" .
elif [ -z "$(docker images -q "$_image" 2>/dev/null)" ]; then
    die "Không có image $_image. Đặt BUILD=true trong $ENV_FILE."
fi

RUN_NAME="${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"

# Phạm vi trainable THẬT — phải khớp train.py:1710-1718, nếu không dòng in ra sẽ nói dối.
# Bẫy đã gặp 2026-08-11: in "${TRAINABLE_MODULES:-<toàn bộ>}" thì TRAINABLE_MODULES rỗng
# + FREEZE_ENCODER=true vẫn hiện "<toàn bộ>" trong khi thực tế chỉ 8,1% param được train.
if [ -n "${TRAINABLE_MODULES:-}" ]; then
    _scope="CHỈ ${TRAINABLE_MODULES}   (TRAINABLE_MODULES thắng FREEZE_ENCODER)"
elif [ "${FREEZE_ENCODER:-false}" = "true" ]; then
    _scope="ĐÓNG BĂNG ENCODER → chỉ ctc_output,joiner,decoder,simple_am_proj,simple_lm_proj ≈ 5.657.408 param (8,1%)"
else
    _scope="TOÀN BỘ 69.651.511 param (full finetune)"
fi

cat <<EOF

🚀 Run MỚI (cách ly) — $RUN_NAME
   env file:         $ENV_FILE
   image:            $_image
   DATA:             $DATA_DIR → /data (ro)
   Fbank:            $FBANK_DIR${FBANK_MOUNT_OPT:+   (DÙNG CHUNG, mount read-only)}
   Callbacks(RIÊNG): $CALLBACKS_DIR
   BPE:              $BPE_DIR (ro, dùng chung — phải khớp vocab của checkpoint)
   GPU:              ${_mygpu:-all}     shm: $SHM_SIZE
   Skip preprocess:  ${SKIP_PREPROCESSING:-false}
   Epoch:            ${START_EPOCH:-1} → ${NUM_EPOCHS:-30}
   BASE_LR:          ${BASE_LR:-?}   MAX_DURATION: ${MAX_DURATION:-?}s
   Trainable:        ${_scope}
   Cắt băng:         BAND_LIMIT_SR=${BAND_LIMIT_SR:-0}$([ "${BAND_LIMIT_SR:-0}" != "0" ] && echo " (low-pass $(( ${BAND_LIMIT_SR:-0} / 2 )) Hz, waveform)")   BAND_CLAMP_BIN=${BAND_CLAMP_BIN:-0}$([ "${BAND_CLAMP_BIN:-0}" != "0" ] && echo " (clamp bin ${BAND_CLAMP_BIN}-79, feature, train-only)")
   WandB:            ${WANDB_PROJECT:-} / $RUN_NAME

   KHÔNG đụng tới: ./callbacks, $MAIN_FBANK
EOF

if [ "$DRY_RUN" = "true" ]; then
    echo "🧪 --dry-run: preflight ĐẠT, dừng trước docker run. Không container nào được tạo."
    exit 0
fi

if [ "$ASSUME_YES" != "true" ]; then
    read -r -p "Chạy? [yes/N] " _ans
    # Cắt khoảng trắng + hạ chữ, và nhận cả 'y': gõ đúng "yes" mà vẫn bị huỷ đã xảy ra
    # một lần (2026-08-11) — nghi input bị BuildKit ăn khi BUILD=true vẽ progress trên
    # TTY, hoặc thừa ký tự. Dùng -y để bỏ hẳn bước này.
    _ans="$(printf '%s' "$_ans" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    case "$_ans" in
        y|yes) ;;
        *) die "Đã huỷ. (đọc được: '$_ans')" ;;
    esac
fi

docker run -d \
    --name "$RUN_NAME" \
    --restart no \
    --gpus all \
    --shm-size="$SHM_SIZE" \
    -v "$DATA_DIR:/data:ro" \
    -v "$CALLBACKS_DIR:/app/callbacks" \
    -v "$FBANK_DIR:/app/icefall/egs/librispeech/ASR/zipformer/data/fbank$FBANK_MOUNT_OPT" \
    -v "$BPE_DIR:/app/icefall/egs/librispeech/ASR/zipformer/data/lang_bpe_500" \
    -v "$SCRIPT_DIR/scripts:/app/scripts:ro" \
    -v "$SCRIPT_DIR/config/word_mapping.json:/app/word_mapping.json:ro" \
    --env-file "$SCRIPT_DIR/$ENV_FILE" \
    -e WANDB_RUN_NAME="$RUN_NAME" \
    "$_image"

cat <<EOF

✅ Container: $RUN_NAME   (--restart no, KHÔNG --rm → log còn sau khi thoát)
   log docker:  docker logs -f $RUN_NAME
   log icefall: tail -f $CALLBACKS_DIR/log/log-train-* # giờ UTC, host +07
   dừng:        docker stop $RUN_NAME && docker rm $RUN_NAME

⚠️  ĐỪNG sửa scripts/prepare_data.py khi container này đang extract: Pool dùng spawn +
    maxtasksperchild=1 nên worker mới import lại file từ đĩa và chết. (scripts/train.py
    thì an toàn — entrypoint.sh:242 đã cp ra bản riêng lúc khởi động.)
EOF
