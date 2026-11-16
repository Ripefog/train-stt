#!/bin/bash
# Export Zipformer → ONNX, rồi inference 3 mẫu đầu tập valid.
#
# Usage:
#   ./export.sh                         # dùng epoch-6.pt, fp32
#   ./export.sh --int8                  # thêm inference bằng int8 model
#   ./export.sh --checkpoint other.pt   # dùng checkpoint khác
#
# Tất cả arg sau "--" được truyền vào infer_onnx.py (vd: --n-samples 5)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="$SCRIPT_DIR/onnx"

# ── Tách arg cho export vs infer ──────────────────────────────────────────────
EXPORT_ARGS=()
INFER_ARGS=()
PASS_TO_INFER=false

for arg in "$@"; do
    case "$arg" in
        --int8)
            EXPORT_ARGS+=(--no-int8)   # bỏ qua: int8 export luôn chạy mặc định
            # (int8 export chạy mặc định; --int8 chỉ ảnh hưởng infer)
            INFER_ARGS+=(--int8)
            PASS_TO_INFER=false
            ;;
        --checkpoint)
            PASS_TO_INFER=false
            EXPORT_ARGS+=("$arg")
            ;;
        --)
            PASS_TO_INFER=true
            ;;
        *)
            if $PASS_TO_INFER; then
                INFER_ARGS+=("$arg")
            else
                EXPORT_ARGS+=("$arg")
            fi
            ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Zipformer ONNX Export + Inference                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Export ────────────────────────────────────────────────────────────
echo "▶ [1/2] Exporting ONNX models → $ONNX_DIR"
echo ""
python3 "$SCRIPT_DIR/scripts/export_onnx.py" \
    --outdir "$ONNX_DIR" \
    "${EXPORT_ARGS[@]}"

echo ""
echo "✓ Export hoàn thành. Files:"
ls -lh "$ONNX_DIR"/*.onnx 2>/dev/null || echo "  (không tìm thấy .onnx files)"

# ── Step 2: Inference ─────────────────────────────────────────────────────────
echo ""
echo "▶ [2/2] Inference trên 3 mẫu đầu tập valid..."
echo ""
python3 "$SCRIPT_DIR/scripts/infer_onnx.py" \
    --onnx-dir "$ONNX_DIR" \
    "${INFER_ARGS[@]}"

echo ""
echo "✓ Xong!"
