#!/bin/bash
# Chuỗi QAT trên host (GPU dùng chung với người khác): chờ GPU rảnh (≤78GB đang dùng)
# → chạy QAT-int4 → xong chờ GPU rảnh nữa → chạy QAT-int8 (bản train lại).
# Log: /tmp/qat_chain.log (watcher), /tmp/qat4_train.log + /tmp/qat8_train.log (training).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

gpu_free_enough() {
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -le 78000 ]
}

echo "$(date -u '+%F %T') watcher chain QAT4→QAT8 khởi động"
until gpu_free_enough; do sleep 300; done
echo "$(date -u '+%F %T') GPU rảnh — chạy QAT-int4"
scripts/launch_qat4_host.sh > /tmp/qat4_train.log 2>&1
echo "$(date -u '+%F %T') QAT-int4 exit=$? — chờ GPU cho QAT-int8"
sleep 120
until gpu_free_enough; do sleep 300; done
echo "$(date -u '+%F %T') GPU rảnh — chạy QAT-int8 (train lại)"
scripts/launch_qat8_host.sh > /tmp/qat8_train.log 2>&1
echo "$(date -u '+%F %T') QAT-int8 exit=$? — chuỗi QAT kết thúc"
