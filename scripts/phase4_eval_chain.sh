#!/bin/bash
# Chuỗi Phase 3–4 hàng 2+3: đợi export xong → eval ONNX fp32 + int8 PTQ bằng sherpa-onnx
# (runtime prod), 4 mode decode, ghi JSONL. WER/recall tính sau bằng scripts/metrics.py.
set -u
cd /home/ubuntu/workspace/train-stt
source /home/ubuntu/workspace/.venv_tools/bin/activate

OUT=onnx_official_avg96100
CKPT=epoch-100-avg-1

# 1) đợi container export thoát
until [ ! "$(docker inspect -f '{{.State.Running}}' icefall_export 2>/dev/null)" = "true" ]; do
  sleep 60
done
EXIT=$(docker inspect -f '{{.State.ExitCode}}' icefall_export)
echo "[$(date -u +%H:%M)] export exit=$EXIT"
if [ "$EXIT" != "0" ]; then
  docker logs icefall_export 2>&1 | tail -15
  exit 1
fi

# 2) bundle cần tokens.txt + bpe.vocab cho hotwords
cp -f lang_bpe_scratch/tokens.txt lang_bpe_scratch/bpe.vocab "$OUT/"
ls -la "$OUT/" | awk '{print $5, $9}'

# 3) eval fp32 + int8 PTQ-dynamic
for SUFFIX in "$CKPT" "$CKPT.int8"; do
  echo "[$(date -u +%H:%M)] eval suffix=$SUFFIX bắt đầu"
  python3 scripts/eval_onnx.py \
    --model-dir "$OUT" \
    --suffix "$SUFFIX" \
    --manifest fbank_scratch/librispeech_cuts_dev-clean.jsonl.gz \
    --audio-root /home/ubuntu/workspace/train-stt/raw_dataset \
    --hotwords config/hotwords.txt \
    --threads 8 \
    --out "eval_onnx_${SUFFIX}.jsonl" 2>&1 | tail -20
  echo "[$(date -u +%H:%M)] eval suffix=$SUFFIX xong → eval_onnx_${SUFFIX}.jsonl"
done
echo "=== CHUỖI EVAL HÀNG 2+3 XONG $(date -u +%H:%M) ==="
