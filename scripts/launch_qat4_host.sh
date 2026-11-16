#!/bin/bash
# Chạy QAT-int4 TRỰC TIẾP trên host qua .venv_train (không docker — image bị prune,
# đĩa không đủ build lại). Dịch từ entrypoint.sh + .env_qat4, giữ nguyên mọi flag.
set -euo pipefail
cd /home/ubuntu/workspace/train-stt/icefall_host/egs/librispeech/ASR/zipformer
source /home/ubuntu/workspace/train-stt/.venv_train/bin/activate
export PYTHONPATH=/home/ubuntu/workspace/train-stt/icefall_host
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=134217728

# env từ .env_qat4 (train.py đọc trực tiếp)
export LOG_INTERVAL=200 EVAL_FIRST=true TRAIN_EVAL_CUTS=1000
export EVAL_MIN_DURATION=1.0 WER_DECODE=both BAND_CLAMP_BIN=0

mkdir -p log
LOGFILE="log/log-train-$(date -u +%Y-%m-%d-%H-%M-%-%S)"
exec python3 train.py \
    --world-size 1 \
    --num-epochs 6 \
    --start-epoch 2 \
    --use-fp16 0 \
    --qat True \
    --qat-calib-epochs 2 \
    --qat-bits 4 \
    --exp-dir exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --num-workers 2 \
    --max-duration 450 \
    --base-lr 0.001 \
    --lr-batches 300000 \
    --lr-epochs 100 \
    --freeze-encoder False \
    --trainable-modules "" \
    --emb-term-mask "" \
    --on-the-fly-feats 0 \
    --full-libri 0 \
    --keep-last-k 5 \
    --save-every-n 9999999 \
    --print-diagnostics False \
    --enable-musan False \
    --use-transducer True \
    --use-ctc True
