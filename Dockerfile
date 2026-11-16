# Zipformer ASR Training Dockerfile
# Compatible with CUDA 12.8, NVIDIA Driver 575+ (PyTorch 2.10.0+cu128)
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    git \
    wget \
    curl \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# PyTorch with CUDA 12.4 support (tránh dùng cu126 vì k2 chưa có bản build sẵn cho cu126)
RUN pip3 install --no-cache-dir torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# k2: build from source (no pre-built wheel for torch==2.10.0+cu128)
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN git clone --depth 1 https://github.com/k2-fsa/k2.git /tmp/k2 && \
    cd /tmp/k2 && \
    K2_CMAKE_ARGS="-DK2_WITH_CUDA=ON" \
    pip3 install --no-cache-dir . && \
    rm -rf /tmp/k2

# Lhotse + icefall requirements
RUN pip3 install --no-cache-dir \
    lhotse \
    lilcom \
    sentencepiece \
    kaldifst \
    kaldilm \
    kaldialign \
    num2words \
    kaldi-decoder \
    onnx \
    onnxruntime \
    onnxoptimizer \
    onnxsim \
    onnxconverter_common \
    black==22.3.0 \
    isort==5.10.1 \
    typeguard \
    dill \
    tensorboard \
    librosa \
    soundfile \
    "numpy<2" \
    scipy \
    tqdm \
    requests \
    datasets \
    huggingface_hub \
    protobuf \
    wandb \
    h5py \
    jiwer

# Clone icefall
RUN git clone https://github.com/k2-fsa/icefall /app/icefall

# Install icefall requirements (bỏ qua các pkg đã cài)
RUN pip3 install --no-cache-dir -r /app/icefall/requirements.txt || true

# Copy scripts
COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh /app/scripts/*.py 2>/dev/null || true

# Tạo thư mục cần thiết
RUN mkdir -p \
    /app/icefall/egs/librispeech/ASR/zipformer/data/lang_bpe_500 \
    /app/icefall/egs/librispeech/ASR/zipformer/data/fbank \
    /app/icefall/egs/librispeech/ASR/zipformer/exp

WORKDIR /app/icefall/egs/librispeech/ASR/zipformer

ENV PYTHONPATH=/app/icefall

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
