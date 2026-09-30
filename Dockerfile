# =============================================================================
# TC-ResiCache: Docker environment for reproducibility
# =============================================================================
# Mirrors the validated reference environment:
#   Python 3.10 + PyTorch 2.6.0 with CUDA 12.4
#
# Build:
#   docker build -t tc-resicache:latest .
#
# Run (GPU):
#   docker run --gpus all -it -v $(pwd):/workspace tc-resicache:latest
#
# Run (CPU-only smoke tests):
#   docker run -it -v $(pwd):/workspace tc-resicache:latest
#
# Quick sanity check inside the container:
#   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Python dependencies (reference versions, see requirements.txt)
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 && \
    pip install "numpy>=1.24,<2.2" "Pillow>=10.0" "pyyaml>=6.0" "tqdm>=4.65"

COPY . /workspace

CMD ["bash"]
