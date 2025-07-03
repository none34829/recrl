FROM nvidia/cuda:12.2.2-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -O /tmp/conda.sh && \
    bash /tmp/conda.sh -b -p /opt/conda && \
    rm /tmp/conda.sh

# Set up conda environment
COPY requirements.txt /tmp/requirements.txt
RUN conda install -y python=3.10 && \
    conda install -y pytorch=2.3 torchvision=0.18 torchaudio=2.3 pytorch-cuda=12.2 -c pytorch -c nvidia && \
    pip install -r /tmp/requirements.txt && \
    pip install bitsandbytes-cuda122==0.43.0

# Install Git LFS
RUN apt-get update && apt-get install -y git-lfs && \
    git lfs install

# Set up working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
