# herzen/.devcontainer/Dockerfile

# --- Stage 1: The Foundation (Ubuntu 24.04 + CUDA 12.9) ---
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# --- Stage 2: System Tools & Dependencies ---
ENV PYTHONUNBUFFERED="1"
ENV DEBIAN_FRONTEND="noninteractive"

# Install all system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    make \
    gfortran \
    libfftw3-dev \
    liblapacke-dev \
    libpng-dev \
    libopenblas-dev \
    qt6-base-dev \
    libgl1 \
    curl \
    git \
    wget \
    openssh-client \
    ffmpeg \
    gdb \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3.13-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 3: Clone & Compile BART from Source ---
# Clone the official BART repository
RUN git clone https://github.com/mrirecon/bart.git /opt/bart

# --- Create the Makefile.local with GPU build flags ---
RUN echo "PARALLEL=1"             >  /opt/bart/Makefile.local && \
    echo "CUDA=1"                 >> /opt/bart/Makefile.local && \
    echo "CUDA_BASE=/usr/local/cuda" >> /opt/bart/Makefile.local && \
    echo "CUDA_LIB=lib64"         >> /opt/bart/Makefile.local && \
    echo "OPENBLAS=1"             >> /opt/bart/Makefile.local && \
    echo "BLAS_THREADSAFE=1"      >> /opt/bart/Makefile.local && \
    echo "DEBUG=1"                >> /opt/bart/Makefile.local


# Compile BART
RUN cd /opt/bart && make

# --- Stage 4: Set up Python Environment ---
# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set the permanent BART environment variables (for build-time)
ENV TOOLBOX_PATH=/opt/bart
ENV PATH="${TOOLBOX_PATH}:${PATH}"
ENV PYTHONPATH="${TOOLBOX_PATH}/python:${PYTHONPATH}"

# Set up venv and install Python packages
WORKDIR /workspace
COPY requirements.txt .
RUN python3.13 -m venv /opt/venv

# Install all packages from requirements.txt
RUN /root/.local/bin/uv pip install --python /opt/venv/bin/python --no-cache-dir -r requirements.txt

# --- Stage 5: Final Environment Setup ---
ENV PATH="/opt/venv/bin:${PATH}"
# Auto-activate the venv AND set all paths for interactive terminals
RUN echo "source /opt/venv/bin/activate" >> /root/.bashrc
RUN echo "export TOOLBOX_PATH=/opt/bart" >> /root/.bashrc
RUN echo "export PATH=\$TOOLBOX_PATH:\$PATH" >> /root/.bashrc
RUN echo "export PYTHONPATH=\$TOOLBOX_PATH/python:\$PYTHONPATH" >> /root/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> /root/.bashrc

# Keep the container alive
CMD ["sleep", "infinity"]
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"