FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    gnupg2 \
    unzip \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add nvidia package repositories
RUN apt-get update && \
    apt-get install -y gnupg2 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update

# Install CUDA 12.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-toolkit-12-1 \
    libcublas-12-1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# this will fail to copy if multiple are present in the same dir
COPY ${CUDNN:-cudnn-linux-x86_64-*_cuda12-archive.tar.xz} /tmp/cudnn.tar.xz

# Extract and install cuDNN
RUN mkdir /tmp/cudnn
RUN tar -xvf /tmp/cudnn.tar.xz -C /tmp/cudnn && \
    cp -P /tmp/cudnn/*/include/cudnn*.h /usr/local/cuda-12.1/include/ && \
    cp -P /tmp/cudnn/*/lib/libcudnn* /usr/local/cuda-12.1/lib64/ && \
    chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*
RUN rm -rf /tmp/cudnn

    # Set environment variables for CUDA and cuDNN
ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

# Create a symlink for libcublas.so.11
RUN ln -s /usr/local/cuda-12.1/lib64/libcublas.so.12 /usr/local/cuda-12.1/lib64/libcublas.so.11

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Install pytorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install git+https://github.com/violetdenim/wavmark.git
RUN pip install git+https://github.com/myshell-ai/MeloTTS.git

RUN python -m unidic download

RUN pip install -e .

RUN pip install jupyter

# Port for jupyter
EXPOSE 8888

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint for the container, see entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
