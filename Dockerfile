FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python 3.10 and pip, as well as other dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    sudo \
    ffmpeg \
    git \
    aria2 \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# Optional: ensure python3 points to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Clone OpenVoice (or use COPY for local code)
RUN git clone https://github.com/namanthapliyal/OpenVoice.git openvoice

WORKDIR /app/openvoice

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir -e .

# Download and place checkpoints/resources
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OpenVoice/resolve/main/checkpoints_1226.zip -d /app/openvoice -o checkpoints_1226.zip && \
    unzip /app/openvoice/checkpoints_1226.zip && \
    rm checkpoints_1226.zip

EXPOSE 7860


CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]
