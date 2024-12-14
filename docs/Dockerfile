# Use the official Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    ffmpeg \
    git \
    aria2 \
    unzip \
 && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install Python packages for the project
RUN pip install gradio==3.50.2 langid faster-whisper whisper-timestamped \
    unidecode eng-to-ipa pypinyin cn2an soundfile librosa inflect jieba silero \
    git+https://github.com/myshell-ai/MeloTTS.git

# Clone the OpenVoice project repository
RUN git clone https://github.com/myshell-ai/OpenVoice openvoice

# Change working directory to OpenVoice for further setup
WORKDIR /app/openvoice

# Install the OpenVoice package
RUN pip install -e .

# Download Japanese dictionary for MeloTTS
RUN python -m unidic download

# Download and extract checkpoints for both versions
# Version 1 checkpoints
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/OpenVoice/resolve/main/checkpoints_1226.zip -o checkpoints_1226.zip \
 && unzip checkpoints_1226.zip && rm checkpoints_1226.zip && mv checkpoints openvoice/checkpoints

# Version 2 checkpoints
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip -o checkpoints_v2_0417.zip \
 && unzip checkpoints_v2_0417.zip && rm checkpoints_v2_0417.zip && mv checkpoints_v2 openvoice/checkpoints

# Move remaining files in place
RUN mv /app/openvoice/resources /app/openvoice/openvoice/resources 

# Configure the Gradio demo application to bind to all network interfaces
RUN sed -i "s/demo.launch(debug=True, show_api=True, share=args.share)/demo.launch(server_name='0.0.0.0', debug=True, show_api=True, share=args.share)/" /app/openvoice/openvoice/openvoice_app.py

# Set the network port the container will listen on
EXPOSE 7860

WORKDIR /app/openvoice/openvoice

# Define the default command that runs when the container starts
CMD ["python3", "-m", "openvoice_app", "--share"]
