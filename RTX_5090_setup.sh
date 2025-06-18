#!/bin/bash

# Exit on error
set -e

echo "Setting up OpenVoice for RTX 5090..."

# Create and activate conda environment
echo "Creating conda environment..."
conda create -n openvoice python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate openvoice

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch with CUDA 12.1 support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia -y

# Install required dependencies
echo "Installing required dependencies..."
conda install numpy=1.21.6 scipy librosa -c conda-forge -y

# Install MeloTTS
echo "Installing MeloTTS..."
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download

# Install NLTK data
echo "Installing NLTK data..."
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

# Install OpenVoice in development mode
echo "Installing OpenVoice..."
pip install -e .

echo "Setup completed successfully!"
echo "Please download the checkpoints:"
echo "V1: https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip"
echo "V2: https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
echo ""
echo "Extract them to the 'checkpoints' and 'checkpoints_v2' folders respectively."
echo ""
echo "You can now run the demo with: python demo_part3.py"

echo " you might need to run the following commands to setup the environment:
conda create -n openvoice python=3.9
conda activate openvoice
git clone git@github.com:myshell-ai/OpenVoice.git
cd OpenVoice
pip install -e .
"