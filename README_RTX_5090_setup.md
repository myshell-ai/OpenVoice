# OpenVoice Setup Guide for RTX 5090

This guide provides specific instructions for setting up OpenVoice on systems with NVIDIA RTX 5090 GPUs. Due to the newer architecture of the RTX 5090 (CUDA capability sm_120), some additional steps are required beyond the standard installation.

## Prerequisites

- NVIDIA RTX 5090 GPU
- CUDA 12.1 or later
- Anaconda or Miniconda

## Installation Steps

1. Create and activate the conda environment:
```bash
conda create -n openvoice python=3.9
conda activate openvoice
```

2. Clone the repository:
```bash
git clone git@github.com:myshell-ai/OpenVoice.git
cd OpenVoice
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Install PyTorch with CUDA 12.1 support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```

5. Install required dependencies:
```bash
conda install numpy=1.21.6 scipy librosa -c conda-forge
```

6. Install MeloTTS:
```bash
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

7. Download NLTK data:
```bash
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```

8. Download the checkpoints:
   - For V1: Download from [here](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip) and extract to the `checkpoints` folder
   - For V2: Download from [here](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip) and extract to the `checkpoints_v2` folder

## Running the Demo

You can run the demo using:
```bash
python demo_part3.py
```

## Troubleshooting

If you encounter CUDA-related errors:
1. Make sure you're using the nightly build of PyTorch with CUDA 12.1 support
2. Verify CUDA installation with `nvidia-smi`
3. Check PyTorch CUDA availability with:
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

## Notes

- The RTX 5090 requires specific PyTorch builds due to its CUDA capability (sm_120)
- Using the nightly build of PyTorch is recommended for best compatibility
- Make sure all dependencies are installed in the correct order as specified above 