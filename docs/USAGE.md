# Usage

## Table of Content

- [Quick Use](#quick-use): directly use OpenVoice without installation.
- [Minimal Demo](#minimal-demo): for users who want to quickly try OpenVoice.
- [Linux Install](#linux-install-for-both-v1-and-v2): installation guide for developers and researchers on Linux.
- [Example Usage](#example-usage): example usage of OpenVoice V1 and V2.
- [Install on Other Platforms](#install-on-other-platforms): unofficial installation guide contributed by the community

## Quick Use

The input speech audio of OpenVoice can be in **Any Language**. OpenVoice can clone the voice in that speech audio, and use the voice to speak in multiple languages. For quick use, we recommend you to try the already deployed services:

- [British English](https://app.myshell.ai/widget/vYjqae)
- [American English](https://app.myshell.ai/widget/nEFFJf)
- [Indian English](https://app.myshell.ai/widget/V3iYze)
- [Australian English](https://app.myshell.ai/widget/fM7JVf)
- [Spanish](https://app.myshell.ai/widget/NNFFVz)
- [French](https://app.myshell.ai/widget/z2uyUz)
- [Chinese](https://app.myshell.ai/widget/fU7nUz)
- [Japanese](https://app.myshell.ai/widget/IfIB3u)
- [Korean](https://app.myshell.ai/widget/q6ZjIn)

## Minimal Demo

For users who want to quickly try OpenVoice and do not require high quality or stability, click any of the following links:

<div align="center">
    <a href="https://app.myshell.ai/bot/z6Bvua/1702636181"><img src="../resources/myshell-hd.png" height="28"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://huggingface.co/spaces/myshell-ai/OpenVoice"><img src="../resources/huggingface.png" height="32"></a>
</div>

## Linux Install (for both V1 and V2)

This section is only for developers and researchers who are familiar with Linux, Python and PyTorch.

### 1. Create a virtual environment:

- #### If you're using Conda
  ```bash
  conda create -n openvoice python=3.9
  conda activate openvoice
  ```

- #### If you're using Virtualenv
  ```bash
  python3 -m venv <your_venv_name>
  source <your_venv_name>/bin/activate
  ```

### 2. Clone the OpenVoice repository and enter the newly created directory:
```bash
git clone git@github.com:myshell-ai/OpenVoice.git
cd OpenVoice
```

### 3. Install in editable mode:
```bash
pip install -e .
```

### 4. Install project dependencies:
```bash
pip install -r requirements.txt
```

### 5. Download the appropriate model checkpoint:

- #### OpenVoice V1:
  Download the checkpoint from [here](https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_1226.zip) and extract it to the `checkpoints` folder.

- #### OpenVoice V2:
  Download the checkpoint from [here](https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip) and extract it to the `checkpoints_v2` folder.

### 6. (OpenVoice V2 only) Install [MeloTTS](https://github.com/myshell-ai/MeloTTS):
```
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

### 7. (GPU) Running OpenVoice on GPU:

If you're seeing an error about missing `libcudnn_ops_infer.so.8` (or similar) when you run OpenVoice, that means you do not have all the necessary `cuDNN` libraries installed, or they have been placed in a directory that is not in your `LD_LIBRARY_PATH` environment variable.

To fix this problem see:

[Common Issues - Missing `cuDNN` libraries](QA.md#missing-cudnn-libraries)

[Common Issues - My system can't find downloaded `cuDNN` libraries](QA.md#my-system-cant-find-downloaded-cudnn-libraries)

## Example Usage

### OpenVoice V1

**1. Flexible Voice Style Control.**
Please see [`demo_part1.ipynb`](../demo_part1.ipynb) for an example usage of how OpenVoice enables flexible style control over the cloned voice.

**2. Cross-Lingual Voice Cloning.**
Please see [`demo_part2.ipynb`](../demo_part2.ipynb) for an example for languages seen or unseen in the MSML training set.

**3. Gradio Demo.**. We provide a minimalist local gradio demo here. We strongly suggest the users to look into `demo_part1.ipynb`, `demo_part2.ipynb` and the [QnA](QA.md) if they run into issues with the gradio demo. Launch a local gradio demo with `python -m openvoice_app --share`.

### OpenVoice V2

**1. Demo usage of V2.** Please see [`demo_part3.ipynb`](../demo_part3.ipynb) for example usage of OpenVoice V2. Now it natively supports English, Spanish, French, Chinese, Japanese and Korean.

## Install on Other Platforms

This section provides the unofficial installation guides by open-source contributors in the community:

- Windows
  - [Guide](https://github.com/Alienpups/OpenVoice/blob/main/docs/USAGE_WINDOWS.md) by [@Alienpups](https://github.com/Alienpups)
  - You are welcome to contribute if you have a better installation guide. We will list you here.
- Docker
  - [Guide](https://github.com/StevenJSCF/OpenVoice/blob/update-docs/docs/DF_USAGE.md) by [@StevenJSCF](https://github.com/StevenJSCF)
  - You are welcome to contribute if you have a better installation guide. We will list you here.

## Common Issues
See [common issues](QA.md) for common issues and solutions.
