# Usage

## Table of Content

- [Quick Use](#quick-use): directly use OpenVoice without installation.
- [Linux Install](#linux-install): for researchers and developers only.
- [Common Installation Steps (V1 and V2)](#common-installation-steps-v1-and-v2): Installation steps applicable to both OpenVoice V1 and V2.
- [OpenVoice V1](#openvoice-v1): Installation and usage instructions for OpenVoice V1.
- [OpenVoice V2](#openvoice-v2): Installation and usage instructions for OpenVoice V2.
- [Windows Install (VS Code)](#windows-install-vs-code): Installation instructions for Windows users.


## Quick Use

For most users, the most convenient way is to directly use the free TTS and Instant Voice Clone services in MyShell.

- [British English](https://app.myshell.ai/widget/vYjqae)
- [American English](https://app.myshell.ai/widget/eIRjAf)
- [Indian English](https://app.myshell.ai/widget/V3iYze)
- [Australian English](https://app.myshell.ai/widget/fM7JVf)
- [Spanish](https://app.myshell.ai/widget/NNFFVz)
- [French](https://app.myshell.ai/widget/z2uyUz)
- [Chinese](https://app.myshell.ai/widget/fU7nUz)
- [Japanese](https://app.myshell.ai/widget/IfIB3u)
- [Korean](https://app.myshell.ai/widget/q6ZjIn)

OpenVoice supports any language as long as you have a base speaker in that language. The OpenVoice team already did the most difficult part (tone color converter training) for you. Base speaker TTS model is relatively easy to train, and multiple existing open-source repositories support it. If you don't want to train by yourself, simply use the OpenAI TTS model as the base speaker.

## Minimal Demo

For users who want to quickly try OpenVoice and do not require high quality or stability, click any of the following links:

<div align="center">
    <a href="https://app.myshell.ai/bot/z6Bvua/1702636181"><img src="../resources/myshell-hd.png" height="28"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://huggingface.co/spaces/myshell-ai/OpenVoice"><img src="../resources/huggingface.png" height="32"></a>
</div>

## Linux Install

This section is only for developers and researchers who are familiar with Linux, Python, and PyTorch.

### Common Installation Steps (V1 and V2)

No matter if you are using V1 or V2, the above installation is the same.

**1. Clone the Repository:**
    ```
    bash
    git clone git@github.com:myshell-ai/OpenVoice.git
    cd OpenVoice 
    ```
    
**2. Create a Python Environment:**
    ```
    conda create -n openvoice python=3.9
    conda activate openvoice
    ```

**3. Install OpenVoice:**
    ```
    pip install -e .
    ```

**3. Next Step:**
    Depending on the version you are using follow the next steps: 
- [OpenVoice V1](#openvoice-v1): Installation and usage instructions for OpenVoice V1.
- [OpenVoice V2](#openvoice-v2): Installation and usage instructions for OpenVoice V2.

### OpenVoice V1

Download the checkpoint from [here](https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_1226.zip) and extract it to the `checkpoints` folder.

**1. Flexible Voice Style Control.**
Please see [`demo_part1.ipynb`](../demo_part1.ipynb) for an example usage of how OpenVoice enables flexible style control over the cloned voice.

**2. Cross-Lingual Voice Cloning.**
Please see [`demo_part2.ipynb`](../demo_part2.ipynb) for an example for languages seen or unseen in the MSML training set.

**3. Gradio Demo.**. We provide a minimalist local gradio demo here. We strongly suggest the users to look into `demo_part1.ipynb`, `demo_part2.ipynb` and the [QnA](QA.md) if they run into issues with the gradio demo. Launch a local gradio demo with `python -m openvoice_app --share`.

### OpenVoice V2

Download the checkpoint from [here](https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip) and extract it to the `checkpoints_v2` folder.

Install [MeloTTS](https://github.com/myshell-ai/MeloTTS):
```
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

**Demo Usage.** Please see [`demo_part3.ipynb`](../demo_part3.ipynb) for example usage of OpenVoice V2. Now it natively supports English, Spanish, French, Chinese, Japanese and Korean.


## Windows Install (VS Code)

Please use [this guide](https://github.com/Alienpups/OpenVoice/blob/main/docs/USAGE_WINDOWS.md) if you want to install and use OpenVoice on Windows.



