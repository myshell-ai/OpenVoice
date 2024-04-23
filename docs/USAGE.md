# Usage

## Table of Content

- [Quick Use](#quick-use): directly use OpenVoice without installation.
- [Linux Install](#linux-install): for researchers and developers only.
    - [V1](#openvoice-v1)
    - [V2](#openvoice-v2)
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

## Linux Install

This section is only for developers and researchers who are familiar with Linux, Python and PyTorch. Clone this repo, and run

```
conda create -n openvoice python=3.9
conda activate openvoice
git clone git@github.com:myshell-ai/OpenVoice.git
cd OpenVoice
pip install -e .
```

No matter if you are using V1 or V2, the above installation is the same.

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


## Install on Other Platforms

This section provides the unofficial installation guides by open-source contributors in the community:

- Windows
  - [Guide](https://github.com/Alienpups/OpenVoice/blob/main/docs/USAGE_WINDOWS.md) by [@Alienpups](https://github.com/Alienpups)
  - You are welcome to contribute if you have a better installation guide. We will list you here.
- Docker
  - [Guide](https://github.com/StevenJSCF/OpenVoice/blob/update-docs/docs/DF_USAGE.md) by [@StevenJSCF](https://github.com/StevenJSCF)
  - You are welcome to contribute if you have a better installation guide. We will list you here.
