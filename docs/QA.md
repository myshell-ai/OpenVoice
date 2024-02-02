# Common Questions and Answers

## General Comments

**OpenVoice is a Technology, not a Product**

Although it works on a majority of voices if used correctly, please do not expect it to work perfectly on every case, as it takes a lot of engineering effort to translate a technology to a stable product. The targeted users of this technology are developers and researchers, not end users. End users expects a perfect product. However, we are confident to say that OpenVoice is the state-of-the-art among the source-available voice cloning technologies.

The contribution of OpenVoice is a versatile instant voice cloning technical approach, not a ready-to-use perfect voice cloning product. However, we firmly believe that by releasing OpenVoice, we can accelerate the open research community's progress on instant voice cloning, and someday in the future the free voice cloning methods will be as good as commercial ones.

## Issues with Voice Quality

**Accent and Emotion of the Generated Voice is not Similar to the Reference Voice**

First of all, OpenVoice only clones the tone color of the reference speaker. It does NOT clone the accent or emotion. The accent and emotion is controlled by the base speaker TTS model, not cloned by the tone color converter (please refer to our [paper](https://arxiv.org/pdf/2312.01479.pdf) for technical details). If the user wants to change the accent or emotion of the output, they need to have a base speaker model with that accent. OpenVoice provides sufficient flexibility for users to integrate their own base speaker model into the framework by simply replacing the current base speaker we provided.

**Bad Audio Quality of the Generated Speech** 

Please check the followings:
- Is your reference audio is clean enough without any background noise? You can find some high-quality reference speech [here](https://aiartes.com/voiceai)
- Is your audio too short?
- Does your audio contain speech from more than one person?
- Does the reference audio contain long blank sections?
- Did you name the reference audio the same name you used before but forgot to delete the `processed` folder?

## Issues with Languages

**Support of Other Languages**

For multi-lingual and cross-lingual usage, please refer to [`demo_part2.ipynb`](https://github.com/myshell-ai/OpenVoice/blob/main/demo_part2.ipynb). OpenVoice supports any language as long as you have a base speaker in that language. The OpenVoice team already did the most difficult part (tone color converter training) for you. Base speaker TTS model is relatively easy to train, and multiple existing open-source repositories support it. If you don't want to train by yourself, simply use the OpenAI TTS model as the base speaker.

## Issues with Installation
**Error Related to Silero**

When calling `get_vad_segments` from `se_extractor.py`, there should be a message like this:
```
Downloading: "https://github.com/snakers4/silero-vad/zipball/master" to /home/user/.cache/torch/hub/master.zip
```
The download would fail if your machine can not access github. Please download the zip from "https://github.com/snakers4/silero-vad/zipball/master" manually and unzip it to `/home/user/.cache/torch/hub/snakers4_silero-vad_master`. You can also see [this issue](https://github.com/myshell-ai/OpenVoice/issues/57) for solutions for other versions of silero.
