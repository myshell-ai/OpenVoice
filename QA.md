# Common Questions and Answers


## Issues with Voice Quality

**Accent of the Generated Voice is not Similar to the Reference Voice**

First of all, OpenVoice only clones the tone color of the reference speaker. It does NOT clone the accent. The accent is controlled by the base speaker TTS model, not cloned by the tone color converter (please refer to our [paper](https://arxiv.org/pdf/2312.01479.pdf) for technical details). If the user wants to change the accent of the output, they need to have a base speaker model with that accent. OpenVoice provides sufficient flexibility for users to integrate their own base speaker model into the framework by simply replacing the current base speaker we provided.

**Bad Audio Quality of the Generated Speech** 

Please check the followings:
- Is your reference audio is clean enough without any background noise?
- Is your audio too short?
- Does your audio contain speech from more than one person?
- Does the reference audio contain long blank sections?
- Did you name the reference audio the same name you used before but forgot to delete the `processed` folder?

## Issues with Languages

**Support of Other Languages**

For multi-lingual and cross-lingual usage, please refer to [`demo_part2.ipynb`](https://github.com/myshell-ai/OpenVoice/blob/main/demo_part2.ipynb). OpenVoice supports any language as long as you have a base speaker in that language. The OpenVoice team already did the most difficult part (tone color converter training) for you. Base speaker TTS model is relatively easy to train, and multiple existing open-source repositories support it. If you don't want to train by yourself, simply use the OpenAI TTS model as the base speaker.