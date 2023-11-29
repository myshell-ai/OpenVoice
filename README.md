# OpenIVC: An Open-source Instant Voice Cloning Framework

This repository contains the open-source implementation of the instant voice cloning of @MyShell

Note that this repo is NOT the source code of the online version of @MyShell. Instead, this repo re-implements the instant voice cloning of @MyShell based on several open-source projects. There are several differences between this repo and the online version:

- base speaker model: this repo uses a speaker from TTS, while the base speaker models of @MyShell are trained on our private dataset.
- training dataset: the checkpoints provided in this repo are trained on a 10-hour dataset, while the IVC in @MyShell is trained on a 100-hour dataset.
- inference time: @MyShell adopts more techniques to accelerate the IVC (~200ms / sentence).


Code and model coming soon!
