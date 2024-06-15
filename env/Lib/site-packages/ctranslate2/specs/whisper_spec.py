from typing import List, Optional, Tuple

import numpy as np

from ctranslate2.specs import common_spec, model_spec, transformer_spec


class WhisperConfig(model_spec.ModelConfig):
    """Configuration for the Whisper model."""

    def __init__(
        self,
        suppress_ids: Optional[List[int]] = None,
        suppress_ids_begin: Optional[List[int]] = None,
        lang_ids: Optional[List[int]] = None,
        alignment_heads: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__(
            suppress_ids=suppress_ids,
            suppress_ids_begin=suppress_ids_begin,
            lang_ids=lang_ids,
            alignment_heads=alignment_heads,
        )


class WhisperSpec(model_spec.LanguageModelSpec):
    """Describes a Whisper model."""

    def __init__(
        self,
        num_encoder_layers,
        num_encoder_heads,
        num_decoder_layers,
        num_decoder_heads,
    ):
        """Initializes the model specification.

        Args:
          num_encoder_layers: The number of encoder layers.
          num_encoder_heads: The number of encoder attention heads.
          num_decoder_layers: The number of decoder layers.
          num_decoder_heads: The number of decoder attention heads.
        """
        super().__init__()
        self.encoder = WhisperEncoderSpec(num_encoder_layers, num_encoder_heads)
        self.decoder = transformer_spec.TransformerDecoderSpec(
            num_decoder_layers,
            num_decoder_heads,
            activation=common_spec.Activation.GELU,
        )
        self.decoder.scale_embeddings = False

    @property
    def name(self):
        return "WhisperSpec"

    @property
    def revision(self):
        return 3

    def get_default_config(self):
        return WhisperConfig()

    def get_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]


class WhisperEncoderSpec(model_spec.LayerSpec):
    def __init__(self, num_layers, num_heads):
        self.num_heads = np.dtype("int16").type(num_heads)
        self.conv1 = common_spec.Conv1DSpec()
        self.conv2 = common_spec.Conv1DSpec()
        self.position_encodings = transformer_spec.PositionEncoderSpec()
        self.layer_norm = common_spec.LayerNormSpec()
        self.layer = [
            transformer_spec.TransformerEncoderLayerSpec() for _ in range(num_layers)
        ]
