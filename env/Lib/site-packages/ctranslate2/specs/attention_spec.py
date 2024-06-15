import enum

import numpy as np

from ctranslate2.specs import common_spec, model_spec


# This enum should match the C++ equivalent in include/ctranslate2/layers/attention.h.
class RotaryScalingType(enum.IntEnum):
    """RoPE scaling type."""

    Linear = 0


class MultiHeadAttentionSpec(model_spec.LayerSpec):
    def __init__(
        self,
        self_attention=False,
        relative_position=False,
        relative_attention_bias=False,
        rms_norm=False,
        rotary_dim=None,
        rotary_interleave=True,
        rotary_scaling_type=None,
        rotary_scaling_factor=1,
        rotary_base=10000,
        num_heads_kv=None,
        sliding_window=None,
    ):
        self.queries_scale = model_spec.OPTIONAL

        self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        self.linear = [
            common_spec.LinearSpec() for _ in range(2 if self_attention else 3)
        ]

        if relative_position:
            self.relative_position_keys = None
            self.relative_position_values = None

        if relative_attention_bias:
            self.relative_attention_bias = None
            self.relative_attention_max_distance = None

        if rotary_dim is not None:
            self.rotary_dim = np.dtype("int32").type(rotary_dim)
            self.rotary_interleave = rotary_interleave
            self.rotary_base = np.dtype("float32").type(rotary_base)

            if rotary_scaling_type is not None:
                self.rotary_scaling_type = np.dtype("int8").type(rotary_scaling_type)
                self.rotary_scaling_factor = np.dtype("float32").type(
                    rotary_scaling_factor
                )

        if num_heads_kv is not None:
            self.num_heads_kv = np.dtype("int32").type(num_heads_kv)

        if sliding_window is not None:
            self.sliding_window = np.dtype("int32").type(sliding_window)
