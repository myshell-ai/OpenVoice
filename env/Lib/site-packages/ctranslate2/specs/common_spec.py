import enum

from ctranslate2.specs import model_spec


# This enum should match the C++ equivalent in include/ctranslate2/ops/activation.h.
class Activation(enum.IntEnum):
    """Activation type."""

    RELU = 0
    GELUTanh = 1
    SWISH = 2
    GELU = 3
    GELUSigmoid = 4
    Tanh = 5


# This enum should match the C++ equivalent in include/ctranslate2/layers/common.h.
class EmbeddingsMerge(enum.IntEnum):
    """Merge strategy for factors embeddings."""

    CONCAT = 0
    ADD = 1


class LayerNormSpec(model_spec.LayerSpec):
    def __init__(self, rms_norm=False):
        self.gamma = None
        if not rms_norm:
            self.beta = None


class LinearSpec(model_spec.LayerSpec):
    def __init__(self):
        self.weight = None
        self.weight_scale = model_spec.OPTIONAL
        self.bias = model_spec.OPTIONAL

    def has_bias(self):
        return not isinstance(self.bias, str)


class Conv1DSpec(model_spec.LayerSpec):
    def __init__(self):
        self.weight = None
        self.bias = model_spec.OPTIONAL


class EmbeddingsSpec(model_spec.LayerSpec):
    def __init__(self):
        self.weight = None
        self.weight_scale = model_spec.OPTIONAL
