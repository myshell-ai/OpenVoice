import numpy as np


def fuse_linear(spec, layers):
    if not layers:
        raise ValueError("Cannot fuse linear layers: at least one layer is required")

    if isinstance(layers[0].weight, np.ndarray):
        concatenate = np.concatenate
        zeros = np.zeros
    else:
        import torch

        concatenate = torch.cat
        zeros = torch.zeros

    spec.weight = concatenate([layer.weight for layer in layers])

    bias_dtype = None
    for layer in layers:
        if layer.has_bias():
            bias_dtype = layer.bias.dtype
            break

    if bias_dtype is not None:
        spec.bias = concatenate(
            [
                layer.bias
                if layer.has_bias()
                else zeros([layer.weight.shape[0]], dtype=bias_dtype)
                for layer in layers
            ]
        )


def permute_for_sliced_rotary(weight, num_heads, rotary_dim=None):
    """Permutes the weight to use the sliced rotary implementation."""
    if rotary_dim is not None:
        weight = weight.reshape(num_heads, weight.shape[0] // num_heads, -1)

        rotary_weight = weight[:, :rotary_dim]
        rotary_weight = permute_for_sliced_rotary(
            rotary_weight.reshape(num_heads * rotary_dim, -1), num_heads
        ).reshape(num_heads, rotary_dim, -1)

        weight[:, :rotary_dim] = rotary_weight

        return weight.reshape(-1, weight.shape[-1])

    return (
        weight.reshape(num_heads, weight.shape[0] // num_heads // 2, 2, weight.shape[1])
        .swapaxes(1, 2)
        .reshape(weight.shape[0], weight.shape[1])
    )


def smooth_activation(layer_norm, linear, activation_scales):
    """Applies the activation smoothing technique described in
    https://github.com/mit-han-lab/smoothquant.
    """
    if not isinstance(linear.weight, np.ndarray):
        linear_weight = linear.weight.numpy()
        activation_scales = activation_scales.numpy()
    else:
        linear_weight = linear.weight

    weight_scales = np.amax(np.absolute(linear_weight), axis=0)
    weight_scales = np.maximum(weight_scales, 1e-5)

    activation_scales = activation_scales.astype(weight_scales.dtype)

    scales = np.sqrt(activation_scales / weight_scales)
    scales = np.maximum(scales, 1e-5)

    if not isinstance(linear.weight, np.ndarray):
        import torch

        scales = torch.from_numpy(scales)

    layer_norm.gamma /= scales
    layer_norm.beta /= scales

    linear.weight *= scales.reshape(1, -1)


def raise_unsupported(reasons):
    message = (
        "The model you are trying to convert is not supported by CTranslate2. "
        "We identified the following reasons:\n"
    )
    for reason in reasons:
        message += "\n- " + reason
    raise ValueError(message)


class ConfigurationChecker:
    def __init__(self):
        self._unsupported_reasons = []

    def __call__(self, assert_condition, error_message):
        if not assert_condition:
            self._unsupported_reasons.append(error_message)

    def validate(self):
        if self._unsupported_reasons:
            raise_unsupported(self._unsupported_reasons)
