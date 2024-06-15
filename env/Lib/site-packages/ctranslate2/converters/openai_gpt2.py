import argparse
import json
import os

from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, model_spec, transformer_spec


class OpenAIGPT2Converter(Converter):
    """Converts GPT-2 models from https://github.com/openai/gpt-2."""

    def __init__(self, model_dir: str):
        """Initializes the OpenAI GPT-2 converter.

        Arguments:
          model_dir: Path to the OpenAI GPT-2 model directory.
        """
        self._model_dir = model_dir

    def _load(self):
        import tensorflow as tf

        reader = tf.train.load_checkpoint(self._model_dir)
        weights = {
            name: reader.get_tensor(name)
            for name in reader.get_variable_to_shape_map().keys()
        }

        with open(os.path.join(self._model_dir, "hparams.json")) as hparams_file:
            hparams = json.load(hparams_file)
        with open(os.path.join(self._model_dir, "encoder.json")) as vocab_file:
            vocab = json.load(vocab_file)
            vocab = [
                token
                for token, index in sorted(vocab.items(), key=lambda item: item[1])
            ]

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            hparams["n_layer"],
            hparams["n_head"],
            pre_norm=True,
            activation=common_spec.Activation.GELUTanh,
        )
        set_decoder(spec.decoder, weights, "model")
        spec.unk_token = "<|endoftext|>"
        spec.bos_token = "<|endoftext|>"
        spec.eos_token = "<|endoftext|>"
        spec.register_vocabulary(vocab)
        return spec


def set_decoder(spec, weights, scope):
    spec.embeddings.weight = weights["%s/wte" % scope]
    spec.position_encodings.encodings = weights["%s/wpe" % scope]
    spec.scale_embeddings = False
    spec.projection.weight = spec.embeddings.weight
    set_layer_norm(spec.layer_norm, weights, "%s/ln_f" % scope)
    for i, layer_spec in enumerate(spec.layer):
        set_layer(layer_spec, weights, "%s/h%d" % (scope, i))


def set_layer_norm(spec, weights, scope):
    spec.gamma = weights["%s/g" % scope]
    spec.beta = weights["%s/b" % scope]


def set_linear(spec, weights, scope):
    spec.weight = weights["%s/w" % scope].squeeze().transpose()
    spec.bias = weights["%s/b" % scope]


def set_layer(spec, weights, scope):
    set_layer_norm(spec.self_attention.layer_norm, weights, "%s/ln_1" % scope)
    set_linear(spec.self_attention.linear[0], weights, "%s/attn/c_attn" % scope)
    set_linear(spec.self_attention.linear[1], weights, "%s/attn/c_proj" % scope)
    set_layer_norm(spec.ffn.layer_norm, weights, "%s/ln_2" % scope)
    set_linear(spec.ffn.linear_0, weights, "%s/mlp/c_fc" % scope)
    set_linear(spec.ffn.linear_1, weights, "%s/mlp/c_proj" % scope)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir", required=True, help="Path to the model directory."
    )
    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = OpenAIGPT2Converter(args.model_dir)
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
