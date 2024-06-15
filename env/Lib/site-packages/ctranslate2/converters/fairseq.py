import argparse
import os

from typing import Optional

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, transformer_spec

_SUPPORTED_MODELS = {
    "bart",
    "multilingual_transformer",
    "transformer",
    "transformer_align",
    "transformer_lm",
}


_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "gelu_accurate": common_spec.Activation.GELUTanh,
    "gelu_fast": common_spec.Activation.GELUTanh,
    "relu": common_spec.Activation.RELU,
    "swish": common_spec.Activation.SWISH,
}


def _get_model_spec(args):
    import fairseq

    activation_fn = getattr(args, "activation_fn", "relu")
    model_name = fairseq.models.ARCH_MODEL_NAME_REGISTRY[args.arch]

    check = utils.ConfigurationChecker()
    check(
        model_name in _SUPPORTED_MODELS,
        "Model '%s' used by architecture '%s' is not supported (supported models are: %s)"
        % (model_name, args.arch, ", ".join(_SUPPORTED_MODELS)),
    )
    check.validate()
    check(
        activation_fn in _SUPPORTED_ACTIVATIONS,
        "Option --activation-fn %s is not supported (supported activations are: %s)"
        % (activation_fn, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
    )
    check(
        not getattr(args, "no_token_positional_embeddings", False),
        "Option --no-token-positional-embeddings is not supported",
    )
    check(
        not getattr(args, "lang_tok_replacing_bos_eos", False),
        "Option --lang-tok-replacing-bos-eos is not supported",
    )

    if model_name == "transformer_lm":
        check(
            not args.character_embeddings,
            "Option --character-embeddings is not supported",
        )
        check(
            not args.adaptive_input,
            "Option --adaptive-input is not supported",
        )
        check.validate()

        return transformer_spec.TransformerDecoderModelSpec.from_config(
            args.decoder_layers,
            args.decoder_attention_heads,
            pre_norm=args.decoder_normalize_before,
            activation=_SUPPORTED_ACTIVATIONS[activation_fn],
            layernorm_embedding=getattr(args, "layernorm_embedding", False),
            no_final_norm=args.no_decoder_final_norm,
            project_in_out=args.decoder_input_dim != args.decoder_embed_dim,
        )

    else:
        check(
            args.encoder_normalize_before == args.decoder_normalize_before,
            "Options --encoder-normalize-before and --decoder-normalize-before "
            "must have the same value",
        )
        check(
            args.encoder_attention_heads == args.decoder_attention_heads,
            "Options --encoder-attention-heads and --decoder-attention-heads "
            "must have the same value",
        )
        check.validate()

        return transformer_spec.TransformerSpec.from_config(
            (args.encoder_layers, args.decoder_layers),
            args.encoder_attention_heads,
            pre_norm=args.encoder_normalize_before,
            activation=_SUPPORTED_ACTIVATIONS[activation_fn],
            alignment_layer=getattr(args, "alignment_layer", -1),
            alignment_heads=getattr(args, "alignment_heads", 0),
            layernorm_embedding=getattr(args, "layernorm_embedding", False),
        )


def _get_vocab(dictionary):
    return ["<blank>" if token == "<pad>" else token for token in dictionary.symbols]


class FairseqConverter(Converter):
    """Converts models trained with Fairseq."""

    def __init__(
        self,
        model_path: str,
        data_dir: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        fixed_dictionary: Optional[str] = None,
        no_default_special_tokens: bool = False,
        user_dir: Optional[str] = None,
    ):
        """Initializes the Fairseq converter.

        Arguments:
          model_path: Path to the Fairseq PyTorch model (.pt file).
          data_dir: Path to the Fairseq data directory containing vocabulary files.
          source_lang: Source language (may be required if not declared in the model).
          target_lang: Target language (may be required if not declared in the model).
          fixed_dictionary: Path to the fixed dictionary for multilingual models.
          no_default_special_tokens: Require all special tokens to be provided by the user
            (e.g. encoder end token, decoder start token).
          user_dir: Path to the user directory containing custom extensions.
        """
        self._model_path = model_path
        self._data_dir = data_dir
        self._fixed_dictionary = fixed_dictionary
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._no_default_special_tokens = no_default_special_tokens
        self._user_dir = user_dir

    def _load(self):
        import fairseq
        import torch

        from fairseq import checkpoint_utils

        if self._user_dir:
            from fairseq.utils import import_user_module

            import_user_module(argparse.Namespace(user_dir=self._user_dir))

        with torch.no_grad():
            checkpoint = checkpoint_utils.load_checkpoint_to_cpu(self._model_path)
            args = checkpoint["args"] or checkpoint["cfg"]["model"]

            args.data = self._data_dir
            if self._fixed_dictionary is not None:
                args.fixed_dictionary = self._fixed_dictionary
            if hasattr(args, "lang_dict") and args.lang_dict:
                args.lang_dict = os.path.join(
                    self._data_dir, os.path.basename(args.lang_dict)
                )

            if self._source_lang is not None:
                args.source_lang = self._source_lang

            if self._target_lang is not None:
                args.target_lang = self._target_lang

            spec = _get_model_spec(args)

            task = fairseq.tasks.setup_task(args)
            model = fairseq.models.build_model(args, task)
            model.eval()
            model.load_state_dict(checkpoint["model"])

            if isinstance(spec, transformer_spec.TransformerDecoderModelSpec):
                set_transformer_decoder(
                    spec.decoder,
                    model.decoder,
                    with_encoder_attention=False,
                )

                spec.register_vocabulary(_get_vocab(task.dictionary))
                if not args.add_bos_token:
                    spec.config.bos_token = spec.config.eos_token

            else:
                set_transformer_encoder(spec.encoder, model.encoder)
                set_transformer_decoder(spec.decoder, model.decoder)

                spec.register_source_vocabulary(_get_vocab(task.source_dictionary))
                spec.register_target_vocabulary(_get_vocab(task.target_dictionary))
                if self._no_default_special_tokens:
                    spec.config.decoder_start_token = None
                else:
                    spec.config.decoder_start_token = spec.config.eos_token
                    spec.config.add_source_eos = True

            return spec


def set_transformer_encoder(spec, module):
    set_input_layers(spec, module)
    for layer_spec, layer in zip(spec.layer, module.layers):
        set_transformer_encoder_layer(layer_spec, layer)
    if module.layer_norm is not None:
        set_layer_norm(spec.layer_norm, module.layer_norm)
    if module.layernorm_embedding is not None:
        set_layer_norm(spec.layernorm_embedding, module.layernorm_embedding)


def set_transformer_decoder(spec, module, with_encoder_attention=True):
    set_input_layers(spec, module)
    set_linear(spec.projection, module.output_projection)
    for layer_spec, layer in zip(spec.layer, module.layers):
        set_transformer_decoder_layer(
            layer_spec,
            layer,
            with_encoder_attention=with_encoder_attention,
        )
    if module.layer_norm is not None:
        set_layer_norm(spec.layer_norm, module.layer_norm)
    if module.layernorm_embedding is not None:
        set_layer_norm(spec.layernorm_embedding, module.layernorm_embedding)
    if module.project_in_dim is not None:
        set_linear(spec.project_in, module.project_in_dim)
    if module.project_out_dim is not None:
        set_linear(spec.project_out, module.project_out_dim)


def set_input_layers(spec, module):
    set_position_encodings(spec.position_encodings, module.embed_positions)
    set_embeddings(
        spec.embeddings[0] if isinstance(spec.embeddings, list) else spec.embeddings,
        module.embed_tokens,
    )
    spec.scale_embeddings = module.embed_scale


def set_transformer_encoder_layer(spec, module):
    set_ffn(spec.ffn, module)
    set_multi_head_attention(spec.self_attention, module.self_attn, self_attention=True)
    set_layer_norm(spec.self_attention.layer_norm, module.self_attn_layer_norm)


def set_transformer_decoder_layer(spec, module, with_encoder_attention=True):
    set_ffn(spec.ffn, module)
    set_multi_head_attention(spec.self_attention, module.self_attn, self_attention=True)
    set_layer_norm(spec.self_attention.layer_norm, module.self_attn_layer_norm)
    if with_encoder_attention:
        set_multi_head_attention(spec.attention, module.encoder_attn)
        set_layer_norm(spec.attention.layer_norm, module.encoder_attn_layer_norm)


def set_ffn(spec, module):
    set_layer_norm(spec.layer_norm, module.final_layer_norm)
    set_linear(spec.linear_0, module.fc1)
    set_linear(spec.linear_1, module.fc2)


def set_multi_head_attention(spec, module, self_attention=False):
    if self_attention:
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        set_linear(split_layers[0], module.q_proj)
        set_linear(split_layers[1], module.k_proj)
        set_linear(split_layers[2], module.v_proj)
        utils.fuse_linear(spec.linear[0], split_layers)
    else:
        set_linear(spec.linear[0], module.q_proj)
        split_layers = [common_spec.LinearSpec() for _ in range(2)]
        set_linear(split_layers[0], module.k_proj)
        set_linear(split_layers[1], module.v_proj)
        utils.fuse_linear(spec.linear[1], split_layers)
    set_linear(spec.linear[-1], module.out_proj)


def set_layer_norm(spec, module):
    spec.gamma = module.weight.numpy()
    spec.beta = module.bias.numpy()


def set_linear(spec, module):
    spec.weight = module.weight.numpy()
    if module.bias is not None:
        spec.bias = module.bias.numpy()


def set_embeddings(spec, module):
    spec.weight = module.weight.numpy()


def set_position_encodings(spec, module):
    import torch

    weight = module.weight if isinstance(module, torch.nn.Embedding) else module.weights
    spec.encodings = weight.numpy()[module.padding_idx + 1 :]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", required=True, help="Model path.")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Data directory containing the source and target vocabularies.",
    )
    parser.add_argument(
        "--user_dir",
        help="Directory containing custom extensions.",
    )
    parser.add_argument(
        "--fixed_dictionary",
        help="Fixed dictionary for multilingual models.",
    )
    parser.add_argument(
        "--source_lang",
        help="Source language. This argument is used to find dictionary file from `data_dir`.",
    )
    parser.add_argument(
        "--target_lang",
        help="Target language. This argument is used to find dictionary file from `data_dir`.",
    )
    parser.add_argument(
        "--no_default_special_tokens",
        action="store_true",
        help=(
            "Require all special tokens to be provided by the user during inference, "
            "including the decoder start token."
        ),
    )
    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = FairseqConverter(
        args.model_path,
        args.data_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        fixed_dictionary=args.fixed_dictionary,
        no_default_special_tokens=args.no_default_special_tokens,
        user_dir=args.user_dir,
    )
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
