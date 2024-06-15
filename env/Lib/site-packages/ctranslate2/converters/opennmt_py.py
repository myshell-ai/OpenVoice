import argparse

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, transformer_spec

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "fast_gelu": common_spec.Activation.GELUTanh,
    "relu": common_spec.Activation.RELU,
    "silu": common_spec.Activation.SWISH,
}

_SUPPORTED_FEATURES_MERGE = {
    "concat": common_spec.EmbeddingsMerge.CONCAT,
    "sum": common_spec.EmbeddingsMerge.ADD,
}


def check_opt(opt, num_source_embeddings):
    with_relative_position = getattr(opt, "max_relative_positions", 0) > 0
    with_rotary = getattr(opt, "max_relative_positions", 0) == -1
    with_alibi = getattr(opt, "max_relative_positions", 0) == -2
    activation_fn = getattr(opt, "pos_ffn_activation_fn", "relu")
    feat_merge = getattr(opt, "feat_merge", "concat")
    self_attn_type = getattr(opt, "self_attn_type", "scaled-dot")

    check = utils.ConfigurationChecker()
    check(
        opt.encoder_type == opt.decoder_type
        and opt.decoder_type in {"transformer", "transformer_lm"},
        "Options --encoder_type and --decoder_type must be"
        " 'transformer' or 'transformer_lm",
    )
    check(
        self_attn_type == "scaled-dot",
        "Option --self_attn_type %s is not supported (supported values are: scaled-dot)"
        % self_attn_type,
    )
    check(
        activation_fn in _SUPPORTED_ACTIVATIONS,
        "Option --pos_ffn_activation_fn %s is not supported (supported activations are: %s)"
        % (activation_fn, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
    )
    check(
        opt.position_encoding != (with_relative_position or with_rotary or with_alibi),
        "Options --position_encoding and --max_relative_positions cannot be both enabled "
        "or both disabled",
    )
    check(
        num_source_embeddings == 1 or feat_merge in _SUPPORTED_FEATURES_MERGE,
        "Option --feat_merge %s is not supported (supported merge modes are: %s)"
        % (feat_merge, " ".join(_SUPPORTED_FEATURES_MERGE.keys())),
    )
    check.validate()


def _get_model_spec_seq2seq(
    opt, variables, src_vocabs, tgt_vocabs, num_source_embeddings
):
    """Creates a model specification from the model options."""
    with_relative_position = getattr(opt, "max_relative_positions", 0) > 0
    activation_fn = getattr(opt, "pos_ffn_activation_fn", "relu")
    feat_merge = getattr(opt, "feat_merge", "concat")

    # Return the first head of the last layer unless the model was trained with alignments.
    if getattr(opt, "lambda_align", 0) == 0:
        alignment_layer = -1
        alignment_heads = 1
    else:
        alignment_layer = opt.alignment_layer
        alignment_heads = opt.alignment_heads

    num_heads = getattr(opt, "heads", 8)

    model_spec = transformer_spec.TransformerSpec.from_config(
        (opt.enc_layers, opt.dec_layers),
        num_heads,
        with_relative_position=with_relative_position,
        activation=_SUPPORTED_ACTIVATIONS[activation_fn],
        alignment_layer=alignment_layer,
        alignment_heads=alignment_heads,
        num_source_embeddings=num_source_embeddings,
        embeddings_merge=_SUPPORTED_FEATURES_MERGE[feat_merge],
        multi_query_attention=getattr(opt, "multiquery", False),
    )

    model_spec.config.decoder_start_token = getattr(opt, "decoder_start_token", "<s>")

    set_transformer_spec(model_spec, variables)
    for src_vocab in src_vocabs:
        model_spec.register_source_vocabulary(src_vocab)
    for tgt_vocab in tgt_vocabs:
        model_spec.register_target_vocabulary(tgt_vocab)

    return model_spec


def _get_model_spec_lm(opt, variables, src_vocabs, tgt_vocabs, num_source_embeddings):
    """Creates a model specification from the model options."""
    with_relative_position = getattr(opt, "max_relative_positions", 0) > 0
    with_rotary = getattr(opt, "max_relative_positions", 0) == -1
    with_alibi = getattr(opt, "max_relative_positions", 0) == -2
    activation_fn = getattr(opt, "pos_ffn_activation_fn", "relu")
    num_heads = getattr(opt, "heads", 8)
    num_kv = getattr(opt, "num_kv", 0)
    if num_kv == num_heads or num_kv == 0:
        num_kv = None
    rotary_dim = 0 if with_rotary else None
    rotary_interleave = getattr(opt, "rotary_interleave", True)
    ffn_glu = activation_fn == "silu"
    sliding_window = getattr(opt, "sliding_window", 0)

    model_spec = transformer_spec.TransformerDecoderModelSpec.from_config(
        opt.dec_layers,
        num_heads,
        activation=_SUPPORTED_ACTIVATIONS[activation_fn],
        ffn_glu=ffn_glu,
        with_relative_position=with_relative_position,
        alibi=with_alibi,
        rms_norm=opt.layer_norm == "rms",
        rotary_dim=rotary_dim,
        rotary_interleave=rotary_interleave,
        multi_query_attention=getattr(opt, "multiquery", False),
        num_heads_kv=num_kv,
        sliding_window=sliding_window,
    )

    model_spec.config.layer_norm_epsilon = getattr(opt, "norm_eps", 1e-6)

    set_transformer_decoder(
        model_spec.decoder,
        variables,
        with_encoder_attention=False,
    )

    for tgt_vocab in tgt_vocabs:
        model_spec.register_vocabulary(tgt_vocab)

    return model_spec


def get_vocabs(vocab):
    if isinstance(vocab, dict) and "src" in vocab:
        if isinstance(vocab["src"], list):
            src_vocabs = [vocab["src"]]
            tgt_vocabs = [vocab["tgt"]]

            src_feats = vocab.get("src_feats")
            if src_feats is not None:
                src_vocabs.extend(src_feats.values())
        else:
            src_vocabs = [field[1].vocab.itos for field in vocab["src"].fields]
            tgt_vocabs = [field[1].vocab.itos for field in vocab["tgt"].fields]
    else:
        # Compatibility with older models.
        src_vocabs = [vocab[0][1].itos]
        tgt_vocabs = [vocab[1][1].itos]

    return src_vocabs, tgt_vocabs


class OpenNMTPyConverter(Converter):
    """Converts models generated by OpenNMT-py."""

    def __init__(self, model_path: str):
        """Initializes the OpenNMT-py converter.

        Arguments:
          model_path: Path to the OpenNMT-py PyTorch model (.pt file).
        """
        self._model_path = model_path

    def _load(self):
        import torch

        checkpoint = torch.load(self._model_path, map_location="cpu")

        src_vocabs, tgt_vocabs = get_vocabs(checkpoint["vocab"])

        check_opt(checkpoint["opt"], num_source_embeddings=len(src_vocabs))

        variables = checkpoint["model"]
        variables.update(
            {
                "generator.%s" % key: value
                for key, value in checkpoint["generator"].items()
            }
        )

        if checkpoint["opt"].decoder_type == "transformer_lm":
            return _get_model_spec_lm(
                checkpoint["opt"],
                variables,
                src_vocabs,
                tgt_vocabs,
                num_source_embeddings=len(src_vocabs),
            )
        else:
            return _get_model_spec_seq2seq(
                checkpoint["opt"],
                variables,
                src_vocabs,
                tgt_vocabs,
                num_source_embeddings=len(src_vocabs),
            )


def set_transformer_spec(spec, variables):
    set_transformer_encoder(spec.encoder, variables)
    set_transformer_decoder(spec.decoder, variables)


def set_transformer_encoder(spec, variables):
    set_input_layers(spec, variables, "encoder")
    set_layer_norm(spec.layer_norm, variables, "encoder.layer_norm")
    for i, layer in enumerate(spec.layer):
        set_transformer_encoder_layer(layer, variables, "encoder.transformer.%d" % i)


def set_transformer_decoder(spec, variables, with_encoder_attention=True):
    set_input_layers(spec, variables, "decoder")
    set_layer_norm(spec.layer_norm, variables, "decoder.layer_norm")
    for i, layer in enumerate(spec.layer):
        set_transformer_decoder_layer(
            layer,
            variables,
            "decoder.transformer_layers.%d" % i,
            with_encoder_attention=with_encoder_attention,
        )

    try:
        set_linear(spec.projection, variables, "generator")
    except KeyError:
        # Compatibility when the generator was a nn.Sequential module.
        set_linear(spec.projection, variables, "generator.0")


def set_input_layers(spec, variables, scope):
    if hasattr(spec, "position_encodings"):
        set_position_encodings(
            spec.position_encodings,
            variables,
            "%s.embeddings.make_embedding.pe" % scope,
        )
    else:
        # See https://github.com/OpenNMT/OpenNMT-py/issues/1722
        spec.scale_embeddings = False

    embeddings_specs = spec.embeddings
    if not isinstance(embeddings_specs, list):
        embeddings_specs = [embeddings_specs]

    for i, embeddings_spec in enumerate(embeddings_specs):
        set_embeddings(
            embeddings_spec,
            variables,
            "%s.embeddings.make_embedding.emb_luts.%d" % (scope, i),
        )


def set_transformer_encoder_layer(spec, variables, scope):
    set_ffn(spec.ffn, variables, "%s.feed_forward" % scope)
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s.self_attn" % scope,
        self_attention=True,
    )
    set_layer_norm(spec.self_attention.layer_norm, variables, "%s.layer_norm" % scope)


def set_transformer_decoder_layer(spec, variables, scope, with_encoder_attention=True):
    set_ffn(spec.ffn, variables, "%s.feed_forward" % scope)
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s.self_attn" % scope,
        self_attention=True,
    )
    set_layer_norm(spec.self_attention.layer_norm, variables, "%s.layer_norm_1" % scope)
    if with_encoder_attention:
        set_multi_head_attention(spec.attention, variables, "%s.context_attn" % scope)
        set_layer_norm(spec.attention.layer_norm, variables, "%s.layer_norm_2" % scope)


def set_ffn(spec, variables, scope):
    set_layer_norm(spec.layer_norm, variables, "%s.layer_norm" % scope)
    set_linear(spec.linear_0, variables, "%s.w_1" % scope)
    set_linear(spec.linear_1, variables, "%s.w_2" % scope)
    if hasattr(spec, "linear_0_noact"):
        set_linear(spec.linear_0_noact, variables, "%s.w_3" % scope)


def set_multi_head_attention(spec, variables, scope, self_attention=False):
    if self_attention:
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        set_linear(split_layers[0], variables, "%s.linear_query" % scope)
        set_linear(split_layers[1], variables, "%s.linear_keys" % scope)
        set_linear(split_layers[2], variables, "%s.linear_values" % scope)
        utils.fuse_linear(spec.linear[0], split_layers)
    else:
        set_linear(spec.linear[0], variables, "%s.linear_query" % scope)
        split_layers = [common_spec.LinearSpec() for _ in range(2)]
        set_linear(split_layers[0], variables, "%s.linear_keys" % scope)
        set_linear(split_layers[1], variables, "%s.linear_values" % scope)
        utils.fuse_linear(spec.linear[1], split_layers)
    set_linear(spec.linear[-1], variables, "%s.final_linear" % scope)
    if hasattr(spec, "relative_position_keys"):
        spec.relative_position_keys = _get_variable(
            variables, "%s.relative_positions_embeddings.weight" % scope
        )
        spec.relative_position_values = spec.relative_position_keys


def set_layer_norm(spec, variables, scope):
    try:
        spec.gamma = _get_variable(variables, "%s.weight" % scope)
    except KeyError:
        # Compatibility with older models using a custom LayerNorm module.
        spec.gamma = _get_variable(variables, "%s.a_2" % scope)
        spec.beta = _get_variable(variables, "%s.b_2" % scope)
    try:
        spec.beta = _get_variable(variables, "%s.bias" % scope)
    except KeyError:
        pass


def set_linear(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)
    bias = variables.get("%s.bias" % scope)
    if bias is not None:
        spec.bias = bias


def set_embeddings(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)


def set_position_encodings(spec, variables, scope):
    spec.encodings = _get_variable(variables, "%s.pe" % scope).squeeze()


def _get_variable(variables, name):
    return variables[name]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", required=True, help="Model path.")
    Converter.declare_arguments(parser)
    args = parser.parse_args()
    OpenNMTPyConverter(args.model_path).convert_from_args(args)


if __name__ == "__main__":
    main()
