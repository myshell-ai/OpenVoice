import argparse
import copy
import os

from typing import Optional, Union

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, transformer_spec

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELUTanh,
    "relu": common_spec.Activation.RELU,
    "swish": common_spec.Activation.SWISH,
}


class OpenNMTTFConverter(Converter):
    """Converts OpenNMT-tf models."""

    @classmethod
    def from_config(
        cls,
        config: Union[str, dict],
        auto_config: bool = False,
        checkpoint_path: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Creates the converter from the configuration.

        Arguments:
          config: Path to the YAML configuration, or a dictionary with the loaded configuration.
          auto_config: Whether the model automatic configuration values should be used.
          checkpoint_path: Path to the checkpoint or checkpoint directory to load. If not set,
            the latest checkpoint from the model directory is loaded.
          model: If the model instance cannot be resolved from the model directory, this argument
             can be set to either the name of the model in the catalog or the path to the model
             configuration.

        Returns:
          A :class:`ctranslate2.converters.OpenNMTTFConverter` instance.
        """
        from opennmt import config as config_util
        from opennmt.utils.checkpoint import Checkpoint

        if isinstance(config, str):
            config = config_util.load_config([config])
        else:
            config = copy.deepcopy(config)

        if model is None:
            model = config_util.load_model(config["model_dir"])
        elif os.path.exists(model):
            model = config_util.load_model_from_file(model)
        else:
            model = config_util.load_model_from_catalog(model)

        if auto_config:
            config_util.merge_config(config, model.auto_config())

        data_config = config_util.try_prefix_paths(config["model_dir"], config["data"])
        model.initialize(data_config)

        checkpoint = Checkpoint.from_config(config, model)
        checkpoint_path = checkpoint.restore(checkpoint_path=checkpoint_path)
        if checkpoint_path is None:
            raise RuntimeError("No checkpoint was restored")

        model.create_variables()
        return cls(model)

    def __init__(self, model):
        """Initializes the converter.

        Arguments:
          model: An initialized and fully-built ``opennmt.models.Model`` instance.
        """
        self._model = model

    def _load(self):
        import opennmt

        if isinstance(self._model, opennmt.models.LanguageModel):
            spec_builder = TransformerDecoderSpecBuilder()
        else:
            spec_builder = TransformerSpecBuilder()

        return spec_builder(self._model)


class TransformerSpecBuilder:
    def __call__(self, model):
        import opennmt

        check = utils.ConfigurationChecker()
        check(
            isinstance(model, opennmt.models.Transformer),
            "Only Transformer models are supported",
        )
        check.validate()

        check(
            isinstance(model.encoder, opennmt.encoders.SelfAttentionEncoder),
            "Parallel encoders are not supported",
        )
        check(
            isinstance(
                model.features_inputter,
                (opennmt.inputters.WordEmbedder, opennmt.inputters.ParallelInputter),
            ),
            "Source inputter must be a WordEmbedder or a ParallelInputter",
        )
        check.validate()

        mha = model.encoder.layers[0].self_attention.layer
        ffn = model.encoder.layers[0].ffn.layer
        with_relative_position = mha.maximum_relative_position is not None
        activation_name = ffn.inner.activation.__name__

        check(
            activation_name in _SUPPORTED_ACTIVATIONS,
            "Activation %s is not supported (supported activations are: %s)"
            % (activation_name, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
        )
        check(
            with_relative_position != bool(model.encoder.position_encoder),
            "Relative position representation and position encoding cannot be both enabled "
            "or both disabled",
        )
        check(
            model.decoder.attention_reduction
            != opennmt.layers.MultiHeadAttentionReduction.AVERAGE_ALL_LAYERS,
            "Averaging all multi-head attention matrices is not supported",
        )

        source_inputters = _get_inputters(model.features_inputter)
        target_inputters = _get_inputters(model.labels_inputter)
        num_source_embeddings = len(source_inputters)
        if num_source_embeddings == 1:
            embeddings_merge = common_spec.EmbeddingsMerge.CONCAT
        else:
            reducer = model.features_inputter.reducer
            embeddings_merge = None
            if reducer is not None:
                if isinstance(reducer, opennmt.layers.ConcatReducer):
                    embeddings_merge = common_spec.EmbeddingsMerge.CONCAT
                elif isinstance(reducer, opennmt.layers.SumReducer):
                    embeddings_merge = common_spec.EmbeddingsMerge.ADD

            check(
                all(
                    isinstance(inputter, opennmt.inputters.WordEmbedder)
                    for inputter in source_inputters
                ),
                "All source inputters must WordEmbedders",
            )
            check(
                embeddings_merge is not None,
                "Unsupported embeddings reducer %s" % reducer,
            )

        alignment_layer = -1
        alignment_heads = 1
        if (
            model.decoder.attention_reduction
            == opennmt.layers.MultiHeadAttentionReduction.AVERAGE_LAST_LAYER
        ):
            alignment_heads = 0

        check.validate()

        encoder_spec = transformer_spec.TransformerEncoderSpec(
            len(model.encoder.layers),
            model.encoder.layers[0].self_attention.layer.num_heads,
            pre_norm=model.encoder.layer_norm is not None,
            activation=_SUPPORTED_ACTIVATIONS[activation_name],
            num_source_embeddings=num_source_embeddings,
            embeddings_merge=embeddings_merge,
            relative_position=with_relative_position,
        )

        decoder_spec = transformer_spec.TransformerDecoderSpec(
            len(model.decoder.layers),
            model.decoder.layers[0].self_attention.layer.num_heads,
            pre_norm=model.decoder.layer_norm is not None,
            activation=_SUPPORTED_ACTIVATIONS[activation_name],
            relative_position=with_relative_position,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        spec = transformer_spec.TransformerSpec(encoder_spec, decoder_spec)

        spec.config.add_source_bos = bool(source_inputters[0].mark_start)
        spec.config.add_source_eos = bool(source_inputters[0].mark_end)
        for inputter in source_inputters:
            spec.register_source_vocabulary(_load_vocab(inputter.vocabulary_file))
        for inputter in target_inputters:
            spec.register_target_vocabulary(_load_vocab(inputter.vocabulary_file))

        self.set_transformer_encoder(
            spec.encoder,
            model.encoder,
            model.features_inputter,
        )
        self.set_transformer_decoder(
            spec.decoder,
            model.decoder,
            model.labels_inputter,
        )

        return spec

    def set_transformer_encoder(self, spec, module, inputter):
        for embedding_spec, inputter in zip(spec.embeddings, _get_inputters(inputter)):
            self.set_embeddings(embedding_spec, inputter)
        if module.position_encoder is not None:
            self.set_position_encodings(
                spec.position_encodings,
                module.position_encoder,
            )

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_multi_head_attention(
                layer_spec.self_attention,
                layer.self_attention,
                self_attention=True,
            )

            self.set_ffn(layer_spec.ffn, layer.ffn)

        if module.layer_norm is not None:
            self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_transformer_decoder(self, spec, module, inputter):
        self.set_embeddings(spec.embeddings, inputter)
        if module.position_encoder is not None:
            self.set_position_encodings(
                spec.position_encodings,
                module.position_encoder,
            )

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_multi_head_attention(
                layer_spec.self_attention,
                layer.self_attention,
                self_attention=True,
            )

            if layer.attention:
                self.set_multi_head_attention(
                    layer_spec.attention,
                    layer.attention[0],
                    self_attention=False,
                )

            self.set_ffn(layer_spec.ffn, layer.ffn)

        if module.layer_norm is not None:
            self.set_layer_norm(spec.layer_norm, module.layer_norm)

        self.set_linear(spec.projection, module.output_layer)

    def set_ffn(self, spec, module):
        self.set_linear(spec.linear_0, module.layer.inner)
        self.set_linear(spec.linear_1, module.layer.outer)
        self.set_layer_norm_from_wrapper(spec.layer_norm, module)

    def set_multi_head_attention(self, spec, module, self_attention=False):
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        self.set_linear(split_layers[0], module.layer.linear_queries)
        self.set_linear(split_layers[1], module.layer.linear_keys)
        self.set_linear(split_layers[2], module.layer.linear_values)

        if self_attention:
            utils.fuse_linear(spec.linear[0], split_layers)
            if module.layer.maximum_relative_position is not None:
                spec.relative_position_keys = (
                    module.layer.relative_position_keys.numpy()
                )
                spec.relative_position_values = (
                    module.layer.relative_position_values.numpy()
                )
        else:
            utils.fuse_linear(spec.linear[0], split_layers[:1])
            utils.fuse_linear(spec.linear[1], split_layers[1:])

        self.set_linear(spec.linear[-1], module.layer.linear_output)
        self.set_layer_norm_from_wrapper(spec.layer_norm, module)

    def set_layer_norm_from_wrapper(self, spec, module):
        self.set_layer_norm(
            spec,
            module.output_layer_norm
            if module.input_layer_norm is None
            else module.input_layer_norm,
        )

    def set_layer_norm(self, spec, module):
        spec.gamma = module.gamma.numpy()
        spec.beta = module.beta.numpy()

    def set_linear(self, spec, module):
        spec.weight = module.kernel.numpy()
        if not module.transpose:
            spec.weight = spec.weight.transpose()
        if module.bias is not None:
            spec.bias = module.bias.numpy()

    def set_embeddings(self, spec, module):
        spec.weight = module.embedding.numpy()

    def set_position_encodings(self, spec, module):
        import opennmt

        if isinstance(module, opennmt.layers.PositionEmbedder):
            spec.encodings = module.embedding.numpy()[1:]


class TransformerDecoderSpecBuilder(TransformerSpecBuilder):
    def __call__(self, model):
        import opennmt

        check = utils.ConfigurationChecker()
        check(
            isinstance(model.decoder, opennmt.decoders.SelfAttentionDecoder),
            "Only self-attention decoders are supported",
        )
        check.validate()

        mha = model.decoder.layers[0].self_attention.layer
        ffn = model.decoder.layers[0].ffn.layer
        activation_name = ffn.inner.activation.__name__

        check(
            activation_name in _SUPPORTED_ACTIVATIONS,
            "Activation %s is not supported (supported activations are: %s)"
            % (activation_name, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
        )
        check.validate()

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            len(model.decoder.layers),
            mha.num_heads,
            pre_norm=model.decoder.layer_norm is not None,
            activation=_SUPPORTED_ACTIVATIONS[activation_name],
        )

        spec.register_vocabulary(_load_vocab(model.features_inputter.vocabulary_file))
        self.set_transformer_decoder(
            spec.decoder,
            model.decoder,
            model.features_inputter,
        )

        return spec


def _get_inputters(inputter):
    import opennmt

    return (
        inputter.inputters
        if isinstance(inputter, opennmt.inputters.MultiInputter)
        else [inputter]
    )


def _load_vocab(vocab, unk_token="<unk>"):
    import opennmt

    if isinstance(vocab, opennmt.data.Vocab):
        tokens = list(vocab.words)
    elif isinstance(vocab, list):
        tokens = list(vocab)
    elif isinstance(vocab, str):
        tokens = opennmt.data.Vocab.from_file(vocab).words
    else:
        raise TypeError("Invalid vocabulary type")

    if unk_token not in tokens:
        tokens.append(unk_token)
    return tokens


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", help="Path to the YAML configuration.")
    parser.add_argument(
        "--auto_config",
        action="store_true",
        help="Use the model automatic configuration values.",
    )
    parser.add_argument(
        "--model_path",
        help=(
            "Path to the checkpoint or checkpoint directory to load. If not set, "
            "the latest checkpoint from the model directory is loaded."
        ),
    )
    parser.add_argument(
        "--model_type",
        help=(
            "If the model instance cannot be resolved from the model directory, "
            "this argument can be set to either the name of the model in the catalog "
            "or the path to the model configuration."
        ),
    )
    parser.add_argument(
        "--src_vocab",
        help="Path to the source vocabulary (required if no configuration is set).",
    )
    parser.add_argument(
        "--tgt_vocab",
        help="Path to the target vocabulary (required if no configuration is set).",
    )
    Converter.declare_arguments(parser)
    args = parser.parse_args()

    config = args.config
    if not config:
        if not args.model_path or not args.src_vocab or not args.tgt_vocab:
            raise ValueError(
                "Options --model_path, --src_vocab, --tgt_vocab are required "
                "when a configuration is not set"
            )

        model_dir = (
            args.model_path
            if os.path.isdir(args.model_path)
            else os.path.dirname(args.model_path)
        )
        config = {
            "model_dir": model_dir,
            "data": {
                "source_vocabulary": args.src_vocab,
                "target_vocabulary": args.tgt_vocab,
            },
        }

    converter = OpenNMTTFConverter.from_config(
        config,
        auto_config=args.auto_config,
        checkpoint_path=args.model_path,
        model=args.model_type,
    )
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
