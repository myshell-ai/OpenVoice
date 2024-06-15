"""Declares specification of the Transformer model."""

from typing import Optional, Tuple, Union

import numpy as np

from ctranslate2.specs import attention_spec, common_spec, model_spec


class TransformerEncoderSpec(model_spec.LayerSpec):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        no_final_norm: bool = False,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        num_source_embeddings: int = 1,
        embeddings_merge: common_spec.EmbeddingsMerge = common_spec.EmbeddingsMerge.CONCAT,
        layernorm_embedding: bool = False,
        relative_position: bool = False,
        relative_attention_bias: bool = False,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        multi_query_attention: bool = False,
    ):
        """Initializes a Transformer encoder specification.

        Args:
          num_layers: Number of layers.
          num_heads: Number of attention heads.
          pre_norm: Enable the pre-norm Transformer architecture.
          no_final_norm: Disable the final layer norm in the pre-norm architecture.
          activation: Activation to apply in the feed-forward network.
          num_source_embeddings: Number of source embeddings.
          embeddings_merge: When :obj:`num_source_embeddings` > 1, specify how the
            embeddings are merged.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          relative_position: Use relative position representations in the self-attention
            layers as described in https://arxiv.org/abs/1803.02155.
          relative_attention_bias: Use relative attention bias in the self-attention
            layers as described in the T5 paper https://arxiv.org/abs/1910.10683.
          ffn_glu: Use gated linear units in the FFN layers as described in
            https://arxiv.org/abs/2002.05202.
          rms_norm: Use the root mean square layer normalization.
          multi_query_attention: Use multi-query attention.
        """
        self.num_heads = np.dtype("int16").type(num_heads)
        self.pre_norm = pre_norm
        self.activation = np.dtype("int8").type(activation)
        self.embeddings_merge = np.dtype("int8").type(embeddings_merge)
        self.embeddings = [
            common_spec.EmbeddingsSpec() for _ in range(num_source_embeddings)
        ]
        self.scale_embeddings = True
        if not relative_position and not relative_attention_bias:
            self.position_encodings = PositionEncoderSpec()
        if pre_norm and not no_final_norm:
            self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        if layernorm_embedding:
            self.layernorm_embedding = common_spec.LayerNormSpec(rms_norm=rms_norm)
        self.layer = [
            TransformerEncoderLayerSpec(
                relative_position=relative_position,
                relative_attention_bias=relative_attention_bias,
                ffn_glu=ffn_glu,
                rms_norm=rms_norm,
                num_heads_kv=1 if multi_query_attention else None,
            )
            for _ in range(num_layers)
        ]


class TransformerDecoderSpec(model_spec.LayerSpec):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        layernorm_embedding: bool = False,
        with_encoder_attention: bool = True,
        no_final_norm: bool = False,
        project_in_out: bool = False,
        relative_position: bool = False,
        relative_attention_bias: bool = False,
        alignment_layer: int = -1,
        alignment_heads: int = 1,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        alibi: bool = False,
        alibi_use_positive_positions: bool = False,
        scale_alibi: bool = False,
        rotary_dim: Optional[int] = None,
        rotary_interleave: bool = True,
        rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = None,
        rotary_scaling_factor: float = 1,
        rotary_base: float = 10000,
        parallel_residual: bool = False,
        shared_layer_norm: bool = False,
        multi_query_attention: bool = False,
        num_heads_kv: Optional[int] = None,
        sliding_window: Optional[int] = None,
    ):
        """Initializes a Transformer decoder specification.

        Args:
          num_layers: Number of layers.
          num_heads: Number of attention heads.
          pre_norm: Enable the pre-norm Transformer architecture.
          activation: Activation to apply in the feed-forward network.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          with_encoder_attention: Enable the encoder attention sublayers.
          no_final_norm: Disable the final layer norm in the pre-norm architecture.
          project_in_out: Add linear transformations after the embedding layer and before
            the final layer.
          relative_position: Use relative position representations in the self-attention
            layers as described in https://arxiv.org/abs/1803.02155.
          relative_attention_bias: Use relative attention bias in the self-attention
            layers as described in the T5 paper https://arxiv.org/abs/1910.10683.
          alignment_layer: Layer index selected for alignment.
          alignment_heads: Number of attention heads selected for alignment.
          ffn_glu: Use gated linear units in the FFN layers as described in
            https://arxiv.org/abs/2002.05202.
          rms_norm: Use the root mean square layer normalization.
          alibi: Use attention with linear biases.
          alibi_use_positive_positions: Use positive positions in the ALiBi definition.
          scale_alibi: Apply the dot product scale factor to ALiBi.
          rotary_dim: Apply rotary embeddings to these first N dimensions. If 0, rotary
            embeddings are applied to all dimensions.
          rotary_interleave: Interleave the head dimensions when rotary embeddings are applied.
            Otherwise the head dimensions are sliced in half.
          rotary_scaling_type: Type of RoPE scaling.
          rotary_scaling_factor: Factor used in the RoPE scaling.
          rotary_base: The base period of the rotary embeddings.
          parallel_residual: Use parallel residual connections in each layer block, as used
            by the GPT-J and GPT-NeoX models.
          shared_layer_norm: When using parallel residual, share the input and post
            attention layer norms.
          multi_query_attention: Use multi-query attention (alias for num_heads_kv=1).
          num_heads_kv: Number of attention heads for the key and value.
          sliding_window: Max sequence length to retain in KV Cache.
        """
        if parallel_residual:
            if not pre_norm:
                raise ValueError("The GPT-J block expects a pre-norm architecture")
            if with_encoder_attention:
                raise ValueError("The GPT-J block does not have cross attention")

        if multi_query_attention:
            if num_heads_kv is not None and num_heads_kv != 1:
                raise ValueError(
                    "Enabling multi_query_attention implies num_heads_kv=1"
                )
            num_heads_kv = 1

        if with_encoder_attention and num_heads_kv not in (None, 1, num_heads):
            raise ValueError(
                "num_heads_kv=%d is not supported in the cross-attention layers"
                % num_heads_kv
            )

        self.num_heads = np.dtype("int16").type(num_heads)
        self.pre_norm = pre_norm
        self.activation = np.dtype("int8").type(activation)
        self.alignment_layer = np.dtype("int16").type(alignment_layer)
        self.alignment_heads = np.dtype("int16").type(alignment_heads)
        self.embeddings = common_spec.EmbeddingsSpec()
        self.scale_embeddings = True
        self.scale_outputs = model_spec.OPTIONAL
        self.alibi = alibi
        self.alibi_use_positive_positions = alibi_use_positive_positions
        self.scale_alibi = scale_alibi
        if sliding_window is not None:
            self.sliding_window = np.dtype("int32").type(sliding_window)
        if (
            not relative_position
            and not relative_attention_bias
            and not alibi
            and rotary_dim is None
        ):
            self.position_encodings = PositionEncoderSpec()
        if pre_norm and not no_final_norm:
            self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        if layernorm_embedding:
            self.layernorm_embedding = common_spec.LayerNormSpec(rms_norm=rms_norm)
        self.projection = common_spec.LinearSpec()
        self.layer = [
            TransformerDecoderLayerSpec(
                with_encoder_attention=with_encoder_attention,
                relative_position=relative_position,
                relative_attention_bias=relative_attention_bias,
                ffn_glu=ffn_glu,
                rms_norm=rms_norm,
                rotary_dim=rotary_dim,
                rotary_interleave=rotary_interleave,
                rotary_scaling_type=rotary_scaling_type,
                rotary_scaling_factor=rotary_scaling_factor,
                rotary_base=rotary_base,
                parallel_residual=parallel_residual,
                shared_layer_norm=shared_layer_norm,
                num_heads_kv=num_heads_kv,
                sliding_window=sliding_window,
            )
            for _ in range(num_layers)
        ]
        self.start_from_zero_embedding = False

        if project_in_out:
            self.project_in = common_spec.LinearSpec()
            self.project_out = common_spec.LinearSpec()


class TransformerEncoderLayerSpec(model_spec.LayerSpec):
    def __init__(
        self,
        relative_position=False,
        relative_attention_bias=False,
        ffn_glu=False,
        rms_norm=False,
        num_heads_kv=None,
        sliding_window=None,
    ):
        self.self_attention = attention_spec.MultiHeadAttentionSpec(
            self_attention=True,
            relative_position=relative_position,
            relative_attention_bias=relative_attention_bias,
            rms_norm=rms_norm,
            num_heads_kv=num_heads_kv,
            sliding_window=sliding_window,
        )
        self.ffn = FeedForwardSpec(glu=ffn_glu, rms_norm=rms_norm)


class TransformerDecoderLayerSpec(model_spec.LayerSpec):
    def __init__(
        self,
        with_encoder_attention=True,
        relative_position=False,
        relative_attention_bias=False,
        ffn_glu=False,
        rms_norm=False,
        rotary_dim=None,
        rotary_interleave=True,
        rotary_scaling_type=None,
        rotary_scaling_factor=1,
        rotary_base=10000,
        parallel_residual=False,
        shared_layer_norm=False,
        num_heads_kv=None,
        sliding_window=None,
    ):
        self.self_attention = attention_spec.MultiHeadAttentionSpec(
            self_attention=True,
            relative_position=relative_position,
            relative_attention_bias=relative_attention_bias,
            rms_norm=rms_norm,
            rotary_dim=rotary_dim,
            rotary_interleave=rotary_interleave,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=rotary_base,
            num_heads_kv=num_heads_kv,
            sliding_window=sliding_window,
        )

        if with_encoder_attention:
            self.attention = attention_spec.MultiHeadAttentionSpec(
                rms_norm=rms_norm,
                num_heads_kv=num_heads_kv,
                sliding_window=sliding_window,
            )

        self.ffn = FeedForwardSpec(glu=ffn_glu, rms_norm=rms_norm)

        if parallel_residual:
            if shared_layer_norm:
                self.shared_layer_norm = common_spec.LayerNormSpec()
            else:
                self.input_layer_norm = common_spec.LayerNormSpec()
                self.post_attention_layer_norm = common_spec.LayerNormSpec()

            delattr(self.self_attention, "layer_norm")
            delattr(self.ffn, "layer_norm")


class FeedForwardSpec(model_spec.LayerSpec):
    def __init__(self, glu=False, rms_norm=False):
        self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        self.linear_0 = common_spec.LinearSpec()
        self.linear_1 = common_spec.LinearSpec()
        if glu:
            self.linear_0_noact = common_spec.LinearSpec()


class PositionEncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.encodings = model_spec.OPTIONAL


class TransformerConfig(model_spec.SequenceToSequenceModelConfig):
    """Configuration for Transformer models."""

    def __init__(self, layer_norm_epsilon: Optional[float] = None, **kwargs):
        """Initializes the configuration for Transformer models.

        Args:
          layer_norm_epsilon: The layer norm epsilon value.
          **kwargs: Additional configuration.
        """
        super().__init__(layer_norm_epsilon=layer_norm_epsilon, **kwargs)


class TransformerSpec(model_spec.SequenceToSequenceModelSpec):
    """Describes a Transformer model.

    The specification is invariant to hidden dimensions but requires to
    explicitly set the number of layers and attention heads.
    """

    def __init__(
        self, encoder: TransformerEncoderSpec, decoder: TransformerDecoderSpec
    ):
        """Initializes a Transformer model specification.

        Args:
          encoder: The encoder specification.
          decoder: The decoder specification.
        """
        if not isinstance(encoder, TransformerEncoderSpec):
            raise TypeError("encoder argument must be a TransformerEncoderSpec")
        if not isinstance(decoder, TransformerDecoderSpec):
            raise TypeError("decoder argument must be a TransformerDecoderSpec")

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def from_config(
        cls,
        num_layers: Union[int, Tuple[int, int]],
        num_heads: int,
        with_relative_position: bool = False,
        pre_norm: bool = True,
        no_final_norm: bool = False,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        alignment_layer: int = -1,
        alignment_heads: int = 1,
        num_source_embeddings: int = 1,
        embeddings_merge: common_spec.EmbeddingsMerge = common_spec.EmbeddingsMerge.CONCAT,
        layernorm_embedding: bool = False,
        relative_attention_bias: bool = False,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        multi_query_attention: bool = False,
    ):
        """Creates a Transformer model specification.

        Args:
          num_layers: Number of encoder and decoder layers, or a 2-tuple if the
            number is different.
          num_heads: Number of attention heads.
          with_relative_position: Use relative position representations in the self-attention
            layers as described in https://arxiv.org/abs/1803.02155.
          pre_norm: Enable the pre-norm Transformer architecture.
          no_final_norm: Disable the final layer norm in the pre-norm architecture.
          activation: Activation to apply in the feed-forward network.
          alignment_layer: Layer index selected for alignment.
          alignment_heads: Number of attention heads selected for alignment.
          num_source_embeddings: Number of source embeddings.
          embeddings_merge: When :obj:`num_source_embeddings` > 1, specify how the
            embeddings are merged.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          relative_attention_bias: Use relative attention bias in the self-attention
            layers as described in the T5 paper https://arxiv.org/abs/1910.10683.
          ffn_glu: Use gated linear units in the FFN layer as described in
            https://arxiv.org/abs/2002.05202.
          rms_norm: Use the root mean square layer normalization.
          multi_query_attention: Use multi-query attention.
        """
        if isinstance(num_layers, (list, tuple)):
            num_encoder_layers, num_decoder_layers = num_layers
        else:
            num_encoder_layers, num_decoder_layers = num_layers, num_layers

        encoder = TransformerEncoderSpec(
            num_encoder_layers,
            num_heads,
            pre_norm=pre_norm,
            no_final_norm=no_final_norm,
            activation=activation,
            num_source_embeddings=num_source_embeddings,
            embeddings_merge=embeddings_merge,
            layernorm_embedding=layernorm_embedding,
            relative_position=with_relative_position,
            relative_attention_bias=relative_attention_bias,
            ffn_glu=ffn_glu,
            rms_norm=rms_norm,
            multi_query_attention=multi_query_attention,
        )

        decoder = TransformerDecoderSpec(
            num_decoder_layers,
            num_heads,
            pre_norm=pre_norm,
            no_final_norm=no_final_norm,
            activation=activation,
            layernorm_embedding=layernorm_embedding,
            relative_position=with_relative_position,
            relative_attention_bias=relative_attention_bias,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            ffn_glu=ffn_glu,
            rms_norm=rms_norm,
            multi_query_attention=multi_query_attention,
        )

        return cls(encoder, decoder)

    @property
    def name(self):
        return "TransformerSpec"

    @property
    def revision(self):
        return 7

    def get_default_config(self):
        return TransformerConfig()

    def get_source_vocabulary_size(self):
        return [spec.weight.shape[0] for spec in self.encoder.embeddings]

    def get_target_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]


class TransformerDecoderModelConfig(model_spec.LanguageModelConfig):
    """Configuration for Transformer decoder models."""

    def __init__(self, layer_norm_epsilon: Optional[float] = None, **kwargs):
        """Initializes the configuration for Transformer decoder models.

        Args:
          layer_norm_epsilon: The layer norm epsilon value.
          **kwargs: Additional configuration.
        """
        super().__init__(layer_norm_epsilon=layer_norm_epsilon, **kwargs)


class TransformerDecoderModelSpec(model_spec.LanguageModelSpec):
    """Describes a Transformer decoder model (e.g. GPT-2)."""

    def __init__(self, decoder: TransformerDecoderSpec):
        """Initializes a Transformer decoder model specification.

        Args:
          decoder: The decoder specification.
        """
        if not isinstance(decoder, TransformerDecoderSpec):
            raise TypeError("decoder argument must be a TransformerDecoderSpec")

        super().__init__()
        self.decoder = decoder

    @classmethod
    def from_config(
        cls,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        layernorm_embedding: bool = False,
        no_final_norm: bool = False,
        project_in_out: bool = False,
        with_relative_position: bool = False,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        alibi: bool = False,
        alibi_use_positive_positions: bool = False,
        scale_alibi: bool = False,
        rotary_dim: Optional[int] = None,
        rotary_interleave: bool = True,
        rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = None,
        rotary_scaling_factor: float = 1,
        rotary_base: float = 10000,
        parallel_residual: bool = False,
        shared_layer_norm: bool = False,
        multi_query_attention: bool = False,
        num_heads_kv: Optional[int] = None,
        sliding_window: Optional[int] = None,
    ):
        """Creates a Transformer decoder model specification.

        Args:
          num_layers: Number of decoder layers.
          num_heads: Number of attention heads.
          pre_norm: Enable the pre-norm Transformer architecture.
          activation: Activation to apply in the feed-forward network.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          no_final_norm: Do not apply layer normalization after the last decoder block.
          project_in_out: Add a linear layer after the embedding layer and another one
            before the final output projection.
          with_relative_position: Enable relative position representations modules.
          ffn_glu: Use gated linear units in the FFN layers as described in
            https://arxiv.org/abs/2002.05202.
          rms_norm: Use the root mean square layer normalization.
          alibi: Use attention with linear biases.
          alibi_use_positive_positions: Use positive positions in the ALiBi definition.
          scale_alibi: Apply the dot product scale factor to ALiBi.
          rotary_dim: Apply rotary embeddings to these first N dimensions. If 0, rotary
            embeddings are applied to all dimensions.
          rotary_interleave: Interleave the head dimensions when rotary embeddings are applied.
            Otherwise the head dimensions are sliced in half.
          rotary_scaling_type: Type of RoPE scaling.
          rotary_scaling_factor: Factor used in the RoPE scaling.
          rotary_base: The base period of the rotary embeddings.
          parallel_residual: Use parallel residual connections in each layer block, as used
            by the GPT-J and GPT-NeoX models.
          shared_layer_norm: When using parallel residual, share the input and post
            attention layer norms.
          multi_query_attention: Use multi-query attention (alias for num_heads_kv=1).
          num_heads_kv: Number of attention heads for the key and value.
          sliding_window: max sequence length to retain KV cache
        """
        decoder = TransformerDecoderSpec(
            num_layers,
            num_heads,
            pre_norm=pre_norm,
            activation=activation,
            layernorm_embedding=layernorm_embedding,
            with_encoder_attention=False,
            no_final_norm=no_final_norm,
            project_in_out=project_in_out,
            relative_position=with_relative_position,
            ffn_glu=ffn_glu,
            rms_norm=rms_norm,
            alibi=alibi,
            alibi_use_positive_positions=alibi_use_positive_positions,
            scale_alibi=scale_alibi,
            rotary_dim=rotary_dim,
            rotary_interleave=rotary_interleave,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=rotary_base,
            parallel_residual=parallel_residual,
            shared_layer_norm=shared_layer_norm,
            multi_query_attention=multi_query_attention,
            num_heads_kv=num_heads_kv,
            sliding_window=sliding_window,
        )

        return cls(decoder)

    @property
    def name(self):
        return "TransformerDecoderSpec"

    @property
    def revision(self):
        return 8

    def get_default_config(self):
        return TransformerDecoderModelConfig()

    def get_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]


class TransformerEncoderModelConfig(model_spec.LanguageModelConfig):
    """Configuration for Transformer encoder models."""

    def __init__(self, layer_norm_epsilon: Optional[float] = None, **kwargs):
        """Initializes the configuration for Transformer encoder models.

        Args:
          layer_norm_epsilon: The layer norm epsilon value.
          **kwargs: Additional configuration.
        """
        super().__init__(layer_norm_epsilon=layer_norm_epsilon, **kwargs)


class TransformerEncoderModelSpec(model_spec.LanguageModelSpec):
    """Describes a Transformer encoder model (e.g. BERT)."""

    def __init__(
        self,
        encoder: TransformerEncoderSpec,
        pooling_layer: bool = False,
        pooling_activation: common_spec.Activation = common_spec.Activation.Tanh,
    ):
        """Initializes a Transformer encoder model specification.

        Args:
          encoder: The encoder specification.
          pooling_layer: Add the pooling layer.
          pooling_activation: The activation to apply after the pooling layer.
        """
        if not isinstance(encoder, TransformerEncoderSpec):
            raise TypeError("encoder argument must be a TransformerEncoderSpec")

        super().__init__()
        self.encoder = encoder

        if pooling_layer:
            self.pooler_dense = common_spec.LinearSpec()
            self.pooler_activation = np.dtype("int8").type(pooling_activation)

    @property
    def name(self):
        return "TransformerEncoderSpec"

    @property
    def revision(self):
        return 1

    def get_default_config(self):
        return TransformerEncoderModelConfig()

    def get_vocabulary_size(self):
        return self.encoder.embeddings[0].weight.shape[0]
