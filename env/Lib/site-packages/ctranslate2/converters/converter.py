import abc
import argparse
import os
import shutil

from typing import Optional

from ctranslate2.specs.model_spec import ACCEPTED_MODEL_TYPES, ModelSpec


class Converter(abc.ABC):
    """Base class for model converters."""

    @staticmethod
    def declare_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Adds common conversion options to the command line parser.

        Arguments:
          parser: Command line argument parser.
        """
        parser.add_argument(
            "--output_dir", required=True, help="Output model directory."
        )
        parser.add_argument(
            "--vocab_mapping", default=None, help="Vocabulary mapping file (optional)."
        )
        parser.add_argument(
            "--quantization",
            default=None,
            choices=ACCEPTED_MODEL_TYPES,
            help="Weight quantization type.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force conversion even if the output directory already exists.",
        )
        return parser

    def convert_from_args(self, args: argparse.Namespace) -> str:
        """Helper function to call :meth:`ctranslate2.converters.Converter.convert`
        with the parsed command line options.

        Arguments:
          args: Namespace containing parsed arguments.

        Returns:
          Path to the output directory.
        """
        return self.convert(
            args.output_dir,
            vmap=args.vocab_mapping,
            quantization=args.quantization,
            force=args.force,
        )

    def convert(
        self,
        output_dir: str,
        vmap: Optional[str] = None,
        quantization: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Converts the model to the CTranslate2 format.

        Arguments:
          output_dir: Output directory where the CTranslate2 model is saved.
          vmap: Optional path to a vocabulary mapping file that will be included
            in the converted model directory.
          quantization: Weight quantization scheme (possible values are: int8, int8_float32,
            int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
          force: Override the output directory if it already exists.

        Returns:
          Path to the output directory.

        Raises:
          RuntimeError: If the output directory already exists and :obj:`force`
            is not set.
          NotImplementedError: If the converter cannot convert this model to the
            CTranslate2 format.
        """
        if os.path.exists(output_dir) and not force:
            raise RuntimeError(
                "output directory %s already exists, use --force to override"
                % output_dir
            )

        model_spec = self._load()
        if model_spec is None:
            raise NotImplementedError(
                "This model is not supported by CTranslate2 or this converter"
            )
        if vmap is not None:
            model_spec.register_vocabulary_mapping(vmap)

        model_spec.validate()
        model_spec.optimize(quantization=quantization)

        # Create model directory.
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        model_spec.save(output_dir)
        return output_dir

    @abc.abstractmethod
    def _load(self):
        raise NotImplementedError()
