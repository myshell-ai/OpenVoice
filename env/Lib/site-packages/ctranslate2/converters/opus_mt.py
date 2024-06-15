import argparse
import os

import yaml

from ctranslate2.converters.marian import MarianConverter


class OpusMTConverter(MarianConverter):
    """Converts models trained with OPUS-MT."""

    def __init__(self, model_dir: str):
        """Initializes the OPUS-MT converter.

        Arguments:
          model_dir: Path the OPUS-MT model directory.
        """
        with open(
            os.path.join(model_dir, "decoder.yml"), encoding="utf-8"
        ) as decoder_file:
            decoder_config = yaml.safe_load(decoder_file)

        model_path = os.path.join(model_dir, decoder_config["models"][0])
        vocab_paths = [
            os.path.join(model_dir, path) for path in decoder_config["vocabs"]
        ]
        super().__init__(model_path, vocab_paths)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir", required=True, help="Path to the OPUS-MT model directory."
    )
    OpusMTConverter.declare_arguments(parser)
    args = parser.parse_args()
    converter = OpusMTConverter(args.model_dir)
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
