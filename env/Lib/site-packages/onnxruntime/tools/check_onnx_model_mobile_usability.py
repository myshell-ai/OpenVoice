# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import pathlib

# need this before the mobile helper imports for some reason
logging.basicConfig(format="%(levelname)s:  %(message)s")

from .mobile_helpers import check_model_can_use_ort_mobile_pkg, usability_checker  # noqa: E402


def check_usability():
    parser = argparse.ArgumentParser(
        description="""Analyze an ONNX model to determine how well it will work in mobile scenarios, and whether
        it is likely to be able to use the pre-built ONNX Runtime Mobile Android or iOS package.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config_path",
        help="Path to required operators and types configuration used to build the pre-built ORT mobile package.",
        required=False,
        type=pathlib.Path,
        default=check_model_can_use_ort_mobile_pkg.get_default_config_path(),
    )
    parser.add_argument(
        "--log_level", choices=["debug", "info", "warning", "error"], default="info", help="Logging level"
    )
    parser.add_argument("model_path", help="Path to ONNX model to check", type=pathlib.Path)

    args = parser.parse_args()
    logger = logging.getLogger("check_usability")

    if args.log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif args.log_level == "info":
        logger.setLevel(logging.INFO)
    elif args.log_level == "warning":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)

    try_eps = usability_checker.analyze_model(args.model_path, skip_optimize=False, logger=logger)
    check_model_can_use_ort_mobile_pkg.run_check(args.model_path, args.config_path, logger)

    logger.info(
        "Run `python -m onnxruntime.tools.convert_onnx_models_to_ort ...` to convert the ONNX model to ORT "
        "format. "
        "By default, the conversion tool will create an ORT format model with saved optimizations which can "
        "potentially be applied at runtime (with a .with_runtime_opt.ort file extension) for use with NNAPI "
        "or CoreML, and a fully optimized ORT format model (with a .ort file extension) for use with the CPU "
        "EP."
    )
    if try_eps:
        logger.info(
            "As NNAPI or CoreML may provide benefits with this model it is recommended to compare the "
            "performance of the <model>.with_runtime_opt.ort model using the NNAPI EP on Android, and the "
            "CoreML EP on iOS, against the performance of the <model>.ort model using the CPU EP."
        )
    else:
        logger.info("For optimal performance the <model>.ort model should be used with the CPU EP. ")


if __name__ == "__main__":
    check_usability()
