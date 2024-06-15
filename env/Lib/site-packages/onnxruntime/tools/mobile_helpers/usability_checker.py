# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import os
import pathlib
import tempfile
from collections import deque
from enum import IntEnum
from typing import Optional

import onnx

from ..onnx_model_utils import (
    ModelProtoWithShapeInfo,
    get_producer_consumer_maps,
    is_fixed_size_tensor,
    iterate_graph_per_graph_func,
    iterate_graph_per_node_func,
    optimize_model,
)


class _SupportedOpsChecker:
    """
    Class to process the md file with list of supported ops and caveats for an execution provider.
    e.g. /tools/ci_build/github/android/nnapi_supported_ops.md
         /tools/ci_build/github/apple/coreml_supported_ops.md
    """

    def __init__(self, filename):
        self._filename = filename
        self._ops = {}  # op to caveats
        self._ops_seen = set()

        with open(filename) as f:
            for line in f.readlines():
                # we're looking for a markdown table with 2 columns. first is op name. second is caveats
                # op name is domain:op
                if line.startswith("|"):
                    pieces = line.strip().split("|")
                    if len(pieces) == 4:  # pre-first '|'. op, caveat, post-last '|'
                        domain_op = pieces[1]
                        caveat = pieces[2]
                        caveat = caveat.replace("<br/>", " ")  # remove some HTML tags
                        # skip lines that don't have the ':' which separates the domain and op
                        # e.g. the table header will fail this check
                        if ":" in domain_op:
                            self._ops[domain_op] = caveat

    def is_op_supported(self, node):
        domain = node.domain if node.domain else "ai.onnx"
        domain_op = domain + ":" + node.op_type

        is_supported = domain_op in self._ops
        if is_supported:
            self._ops_seen.add(domain_op)

        return is_supported

    def get_caveats(self):
        caveats = []
        for op in sorted(self._ops_seen):
            caveat = self._ops[op]
            if caveat:
                caveats.append(f"{op}:{caveat}")

        return caveats


class PartitioningInfo:
    class TryWithEP(IntEnum):
        NO = (0,)
        MAYBE = (1,)
        YES = 2

    def __init__(self):
        self.num_nodes = -1  # main graph only
        self.num_supported_nodes = -1
        self.num_partitions = -1
        self.num_nodes_in_subgraphs = -1  # nodes not covered as we don't currently handle subgraphs in nnapi/coreml
        self.supported_ops_checker = None
        self.supported_groups = []
        self.unsupported_ops = set()
        self.nodes_unsupported_due_to_op = -1
        self.nodes_unsupported_due_to_dynamic_input = -1

    def suitability(self):
        # for now add up all the nodes. if there are subgraphs, the percentage of covered nodes will be reduced by all
        # nodes in the subgraphs.
        num_nodes = self.num_nodes + self.num_nodes_in_subgraphs

        # semi-arbitrary choices that err on the side of MAYBE.
        # having 1 partition is always preferred, but if that is small it may not be useful.
        # having 2 partitions may be okay if they cover most nodes
        # more than 2 partitions and the device copy cost is almost guaranteed to outweight the benefit of using the NPU
        # NOTE: This assumes the EP is not CPU based and there is device copy overhead to consider
        pct_supported = self.num_supported_nodes / num_nodes * 100
        if self.num_partitions == 1:
            if pct_supported > 75:
                return PartitioningInfo.TryWithEP.YES
            elif pct_supported > 50:
                return PartitioningInfo.TryWithEP.MAYBE
            else:
                return PartitioningInfo.TryWithEP.NO

        if self.num_partitions == 2:
            if pct_supported > 75:
                return PartitioningInfo.TryWithEP.MAYBE
            else:
                return PartitioningInfo.TryWithEP.NO

        return PartitioningInfo.TryWithEP.NO

    def dump_analysis(self, logger: logging.Logger, ep_name: str):
        """
        Analyze the partitioning information and log the analysis
        :param logger: Logger to use
        :param ep_name: Execution provider name to use in the log messages
        """

        num_nodes = self.num_nodes + self.num_nodes_in_subgraphs
        logger.info(
            f"{self.num_partitions} partitions with a total of {self.num_supported_nodes}/{num_nodes} "
            f"nodes can be handled by the {ep_name} EP."
        )
        if self.num_nodes_in_subgraphs:
            logger.info(f"{self.num_nodes_in_subgraphs} nodes are in subgraphs, which are currently not handled.")

        if self.supported_groups:
            logger.info(f'Partition sizes: [{", ".join([str(len(partition)) for partition in self.supported_groups])}]')
            logger.info(f"Unsupported nodes due to operator={self.nodes_unsupported_due_to_op}")
            if self.nodes_unsupported_due_to_dynamic_input:
                logger.info(
                    "Unsupported nodes due to input having a dynamic shape=%d",
                    self.nodes_unsupported_due_to_dynamic_input,
                )

        if logger.getEffectiveLevel() <= logging.DEBUG:
            # Enable this manually if you need to look at specific partitions.
            # for group in supported_groups:
            #     logger.debug(f'Nodes in group: {",".join([f"{node.name}:{node.op_type}" for node in group])}')
            if self.unsupported_ops:
                logger.info(f'Unsupported ops: {",".join(sorted(self.unsupported_ops))}')

            caveats = self.supported_ops_checker.get_caveats()
            if caveats:
                indent = " " * 5
                logger.debug(
                    "Caveats that have not been checked and may result in a node not being supported:  "
                    f'{"".join([os.linesep + indent + caveat for caveat in caveats])}'
                )

        pct_nodes_using_ep = self.num_supported_nodes / num_nodes * 100
        if self.num_partitions == 0:
            logger.info(f"{ep_name} cannot run any nodes in this model.")
        elif self.num_partitions == 1:
            if pct_nodes_using_ep > 75:
                logger.info(
                    f"{ep_name} should work well for this model as there is one partition "
                    f"covering {pct_nodes_using_ep:.1f}% of the nodes in the model."
                )
            elif pct_nodes_using_ep > 50:
                logger.info(
                    f"{ep_name} may work well for this model, however only {pct_nodes_using_ep:.1f}% of nodes "
                    "will use it. Performance testing is required to validate."
                )
            else:
                logger.info(
                    f"{ep_name} will probably not work will for this model as only {pct_nodes_using_ep:.2f}% "
                    "of nodes will use it."
                )

        elif self.num_partitions == 2 and pct_nodes_using_ep > 75:
            logger.info(
                f"{ep_name} can be considered for this model as there are two partitions "
                f"covering {pct_nodes_using_ep:.1f}% of the nodes. "
                "Performance testing is required to validate."
            )
        else:
            logger.info(
                f"{ep_name} is not recommended with this model as there are {self.num_partitions} partitions "
                f"covering {pct_nodes_using_ep:.1f}% of the nodes in the model. "
                "This will most likely result in worse performance than just using the CPU EP."
            )


def check_partitioning(
    graph: onnx.GraphProto,
    supported_ops_checker: _SupportedOpsChecker,
    require_fixed_input_sizes: bool = False,
    value_info: Optional[dict] = None,
):
    """
    Estimate the partitions the graph will be split into for nodes that is_node_supported_fn returns true for.

    The check on whether a node is supported is purely based on the operator type. Additional limitations
    (e.g. NNAPI EP only supports 2D Conv) are not checked, so partitions may not be 100% accurate. The limitations
    for operators in the partitions are printed so the user can manually check.
    :param graph: Graph to process
    :param supported_ops_checker: Checker with info on supported ops.
    :param require_fixed_input_sizes: If True, require that the inputs to a potentially supported node are
                                       fixed size tensors for it to be considered as supported.
                                       If True, onnx.shape_inference.infer_shapes should have been run on the model
                                       to populate the shape information.
    :param value_info: Map of value name to ValueInfoProto. Required if require_fixed_input_sizes is True to lookup
                       the shape of a value.
    :return PartitioningInfo instance with details
    """

    if require_fixed_input_sizes and not value_info:
        raise ValueError("value_info must be provided if require_fixed_input_sizes is True.")

    node_to_producers, node_to_consumers = get_producer_consumer_maps(graph)

    # initializers have fixed sizes.
    # TODO: when adding subgraph support we also need to match against initializers in ancestor graphs as they are
    # be accessible from the outer scope (unless shadowed locally)
    initializers = [i.name for i in graph.initializer]

    def _is_fixed_shape_value(value):
        if value in value_info:
            return is_fixed_size_tensor(value_info[value])
        if value in initializers:
            return True

        # if something has an unknown shape (e.g. something downstream of a Reshape with dynamic input for the shape)
        # it won't have an entry in value_info
        return False

    #
    # Replicate logic from /onnxruntime/core/providers/partitioning_utils.cc:CreateSupportedPartitionNodeGroups
    # to roughly estimate number of partitions for nodes that is_node_supported_fn returns true for.
    #
    # We keep the structure and variable names as close as possible to the C++ implementation to simplify keeping them
    # in sync if future updates are needed.
    #

    # we don't currently support a callback for additional group closure checks in the python implementation
    on_group_closed_fn = None

    supported_groups = []
    # number of inputs from unprocessed nodes (in-degree) per node
    in_degree = {}
    # nodes that are ready to process
    nodes_to_process = deque()  # deque of Node instances
    # nodes that will be processed when considering the next partition node group
    nodes_to_process_with_next_group = deque()

    # initialize in-degrees and find root nodes
    for node in graph.node:
        node_input_edge_count = len(node_to_producers[node]) if node in node_to_producers else 0
        in_degree[node] = node_input_edge_count
        if node_input_edge_count == 0:
            # node is only dependent on graph input or initializers
            nodes_to_process.append(node)

    # currently we don't support checking subgraphs in the partitioning as they're not handled by NNAPI/CoreML.
    # check how many nodes are in that blind spot so we can adjust the recommendation accordingly.
    # note: need to pass count in an array so that it's by reference
    def _count_subgraph_nodes(cur_graph: onnx.GraphProto, original_graph: onnx.GraphProto, count: [int]):
        if cur_graph != original_graph:
            count[0] += len(cur_graph.node)

    nodes_in_subgraphs = [0]  # array with single value
    iterate_graph_per_graph_func(graph, _count_subgraph_nodes, original_graph=graph, count=nodes_in_subgraphs)

    supported_group = []
    # the partition node group's border is the aggregate of its nodes' output nodes
    supported_group_border = set()
    num_supported_nodes = 0
    num_unsupported_nodes_due_to_op = 0
    num_unsupported_nodes_due_to_dynamic_input = 0
    unsupported_ops = set()

    def close_group():
        if supported_group:
            keep_partition = not on_group_closed_fn or on_group_closed_fn(supported_group)

            if keep_partition:
                supported_groups.append(supported_group.copy())

            supported_group.clear()
            supported_group_border.clear()

    while nodes_to_process or nodes_to_process_with_next_group:
        if not nodes_to_process:
            close_group()
            nodes_to_process = nodes_to_process_with_next_group
            nodes_to_process_with_next_group = deque()
            continue

        node = nodes_to_process.popleft()

        is_op_supported = supported_ops_checker.is_op_supported(node)
        is_input_shape_supported = not require_fixed_input_sizes or all(_is_fixed_shape_value(i) for i in node.input)
        is_node_supported = is_op_supported and is_input_shape_supported

        if not is_node_supported:
            if node in supported_group_border:
                # an unsupported node on the border will be processed after the current partition node group
                # so skip any additional processing/counting here
                nodes_to_process_with_next_group.append(node)
                continue

            if not is_op_supported:
                unsupported_ops.add(f'{node.domain if node.domain else "ai.onnx"}:{node.op_type}')
                num_unsupported_nodes_due_to_op += 1
            else:
                num_unsupported_nodes_due_to_dynamic_input += 1

        if is_node_supported:
            num_supported_nodes += 1

            # add node to the partition node group
            supported_group.append(node)

            # remove node from the border and add its outputs to the border
            if node in supported_group_border:
                supported_group_border.remove(node)

            # for each consumer node add to supported_group_border
            if node in node_to_consumers:
                for consumer in node_to_consumers[node]:
                    supported_group_border.add(consumer)

        # adjust in-degrees of the node outputs and add any new nodes to process
        if node in node_to_consumers:
            for consumer in node_to_consumers[node]:
                consumer_node_in_degree = in_degree[consumer]
                consumer_node_in_degree -= 1
                if consumer_node_in_degree == 0:
                    nodes_to_process.append(consumer)

                in_degree[consumer] = consumer_node_in_degree

    close_group()

    # find any subgraphs and check supported for nodes in the subgraphs. this won't change the partitioning as we skip
    # Scan/Loop/If nodes, but will provide additional info on operators that are not supported if we changed that.
    iterate_graph_per_node_func(graph, supported_ops_checker.is_op_supported)

    num_nodes = len(graph.node)
    num_partitions = len(supported_groups)

    info = PartitioningInfo()
    info.num_nodes = num_nodes
    info.num_supported_nodes = num_supported_nodes
    info.num_partitions = num_partitions
    info.num_nodes_in_subgraphs = nodes_in_subgraphs[0]
    info.supported_ops_checker = supported_ops_checker
    info.supported_groups = supported_groups
    info.unsupported_ops = unsupported_ops
    info.nodes_unsupported_due_to_op = num_unsupported_nodes_due_to_op
    info.nodes_unsupported_due_to_dynamic_input = num_unsupported_nodes_due_to_dynamic_input

    return info


def _check_ep_partitioning(model, supported_ops_config, value_info: Optional[dict] = None):
    supported_ops = _SupportedOpsChecker(supported_ops_config)
    partition_info = check_partitioning(model.graph, supported_ops, value_info is not None, value_info)
    return partition_info


def check_nnapi_partitions(model, value_info: Optional[dict] = None):
    # if we're running in the ORT python package the file should be local. otherwise assume we're running from the
    # ORT repo
    script_dir = pathlib.Path(__file__).parent
    local_config = script_dir / "nnapi_supported_ops.md"
    if local_config.exists():
        config_path = local_config
    else:
        ort_root = script_dir.parents[3]
        config_path = ort_root / "tools" / "ci_build" / "github" / "android" / "nnapi_supported_ops.md"

    return _check_ep_partitioning(model, config_path, value_info)


def check_coreml_partitions(model, value_info: Optional[dict] = None):
    # if we're running in the ORT python package the file should be local. otherwise assume we're running from the
    # ORT repo
    script_dir = pathlib.Path(__file__).parent
    local_config = script_dir / "coreml_supported_ops.md"
    if local_config.exists():
        config_path = local_config
    else:
        ort_root = script_dir.parents[3]
        config_path = ort_root / "tools" / "ci_build" / "github" / "apple" / "coreml_supported_ops.md"

    return _check_ep_partitioning(model, config_path, value_info)


def check_shapes(graph: onnx.GraphProto, logger: Optional[logging.Logger] = None):
    """
    Check the shapes of graph inputs, values and graph outputs to determine if they have static or dynamic sizes.
    NNAPI and CoreML do not support dynamically sized values.
    :param graph: Graph to check. If shape inferencing has been run the checks on values will be meaningful.
    :param logger: Optional logger for diagnostic information.
    :return: Tuple of List of inputs with dynamic shapes, Number of dynamic values found
    """

    # it's OK if the input is dynamically sized and we do a Resize early to a fixed size.
    # it's not good if lots of ops have dynamic inputs

    num_fixed_values = 0
    num_dynamic_values = 0

    dynamic_inputs = []
    for i in graph.input:
        if not is_fixed_size_tensor(i):
            dynamic_inputs.append(i)
            # split/join to remove repeated whitespace and newlines from str(i)
            if logger:
                logger.info(f"Input is not a fixed size tensor: {' '.join(str(i).split())}")
            num_dynamic_values += 1
        else:
            num_fixed_values += 1

    dynamic_outputs = []
    for o in graph.output:
        if not is_fixed_size_tensor(o):
            dynamic_outputs.append(o)
            if logger:
                logger.info(f"Output is not a fixed size tensor: {' '.join(str(o).split())}")
            num_dynamic_values += 1
        else:
            num_fixed_values += 1

    # check we have value info.
    # special case some test graphs with a single node which only have graph input and output values, and
    # a model where all inputs are dynamic (results in no value_info)
    if not graph.value_info and not (len(graph.node) == 1 or len(dynamic_inputs) == len(graph.input)):
        logger.warning(
            "Unable to check shapes within model. "
            "ONNX shape inferencing should be run on the model prior to checking."
        )

    for vi in graph.value_info:
        if is_fixed_size_tensor(vi):
            num_fixed_values += 1
        else:
            num_dynamic_values += 1

    if logger:
        logger.info(
            f"Num values with fixed shape={num_fixed_values}. Num values with dynamic shape={num_dynamic_values}"
        )

        if dynamic_inputs:
            if dynamic_outputs:
                logger.info(
                    "Model has dynamic inputs and outputs. Consider re-exporting model with fixed sizes "
                    "if NNAPI or CoreML can be used with this model."
                )
            else:
                logger.info(
                    """Model has dynamically sized inputs but fixed sized outputs.
                       If the sizes become fixed early in the model (e.g. pre-processing of a dynamic input size
                       results in a fixed input size for the majority of the model) performance with NNAPI and CoreML,
                       if applicable, should not be significantly impacted."""
                )

    return dynamic_inputs, num_dynamic_values


def checker(model_path: pathlib.Path, logger: logging.Logger):
    model_with_shape_info_wrapper = ModelProtoWithShapeInfo(model_path)
    model_with_shape_info = model_with_shape_info_wrapper.model_with_shape_info

    # create lookup map for efficiency
    value_to_shape = {}
    for v in model_with_shape_info.graph.input:
        value_to_shape[v.name] = v
    for v in model_with_shape_info.graph.output:
        value_to_shape[v.name] = v
    for v in model_with_shape_info.graph.value_info:
        value_to_shape[v.name] = v

    dynamic_inputs, num_dynamic_values = check_shapes(model_with_shape_info.graph)

    def check_ep(ep_name, checker_func):
        logger.info(f"Checking {ep_name}")

        # check with shape info first so supported nodes takes into account values with dynamic shapes
        partition_info = checker_func(model_with_shape_info, value_to_shape)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            partition_info.dump_analysis(logger, ep_name)

        suitability = partition_info.suitability()
        logger.info(f"Model should perform well with {ep_name} as is: {suitability.name}")

        if suitability != PartitioningInfo.TryWithEP.YES and dynamic_inputs:
            logger.info("Checking if model will perform better if the dynamic shapes are fixed...")
            partition_info_with_fixed_shapes = checker_func(model_with_shape_info)
            if logger.getEffectiveLevel() <= logging.DEBUG:
                # analyze and log detailed info
                logger.info("Partition information if the model was updated to make the shapes fixed:")
                partition_info_with_fixed_shapes.dump_analysis(logger, ep_name)

            fixed_shape_suitability = partition_info_with_fixed_shapes.suitability()
            logger.info(
                f"Model should perform well with {ep_name} if modified to have fixed input shapes: "
                f"{fixed_shape_suitability.name}"
            )
            if fixed_shape_suitability != PartitioningInfo.TryWithEP.NO:
                logger.info("Shapes can be altered using python -m onnxruntime.tools.make_dynamic_shape_fixed")

            if fixed_shape_suitability.value > suitability.value:
                suitability = fixed_shape_suitability

        return suitability

    nnapi_suitability = check_ep("NNAPI", check_nnapi_partitions)
    coreml_suitability = check_ep("CoreML", check_coreml_partitions)

    if (
        nnapi_suitability != PartitioningInfo.TryWithEP.YES or coreml_suitability != PartitioningInfo.TryWithEP.YES
    ) and logger.getEffectiveLevel() > logging.DEBUG:
        logger.info("Re-run with log level of DEBUG for more details on the NNAPI/CoreML issues.")

    logger.info("---------------")
    return nnapi_suitability != PartitioningInfo.TryWithEP.NO or coreml_suitability != PartitioningInfo.TryWithEP.NO


def analyze_model(model_path: pathlib.Path, skip_optimize: bool = False, logger: Optional[logging.Logger] = None):
    """
    Analyze the provided model to determine if it's likely to work well with the NNAPI or CoreML Execution Providers
    :param model_path: Model to analyze.
    :param skip_optimize: Skip optimizing to BASIC level before checking. When exporting to ORT format we will do this
                          optimization..
    :param logger: Logger for output
    :return: True if either the NNAPI or CoreML Execution Providers may work well with this model.
    """
    if not logger:
        logger = logging.getLogger("usability_checker")
        logger.setLevel(logging.INFO)

    logger.info(f"Checking {model_path} for usability with ORT Mobile.")

    with tempfile.TemporaryDirectory() as tmp:
        if not skip_optimize:
            tmp_path = pathlib.Path(tmp) / model_path.name
            optimize_model(model_path, tmp_path, use_external_initializers=True)
            model_path = tmp_path

        try_eps = checker(model_path.resolve(strict=True), logger)

    return try_eps


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="""Analyze an ONNX model for usage with the ORT mobile"""
    )

    parser.add_argument(
        "--log_level", choices=["debug", "info", "warning", "error"], default="info", help="Logging level"
    )
    parser.add_argument(
        "--skip_optimize",
        action="store_true",
        help="Don't optimize the model to BASIC level prior to analyzing. "
        "Optimization will occur when exporting the model to ORT format, so in general "
        "should not be skipped unless you have a specific reason to do so.",
    )
    parser.add_argument("model_path", type=pathlib.Path, help="Provide path to ONNX model")

    return parser.parse_args()


def run_analyze_model():
    args = parse_args()
    logger = logging.getLogger("default")

    if args.log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif args.log_level == "info":
        logger.setLevel(logging.INFO)
    elif args.log_level == "warning":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)

    model_path = args.model_path.resolve()
    analyze_model(model_path, args.skip_optimize, logger)


if __name__ == "__main__":
    run_analyze_model()
