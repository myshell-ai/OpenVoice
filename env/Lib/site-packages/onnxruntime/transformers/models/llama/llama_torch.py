# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging
import os

import torch
from dist_settings import barrier, get_rank, get_size
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger("")


def setup_torch_model(args, location, use_auth_token, torch_dtype=torch.float32, device=None):
    world_size = get_size()
    logger.info(f"world_size: {world_size}")
    rank = get_rank()
    barrier()

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)

    for i in range(world_size):
        if i == rank % (world_size):
            l_config = AutoConfig.from_pretrained(location, use_auth_token=use_auth_token, cache_dir=args.cache_dir)
            l_config.use_cache = True
            l_config._attn_implementation = "eager"  # "eager" uses LlamaAttention for attention layer
            llama = AutoModelForCausalLM.from_pretrained(
                location,
                use_auth_token=use_auth_token,
                config=l_config,
                torch_dtype=torch_dtype,
                cache_dir=args.cache_dir,
            )
            if world_size > 1:
                llama.parallel_model()
            if device:
                llama.to(device)
            llama.eval()
            llama.requires_grad_(False)
        barrier()
    return l_config, llama
