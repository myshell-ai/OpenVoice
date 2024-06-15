#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities
=========

Array operations
----------------
.. autosummary::
    :toctree: generated/

    frame
    pad_center
    expand_to
    fix_length
    fix_frames
    index_to_slice
    softmask
    stack
    sync

    axis_sort
    normalize
    shear
    sparsify_rows

    buf_to_float
    tiny

Matching
--------
.. autosummary::
    :toctree: generated/

    match_intervals
    match_events

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    localmax
    localmin
    peak_pick
    nnls
    cyclic_gradient
    dtype_c2r
    dtype_r2c
    count_unique
    is_unique


Input validation
----------------
.. autosummary::
    :toctree: generated/

    valid_audio
    valid_int
    valid_intervals


File operations
---------------

.. autosummary::
    :toctree: generated/

    example
    example_info
    list_examples
    find_files

"""

from .utils import *  # pylint: disable=wildcard-import
from .files import *  # pylint: disable=wildcard-import
from .matching import *  # pylint: disable=wildcard-import
from .deprecation import *  # pylint: disable=wildcard-import
from ._nnls import *  # pylint: disable=wildcard-import
from . import decorators
from . import exceptions
