#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal segmentation
=====================

Recurrence and self-similarity
------------------------------
.. autosummary::
    :toctree: generated/

    cross_similarity
    recurrence_matrix
    recurrence_to_lag
    lag_to_recurrence
    timelag_filter
    path_enhance

Temporal clustering
-------------------
.. autosummary::
    :toctree: generated/

    agglomerative
    subsegment
"""

from decorator import decorator

import numpy as np
import scipy
import scipy.signal
import scipy.ndimage

import sklearn
import sklearn.cluster
import sklearn.feature_extraction
import sklearn.neighbors

from ._cache import cache
from . import util
from .filters import diagonal_filter
from .util.exceptions import ParameterError
from .util.decorators import deprecate_positional_args

__all__ = [
    "cross_similarity",
    "recurrence_matrix",
    "recurrence_to_lag",
    "lag_to_recurrence",
    "timelag_filter",
    "agglomerative",
    "subsegment",
    "path_enhance",
]


@deprecate_positional_args
@cache(level=30)
def cross_similarity(
    data,
    data_ref,
    *,
    k=None,
    metric="euclidean",
    sparse=False,
    mode="connectivity",
    bandwidth=None,
):
    """Compute cross-similarity from one data sequence to a reference sequence.

    The output is a matrix ``xsim``, where ``xsim[i, j]`` is non-zero
    if ``data_ref[..., i]`` is a k-nearest neighbor of ``data[..., j]``.

    Parameters
    ----------
    data : np.ndarray [shape=(..., d, n)]
        A feature matrix for the comparison sequence.
        If the data has more than two dimensions (e.g., for multi-channel inputs),
        the leading dimensions are flattened prior to comparison.
        For example, a stereo input with shape `(2, d, n)` is
        automatically reshaped to `(2 * d, n)`.

    data_ref : np.ndarray [shape=(..., d, n_ref)]
        A feature matrix for the reference sequence
        If the data has more than two dimensions (e.g., for multi-channel inputs),
        the leading dimensions are flattened prior to comparison.
        For example, a stereo input with shape `(2, d, n_ref)` is
        automatically reshaped to `(2 * d, n_ref)`.

    k : int > 0 [scalar] or None
        the number of nearest-neighbors for each sample

        Default: ``k = 2 * ceil(sqrt(n_ref))``,
        or ``k = 2`` if ``n_ref <= 3``

    metric : str
        Distance metric to use for nearest-neighbor calculation.

        See `sklearn.neighbors.NearestNeighbors` for details.

    sparse : bool [scalar]
        if False, returns a dense type (ndarray)
        if True, returns a sparse type (scipy.sparse.csc_matrix)

    mode : str, {'connectivity', 'distance', 'affinity'}
        If 'connectivity', a binary connectivity matrix is produced.

        If 'distance', then a non-zero entry contains the distance between
        points.

        If 'affinity', then non-zero entries are mapped to
        ``exp( - distance(i, j) / bandwidth)`` where ``bandwidth`` is
        as specified below.

    bandwidth : None or float > 0
        If using ``mode='affinity'``, this can be used to set the
        bandwidth on the affinity kernel.

        If no value is provided, it is set automatically to the median
        distance to the k'th nearest neighbor of each ``data[:, i]``.

    Returns
    -------
    xsim : np.ndarray or scipy.sparse.csc_matrix, [shape=(n_ref, n)]
        Cross-similarity matrix

    See Also
    --------
    recurrence_matrix
    recurrence_to_lag
    librosa.feature.stack_memory
    sklearn.neighbors.NearestNeighbors
    scipy.spatial.distance.cdist

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Find nearest neighbors in CQT space between two sequences

    >>> hop_length = 1024
    >>> y_ref, sr = librosa.load(librosa.ex('pistachio'))
    >>> y_comp, sr = librosa.load(librosa.ex('pistachio'), offset=10)
    >>> chroma_ref = librosa.feature.chroma_cqt(y=y_ref, sr=sr, hop_length=hop_length)
    >>> chroma_comp = librosa.feature.chroma_cqt(y=y_comp, sr=sr, hop_length=hop_length)
    >>> # Use time-delay embedding to get a cleaner recurrence matrix
    >>> x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
    >>> x_comp = librosa.feature.stack_memory(chroma_comp, n_steps=10, delay=3)
    >>> xsim = librosa.segment.cross_similarity(x_comp, x_ref)

    Or fix the number of nearest neighbors to 5

    >>> xsim = librosa.segment.cross_similarity(x_comp, x_ref, k=5)

    Use cosine similarity instead of Euclidean distance

    >>> xsim = librosa.segment.cross_similarity(x_comp, x_ref, metric='cosine')

    Use an affinity matrix instead of binary connectivity

    >>> xsim_aff = librosa.segment.cross_similarity(x_comp, x_ref, metric='cosine', mode='affinity')

    Plot the feature and recurrence matrices

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    >>> imgsim = librosa.display.specshow(xsim, x_axis='s', y_axis='s',
    ...                          hop_length=hop_length, ax=ax[0])
    >>> ax[0].set(title='Binary recurrence (symmetric)')
    >>> imgaff = librosa.display.specshow(xsim_aff, x_axis='s', y_axis='s',
    ...                          cmap='magma_r', hop_length=hop_length, ax=ax[1])
    >>> ax[1].set(title='Affinity recurrence')
    >>> ax[1].label_outer()
    >>> fig.colorbar(imgsim, ax=ax[0], orientation='horizontal', ticks=[0, 1])
    >>> fig.colorbar(imgaff, ax=ax[1], orientation='horizontal')
    """
    data_ref = np.atleast_2d(data_ref)
    data = np.atleast_2d(data)

    if not np.allclose(data_ref.shape[:-1], data.shape[:-1]):
        raise ParameterError(
            "data_ref.shape={} and data.shape={} do not match on leading dimension(s)".format(
                data_ref.shape, data.shape
            )
        )

    # swap data axes so the feature axis is last
    data_ref = np.swapaxes(data_ref, -1, 0)
    n_ref = data_ref.shape[0]
    # Use F-ordering for reshape to preserve leading axis
    data_ref = data_ref.reshape((n_ref, -1), order="F")

    data = np.swapaxes(data, -1, 0)
    n = data.shape[0]
    data = data.reshape((n, -1), order="F")

    if mode not in ["connectivity", "distance", "affinity"]:
        raise ParameterError(
            (
                "Invalid mode='{}'. Must be one of "
                "['connectivity', 'distance', "
                "'affinity']"
            ).format(mode)
        )
    if k is None:
        k = min(n_ref, 2 * np.ceil(np.sqrt(n_ref)))

    k = int(k)

    if bandwidth is not None:
        if bandwidth <= 0:
            raise ParameterError(
                "Invalid bandwidth={}. " "Must be strictly positive.".format(bandwidth)
            )

    # Build the neighbor search object
    # `auto` mode does not work with some choices of metric.  Rather than special-case
    # those here, we instead use a fall-back to brute force if auto fails.
    try:
        knn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(n_ref, k), metric=metric, algorithm="auto"
        )
    except ValueError:
        knn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(n_ref, k), metric=metric, algorithm="brute"
        )

    knn.fit(data_ref)

    # Get the knn graph
    if mode == "affinity":
        # sklearn's nearest neighbor doesn't support affinity,
        # so we use distance here and then do the conversion post-hoc
        kng_mode = "distance"
    else:
        kng_mode = mode

    xsim = knn.kneighbors_graph(X=data, mode=kng_mode).tolil()

    # Retain only the top-k links per point
    for i in range(n):
        # Get the links from point i
        links = xsim[i].nonzero()[1]

        # Order them ascending
        idx = links[np.argsort(xsim[i, links].toarray())][0]

        # Everything past the kth closest gets squashed
        xsim[i, idx[k:]] = 0

    # Convert a compressed sparse row (CSR) format
    xsim = xsim.tocsr()
    xsim.eliminate_zeros()

    if mode == "connectivity":
        xsim = xsim.astype(np.bool)
    elif mode == "affinity":
        if bandwidth is None:
            bandwidth = np.nanmedian(xsim.max(axis=1).data)
        xsim.data[:] = np.exp(xsim.data / (-1 * bandwidth))

    # Transpose to n_ref by n
    xsim = xsim.T

    if not sparse:
        xsim = xsim.toarray()

    return xsim


@deprecate_positional_args
@cache(level=30)
def recurrence_matrix(
    data,
    *,
    k=None,
    width=1,
    metric="euclidean",
    sym=False,
    sparse=False,
    mode="connectivity",
    bandwidth=None,
    self=False,
    axis=-1,
):
    """Compute a recurrence matrix from a data matrix.

    ``rec[i, j]`` is non-zero if ``data[..., i]`` is a k-nearest neighbor
    of ``data[..., j]`` and ``|i - j| >= width``

    The specific value of ``rec[i, j]`` can have several forms, governed
    by the ``mode`` parameter below:

        - Connectivity: ``rec[i, j] = 1 or 0`` indicates that frames ``i`` and ``j`` are repetitions

        - Affinity: ``rec[i, j] > 0`` measures how similar frames ``i`` and ``j`` are.  This is also
          known as a (sparse) self-similarity matrix.

        - Distance: ``rec[i, j] > 0`` measures how distant frames ``i`` and ``j`` are.  This is also
          known as a (sparse) self-distance matrix.

    The general term *recurrence matrix* can refer to any of the three forms above.

    Parameters
    ----------
    data : np.ndarray [shape=(..., d, n)]
        A feature matrix.
        If the data has more than two dimensions (e.g., for multi-channel inputs),
        the leading dimensions are flattened prior to comparison.
        For example, a stereo input with shape `(2, d, n)` is
        automatically reshaped to `(2 * d, n)`.

    k : int > 0 [scalar] or None
        the number of nearest-neighbors for each sample

        Default: ``k = 2 * ceil(sqrt(t - 2 * width + 1))``,
        or ``k = 2`` if ``t <= 2 * width + 1``

    width : int >= 1 [scalar]
        only link neighbors ``(data[..., i], data[..., j])``
        if ``|i - j| >= width``

        ``width`` cannot exceed the length of the data.

    metric : str
        Distance metric to use for nearest-neighbor calculation.

        See `sklearn.neighbors.NearestNeighbors` for details.

    sym : bool [scalar]
        set ``sym=True`` to only link mutual nearest-neighbors

    sparse : bool [scalar]
        if False, returns a dense type (ndarray)
        if True, returns a sparse type (scipy.sparse.csc_matrix)

    mode : str, {'connectivity', 'distance', 'affinity'}
        If 'connectivity', a binary connectivity matrix is produced.

        If 'distance', then a non-zero entry contains the distance between
        points.

        If 'affinity', then non-zero entries are mapped to
        ``exp( - distance(i, j) / bandwidth)`` where ``bandwidth`` is
        as specified below.

    bandwidth : None or float > 0
        If using ``mode='affinity'``, this can be used to set the
        bandwidth on the affinity kernel.

        If no value is provided, it is set automatically to the median
        distance between furthest nearest neighbors.

    self : bool
        If ``True``, then the main diagonal is populated with self-links:
        0 if ``mode='distance'``, and 1 otherwise.

        If ``False``, the main diagonal is left empty.

    axis : int
        The axis along which to compute recurrence.
        By default, the last index (-1) is taken.

    Returns
    -------
    rec : np.ndarray or scipy.sparse.csc_matrix, [shape=(t, t)]
        Recurrence matrix

    See Also
    --------
    sklearn.neighbors.NearestNeighbors
    scipy.spatial.distance.cdist
    librosa.feature.stack_memory
    recurrence_to_lag

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Find nearest neighbors in CQT space

    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 1024
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> # Use time-delay embedding to get a cleaner recurrence matrix
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> R = librosa.segment.recurrence_matrix(chroma_stack)

    Or fix the number of nearest neighbors to 5

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, k=5)

    Suppress neighbors within +- 7 frames

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, width=7)

    Use cosine similarity instead of Euclidean distance

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, metric='cosine')

    Require mutual nearest neighbors

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, sym=True)

    Use an affinity matrix instead of binary connectivity

    >>> R_aff = librosa.segment.recurrence_matrix(chroma_stack, metric='cosine',
    ...                                           mode='affinity')

    Plot the feature and recurrence matrices

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    >>> imgsim = librosa.display.specshow(R, x_axis='s', y_axis='s',
    ...                          hop_length=hop_length, ax=ax[0])
    >>> ax[0].set(title='Binary recurrence (symmetric)')
    >>> imgaff = librosa.display.specshow(R_aff, x_axis='s', y_axis='s',
    ...                          hop_length=hop_length, cmap='magma_r', ax=ax[1])
    >>> ax[1].set(title='Affinity recurrence')
    >>> ax[1].label_outer()
    >>> fig.colorbar(imgsim, ax=ax[0], orientation='horizontal', ticks=[0, 1])
    >>> fig.colorbar(imgaff, ax=ax[1], orientation='horizontal')
    """

    data = np.atleast_2d(data)

    # Swap observations to the first dimension and flatten the rest
    data = np.swapaxes(data, axis, 0)
    t = data.shape[0]
    # Use F-ordering here to preserve leading axis layout
    data = data.reshape((t, -1), order="F")

    if width < 1 or width > t:
        raise ParameterError(
            "width={} must be at least 1 and at most data.shape[{}]={}".format(
                width, axis, t
            )
        )

    if mode not in ["connectivity", "distance", "affinity"]:
        raise ParameterError(
            (
                "Invalid mode='{}'. Must be one of "
                "['connectivity', 'distance', "
                "'affinity']"
            ).format(mode)
        )
    if k is None:
        if t > 2 * width + 1:
            k = 2 * np.ceil(np.sqrt(t - 2 * width + 1))
        else:
            k = 2

    if bandwidth is not None:
        if bandwidth <= 0:
            raise ParameterError(
                "Invalid bandwidth={}. " "Must be strictly positive.".format(bandwidth)
            )

    k = int(k)

    # Build the neighbor search object
    try:
        knn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(t - 1, k + 2 * width), metric=metric, algorithm="auto"
        )
    except ValueError:
        knn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(t - 1, k + 2 * width), metric=metric, algorithm="brute"
        )

    knn.fit(data)

    # Get the knn graph
    if mode == "affinity":
        kng_mode = "distance"
    else:
        kng_mode = mode

    rec = knn.kneighbors_graph(mode=kng_mode).tolil()

    # Remove connections within width
    for diag in range(-width + 1, width):
        rec.setdiag(0, diag)

    # Retain only the top-k links per point
    for i in range(t):
        # Get the links from point i
        links = rec[i].nonzero()[1]

        # Order them ascending
        idx = links[np.argsort(rec[i, links].toarray())][0]

        # Everything past the kth closest gets squashed
        rec[i, idx[k:]] = 0

    if self:
        if mode == "connectivity":
            rec.setdiag(1)
        elif mode == "affinity":
            # we need to keep the self-loop in here, but not mess up the
            # bandwidth estimation
            #
            # using negative distances here preserves the structure without changing
            # the statistics of the data
            rec.setdiag(-1)

    # symmetrize
    if sym:
        # Note: this operation produces a CSR (compressed sparse row) matrix!
        # This is why we have to do it after filling the diagonal in self-mode
        rec = rec.minimum(rec.T)

    rec = rec.tocsr()
    rec.eliminate_zeros()

    if mode == "connectivity":
        rec = rec.astype(np.bool)
    elif mode == "affinity":
        if bandwidth is None:
            bandwidth = np.nanmedian(rec.max(axis=1).data)
        # Set all the negatives back to 0
        # Negatives are temporarily inserted above to preserve the sparsity structure
        # of the matrix without corrupting the bandwidth calculations
        rec.data[rec.data < 0] = 0.0
        rec.data[:] = np.exp(rec.data / (-1 * bandwidth))

    # Transpose to be column-major
    rec = rec.T

    if not sparse:
        rec = rec.toarray()

    return rec


@deprecate_positional_args
def recurrence_to_lag(rec, *, pad=True, axis=-1):
    """Convert a recurrence matrix into a lag matrix.

        ``lag[i, j] == rec[i+j, j]``

    This transformation turns diagonal structures in the recurrence matrix
    into horizontal structures in the lag matrix.
    These horizontal structures can be used to infer changes in the repetition
    structure of a piece, e.g., the beginning of a new section as done in [#]_.

    .. [#] Serra, J., Müller, M., Grosche, P., & Arcos, J. L. (2014).
           Unsupervised music structure annotation by time series structure
           features and segment similarity.
           IEEE Transactions on Multimedia, 16(5), 1229-1240.

    Parameters
    ----------
    rec : np.ndarray, or scipy.sparse.spmatrix [shape=(n, n)]
        A (binary) recurrence matrix, as returned by `recurrence_matrix`

    pad : bool
        If False, ``lag`` matrix is square, which is equivalent to
        assuming that the signal repeats itself indefinitely.

        If True, ``lag`` is padded with ``n`` zeros, which eliminates
        the assumption of repetition.

    axis : int
        The axis to keep as the ``time`` axis.
        The alternate axis will be converted to lag coordinates.

    Returns
    -------
    lag : np.ndarray
        The recurrence matrix in (lag, time) (if ``axis=1``)
        or (time, lag) (if ``axis=0``) coordinates

    Raises
    ------
    ParameterError : if ``rec`` is non-square

    See Also
    --------
    recurrence_matrix
    lag_to_recurrence
    util.shear

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 1024
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> recurrence = librosa.segment.recurrence_matrix(chroma_stack)
    >>> lag_pad = librosa.segment.recurrence_to_lag(recurrence, pad=True)
    >>> lag_nopad = librosa.segment.recurrence_to_lag(recurrence, pad=False)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(lag_pad, x_axis='time', y_axis='lag',
    ...                          hop_length=hop_length, ax=ax[0])
    >>> ax[0].set(title='Lag (zero-padded)')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(lag_nopad, x_axis='time', y_axis='lag',
    ...                          hop_length=hop_length, ax=ax[1])
    >>> ax[1].set(title='Lag (no padding)')
    """

    axis = np.abs(axis)

    if rec.ndim != 2 or rec.shape[0] != rec.shape[1]:
        raise ParameterError(
            "non-square recurrence matrix shape: " "{}".format(rec.shape)
        )

    sparse = scipy.sparse.issparse(rec)

    if sparse:
        fmt = rec.format

    t = rec.shape[axis]

    if pad:
        if sparse:
            padding = np.asarray([[1, 0]], dtype=rec.dtype).swapaxes(axis, 0)
            if axis == 0:
                rec_fmt = "csr"
            else:
                rec_fmt = "csc"
            rec = scipy.sparse.kron(padding, rec, format=rec_fmt)
        else:
            padding = [(0, 0), (0, 0)]
            padding[(1 - axis)] = (0, t)
            rec = np.pad(rec, padding, mode="constant")

    lag = util.shear(rec, factor=-1, axis=axis)

    if sparse:
        lag = lag.asformat(fmt)

    return lag


@deprecate_positional_args
def lag_to_recurrence(lag, *, axis=-1):
    """Convert a lag matrix into a recurrence matrix.

    Parameters
    ----------
    lag : np.ndarray or scipy.sparse.spmatrix
        A lag matrix, as produced by ``recurrence_to_lag``
    axis : int
        The axis corresponding to the time dimension.
        The alternate axis will be interpreted in lag coordinates.

    Returns
    -------
    rec : np.ndarray or scipy.sparse.spmatrix [shape=(n, n)]
        A recurrence matrix in (time, time) coordinates
        For sparse matrices, format will match that of ``lag``.

    Raises
    ------
    ParameterError : if ``lag`` does not have the correct shape

    See Also
    --------
    recurrence_to_lag

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 1024
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> recurrence = librosa.segment.recurrence_matrix(chroma_stack)
    >>> lag_pad = librosa.segment.recurrence_to_lag(recurrence, pad=True)
    >>> lag_nopad = librosa.segment.recurrence_to_lag(recurrence, pad=False)
    >>> rec_pad = librosa.segment.lag_to_recurrence(lag_pad)
    >>> rec_nopad = librosa.segment.lag_to_recurrence(lag_nopad)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
    >>> librosa.display.specshow(lag_pad, x_axis='s', y_axis='lag',
    ...                          hop_length=hop_length, ax=ax[0, 0])
    >>> ax[0, 0].set(title='Lag (zero-padded)')
    >>> ax[0, 0].label_outer()
    >>> librosa.display.specshow(lag_nopad, x_axis='s', y_axis='time',
    ...                          hop_length=hop_length, ax=ax[0, 1])
    >>> ax[0, 1].set(title='Lag (no padding)')
    >>> ax[0, 1].label_outer()
    >>> librosa.display.specshow(rec_pad, x_axis='s', y_axis='time',
    ...                          hop_length=hop_length, ax=ax[1, 0])
    >>> ax[1, 0].set(title='Recurrence (with padding)')
    >>> librosa.display.specshow(rec_nopad, x_axis='s', y_axis='time',
    ...                          hop_length=hop_length, ax=ax[1, 1])
    >>> ax[1, 1].set(title='Recurrence (without padding)')
    >>> ax[1, 1].label_outer()
    """

    if axis not in [0, 1, -1]:
        raise ParameterError("Invalid target axis: {}".format(axis))

    axis = np.abs(axis)

    if lag.ndim != 2 or (
        lag.shape[0] != lag.shape[1] and lag.shape[1 - axis] != 2 * lag.shape[axis]
    ):
        raise ParameterError("Invalid lag matrix shape: {}".format(lag.shape))

    # Since lag must be 2-dimensional, abs(axis) = axis
    t = lag.shape[axis]

    rec = util.shear(lag, factor=+1, axis=axis)

    sub_slice = [slice(None)] * rec.ndim
    sub_slice[1 - axis] = slice(t)
    return rec[tuple(sub_slice)]


def timelag_filter(function, pad=True, index=0):
    """Filtering in the time-lag domain.

    This is primarily useful for adapting image filters to operate on
    `recurrence_to_lag` output.

    Using `timelag_filter` is equivalent to the following sequence of
    operations:

    >>> data_tl = librosa.segment.recurrence_to_lag(data)
    >>> data_filtered_tl = function(data_tl)
    >>> data_filtered = librosa.segment.lag_to_recurrence(data_filtered_tl)

    Parameters
    ----------
    function : callable
        The filtering function to wrap, e.g., `scipy.ndimage.median_filter`
    pad : bool
        Whether to zero-pad the structure feature matrix
    index : int >= 0
        If ``function`` accepts input data as a positional argument, it should be
        indexed by ``index``

    Returns
    -------
    wrapped_function : callable
        A new filter function which applies in time-lag space rather than
        time-time space.

    Examples
    --------
    Apply a 31-bin median filter to the diagonal of a recurrence matrix.
    With default, parameters, this corresponds to a time window of about
    0.72 seconds.

    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=30)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=3, delay=3)
    >>> rec = librosa.segment.recurrence_matrix(chroma_stack)
    >>> from scipy.ndimage import median_filter
    >>> diagonal_median = librosa.segment.timelag_filter(median_filter)
    >>> rec_filtered = diagonal_median(rec, size=(1, 31), mode='mirror')

    Or with affinity weights

    >>> rec_aff = librosa.segment.recurrence_matrix(chroma_stack, mode='affinity')
    >>> rec_aff_fil = diagonal_median(rec_aff, size=(1, 31), mode='mirror')

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(rec, y_axis='s', x_axis='s', ax=ax[0, 0])
    >>> ax[0, 0].set(title='Raw recurrence matrix')
    >>> ax[0, 0].label_outer()
    >>> librosa.display.specshow(rec_filtered, y_axis='s', x_axis='s', ax=ax[0, 1])
    >>> ax[0, 1].set(title='Filtered recurrence matrix')
    >>> ax[0, 1].label_outer()
    >>> librosa.display.specshow(rec_aff, x_axis='s', y_axis='s',
    ...                          cmap='magma_r', ax=ax[1, 0])
    >>> ax[1, 0].set(title='Raw affinity matrix')
    >>> librosa.display.specshow(rec_aff_fil, x_axis='s', y_axis='s',
    ...                          cmap='magma_r', ax=ax[1, 1])
    >>> ax[1, 1].set(title='Filtered affinity matrix')
    >>> ax[1, 1].label_outer()
    """

    def __my_filter(wrapped_f, *args, **kwargs):
        """Decorator to wrap the filter"""
        # Map the input data into time-lag space
        args = list(args)

        args[index] = recurrence_to_lag(args[index], pad=pad)

        # Apply the filtering function
        result = wrapped_f(*args, **kwargs)

        # Map back into time-time and return
        return lag_to_recurrence(result)

    return decorator(__my_filter, function)


@deprecate_positional_args
@cache(level=30)
def subsegment(data, frames, *, n_segments=4, axis=-1):
    """Sub-divide a segmentation by feature clustering.

    Given a set of frame boundaries (``frames``), and a data matrix (``data``),
    each successive interval defined by ``frames`` is partitioned into
    ``n_segments`` by constrained agglomerative clustering.

    .. note::
        If an interval spans fewer than ``n_segments`` frames, then each
        frame becomes a sub-segment.

    Parameters
    ----------
    data : np.ndarray
        Data matrix to use in clustering
    frames : np.ndarray [shape=(n_boundaries,)], dtype=int, non-negative]
        Array of beat or segment boundaries, as provided by
        `librosa.beat.beat_track`,
        `librosa.onset.onset_detect`,
        or `agglomerative`.
    n_segments : int > 0
        Maximum number of frames to sub-divide each interval.
    axis : int
        Axis along which to apply the segmentation.
        By default, the last index (-1) is taken.

    Returns
    -------
    boundaries : np.ndarray [shape=(n_subboundaries,)]
        List of sub-divided segment boundaries

    See Also
    --------
    agglomerative : Temporal segmentation
    librosa.onset.onset_detect : Onset detection
    librosa.beat.beat_track : Beat tracking

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Load audio, detect beat frames, and subdivide in twos by CQT

    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    >>> beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    >>> cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512))
    >>> subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)
    >>> subseg_t = librosa.frames_to_time(subseg, sr=sr, hop_length=512)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(librosa.amplitude_to_db(cqt,
    ...                                                  ref=np.max),
    ...                          y_axis='cqt_hz', x_axis='time', ax=ax)
    >>> lims = ax.get_ylim()
    >>> ax.vlines(beat_times, lims[0], lims[1], color='lime', alpha=0.9,
    ...            linewidth=2, label='Beats')
    >>> ax.vlines(subseg_t, lims[0], lims[1], color='linen', linestyle='--',
    ...            linewidth=1.5, alpha=0.5, label='Sub-beats')
    >>> ax.legend()
    >>> ax.set(title='CQT + Beat and sub-beat markers')
    """

    frames = util.fix_frames(frames, x_min=0, x_max=data.shape[axis], pad=True)

    if n_segments < 1:
        raise ParameterError("n_segments must be a positive integer")

    boundaries = []
    idx_slices = [slice(None)] * data.ndim

    for seg_start, seg_end in zip(frames[:-1], frames[1:]):
        idx_slices[axis] = slice(seg_start, seg_end)
        boundaries.extend(
            seg_start
            + agglomerative(
                data[tuple(idx_slices)], min(seg_end - seg_start, n_segments), axis=axis
            )
        )

    return np.array(boundaries)


@deprecate_positional_args
def agglomerative(data, k, *, clusterer=None, axis=-1):
    """Bottom-up temporal segmentation.

    Use a temporally-constrained agglomerative clustering routine to partition
    ``data`` into ``k`` contiguous segments.

    Parameters
    ----------
    data : np.ndarray
        data to cluster
    k : int > 0 [scalar]
        number of segments to produce
    clusterer : sklearn.cluster.AgglomerativeClustering, optional
        An optional AgglomerativeClustering object.
        If `None`, a constrained Ward object is instantiated.
    axis : int
        axis along which to cluster.
        By default, the last axis (-1) is chosen.

    Returns
    -------
    boundaries : np.ndarray [shape=(k,)]
        left-boundaries (frame numbers) of detected segments. This
        will always include `0` as the first left-boundary.

    See Also
    --------
    sklearn.cluster.AgglomerativeClustering

    Examples
    --------
    Cluster by chroma similarity, break into 20 segments

    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> bounds = librosa.segment.agglomerative(chroma, 20)
    >>> bound_times = librosa.frames_to_time(bounds, sr=sr)
    >>> bound_times
    array([ 0.   ,  0.65 ,  1.091,  1.927,  2.438,  2.902,  3.924,
            4.783,  5.294,  5.712,  6.13 ,  7.314,  8.522,  8.916,
            9.66 , 10.844, 11.238, 12.028, 12.492, 14.095])

    Plot the segmentation over the chromagram

    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.transforms as mpt
    >>> fig, ax = plt.subplots()
    >>> trans = mpt.blended_transform_factory(
    ...             ax.transData, ax.transAxes)
    >>> librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    >>> ax.vlines(bound_times, 0, 1, color='linen', linestyle='--',
    ...           linewidth=2, alpha=0.9, label='Segment boundaries',
    ...           transform=trans)
    >>> ax.legend()
    >>> ax.set(title='Power spectrogram')
    """

    # Make sure we have at least two dimensions
    data = np.atleast_2d(data)

    # Swap data index to position 0
    data = np.swapaxes(data, axis, 0)

    # Flatten the features
    n = data.shape[0]
    data = data.reshape((n, -1), order="F")

    if clusterer is None:
        # Connect the temporal connectivity graph
        grid = sklearn.feature_extraction.image.grid_to_graph(n_x=n, n_y=1, n_z=1)

        # Instantiate the clustering object
        clusterer = sklearn.cluster.AgglomerativeClustering(
            n_clusters=k, connectivity=grid, memory=cache.memory
        )

    # Fit the model
    clusterer.fit(data)

    # Find the change points from the labels
    boundaries = [0]
    boundaries.extend(list(1 + np.nonzero(np.diff(clusterer.labels_))[0].astype(int)))
    return np.asarray(boundaries)


@deprecate_positional_args
def path_enhance(
    R,
    n,
    *,
    window="hann",
    max_ratio=2.0,
    min_ratio=None,
    n_filters=7,
    zero_mean=False,
    clip=True,
    **kwargs,
):
    """Multi-angle path enhancement for self- and cross-similarity matrices.

    This function convolves multiple diagonal smoothing filters with a self-similarity (or
    recurrence) matrix R, and aggregates the result by an element-wise maximum.

    Technically, the output is a matrix R_smooth such that::

        R_smooth[i, j] = max_theta (R * filter_theta)[i, j]

    where `*` denotes 2-dimensional convolution, and ``filter_theta`` is a smoothing filter at
    orientation theta.

    This is intended to provide coherent temporal smoothing of self-similarity matrices
    when there are changes in tempo.

    Smoothing filters are generated at evenly spaced orientations between min_ratio and
    max_ratio.

    This function is inspired by the multi-angle path enhancement of [#]_, but differs by
    modeling tempo differences in the space of similarity matrices rather than re-sampling
    the underlying features prior to generating the self-similarity matrix.

    .. [#] Müller, Meinard and Frank Kurth.
            "Enhancing similarity matrices for music audio analysis."
            2006 IEEE International Conference on Acoustics Speech and Signal Processing Proceedings.
            Vol. 5. IEEE, 2006.

    .. note:: if using recurrence_matrix to construct the input similarity matrix, be sure to include the main
              diagonal by setting ``self=True``.  Otherwise, the diagonal will be suppressed, and this is likely to
              produce discontinuities which will pollute the smoothing filter response.

    Parameters
    ----------
    R : np.ndarray
        The self- or cross-similarity matrix to be smoothed.
        Note: sparse inputs are not supported.

        If the recurrence matrix is multi-dimensional, e.g. `shape=(c, n, n)`,
        then enhancement is conducted independently for each leading channel.

    n : int > 0
        The length of the smoothing filter

    window : window specification
        The type of smoothing filter to use.  See `filters.get_window` for more information
        on window specification formats.

    max_ratio : float > 0
        The maximum tempo ratio to support

    min_ratio : float > 0
        The minimum tempo ratio to support.
        If not provided, it will default to ``1/max_ratio``

    n_filters : int >= 1
        The number of different smoothing filters to use, evenly spaced
        between ``min_ratio`` and ``max_ratio``.

        If ``min_ratio = 1/max_ratio`` (the default), using an odd number
        of filters will ensure that the main diagonal (ratio=1) is included.

    zero_mean : bool
        By default, the smoothing filters are non-negative and sum to one (i.e. are averaging
        filters).

        If ``zero_mean=True``, then the smoothing filters are made to sum to zero by subtracting
        a constant value from the non-diagonal coordinates of the filter.  This is primarily
        useful for suppressing blocks while enhancing diagonals.

    clip : bool
        If True, the smoothed similarity matrix will be thresholded at 0, and will not contain
        negative entries.

    **kwargs : additional keyword arguments
        Additional arguments to pass to `scipy.ndimage.convolve`

    Returns
    -------
    R_smooth : np.ndarray, shape=R.shape
        The smoothed self- or cross-similarity matrix

    See Also
    --------
    librosa.filters.diagonal_filter
    recurrence_matrix

    Examples
    --------
    Use a 51-frame diagonal smoothing filter to enhance paths in a recurrence matrix

    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 2048
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> rec = librosa.segment.recurrence_matrix(chroma_stack, mode='affinity', self=True)
    >>> rec_smooth = librosa.segment.path_enhance(rec, 51, window='hann', n_filters=7)

    Plot the recurrence matrix before and after smoothing

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(rec, x_axis='s', y_axis='s',
    ...                          hop_length=hop_length, ax=ax[0])
    >>> ax[0].set(title='Unfiltered recurrence')
    >>> imgpe = librosa.display.specshow(rec_smooth, x_axis='s', y_axis='s',
    ...                          hop_length=hop_length, ax=ax[1])
    >>> ax[1].set(title='Multi-angle enhanced recurrence')
    >>> ax[1].label_outer()
    >>> fig.colorbar(img, ax=ax[0], orientation='horizontal')
    >>> fig.colorbar(imgpe, ax=ax[1], orientation='horizontal')
    """

    if min_ratio is None:
        min_ratio = 1.0 / max_ratio
    elif min_ratio > max_ratio:
        raise ParameterError(
            "min_ratio={} cannot exceed max_ratio={}".format(min_ratio, max_ratio)
        )

    R_smooth = None
    for ratio in np.logspace(
        np.log2(min_ratio), np.log2(max_ratio), num=n_filters, base=2
    ):
        kernel = diagonal_filter(window, n, slope=ratio, zero_mean=zero_mean)

        # Expand leading dimensions to match R
        # This way, if R has shape, eg, [2, 3, n, n]
        # the expanded kernel will have shape [1, 1, m, m]

        # The following is valid for numpy >= 1.18
        # kernel = np.expand_dims(kernel, axis=list(np.arange(R.ndim - kernel.ndim)))

        # This is functionally equivalent, but works on numpy 1.17
        shape = [1] * R.ndim
        shape[-2:] = kernel.shape
        kernel = np.reshape(kernel, shape)

        if R_smooth is None:
            R_smooth = scipy.ndimage.convolve(R, kernel, **kwargs)
        else:
            # Compute the point-wise maximum in-place
            np.maximum(
                R_smooth, scipy.ndimage.convolve(R, kernel, **kwargs), out=R_smooth
            )

    if clip:
        # Clip the output in-place
        np.clip(R_smooth, 0, None, out=R_smooth)

    return R_smooth
