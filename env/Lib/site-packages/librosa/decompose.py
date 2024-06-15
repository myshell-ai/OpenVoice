#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spectrogram decomposition
=========================
.. autosummary::
    :toctree: generated/

    decompose
    hpss
    nn_filter
"""

import numpy as np

import scipy.sparse
from scipy.ndimage import median_filter

import sklearn.decomposition

from . import core
from ._cache import cache
from . import segment
from . import util
from .util.exceptions import ParameterError
from .util.decorators import deprecate_positional_args

__all__ = ["decompose", "hpss", "nn_filter"]


@deprecate_positional_args
def decompose(
    S, *, n_components=None, transformer=None, sort=False, fit=True, **kwargs
):
    """Decompose a feature matrix.

    Given a spectrogram ``S``, produce a decomposition into ``components``
    and ``activations`` such that ``S ~= components.dot(activations)``.

    By default, this is done with with non-negative matrix factorization (NMF),
    but any `sklearn.decomposition`-type object will work.

    Parameters
    ----------
    S : np.ndarray [shape=(..., n_features, n_samples), dtype=float]
        The input feature matrix (e.g., magnitude spectrogram)

        If the input has multiple channels (leading dimensions), they will be automatically
        flattened prior to decomposition.

        If the input is multi-channel, channels and features are automatically flattened into
        a single axis before the decomposition.
        For example, a stereo input `S` with shape `(2, n_features, n_samples)` is
        automatically reshaped to `(2 * n_features, n_samples)`.

    n_components : int > 0 [scalar] or None
        number of desired components

        if None, then ``n_features`` components are used

    transformer : None or object
        If None, use `sklearn.decomposition.NMF`

        Otherwise, any object with a similar interface to NMF should work.
        ``transformer`` must follow the scikit-learn convention, where
        input data is ``(n_samples, n_features)``.

        `transformer.fit_transform()` will be run on ``S.T`` (not ``S``),
        the return value of which is stored (transposed) as ``activations``

        The components will be retrieved as ``transformer.components_.T``::

            S ~= np.dot(activations, transformer.components_).T

        or equivalently::

            S ~= np.dot(transformer.components_.T, activations.T)

    sort : bool
        If ``True``, components are sorted by ascending peak frequency.

        .. note:: If used with ``transformer``, sorting is applied to copies
            of the decomposition parameters, and not to ``transformer``
            internal parameters.

        .. warning:: If the input array has more than two dimensions
            (e.g., if it's a multi-channel spectrogram), then axis sorting
            is not supported and a `ParameterError` exception is raised.

    fit : bool
        If `True`, components are estimated from the input ``S``.

        If `False`, components are assumed to be pre-computed and stored
        in ``transformer``, and are not changed.

    **kwargs : Additional keyword arguments to the default transformer
        `sklearn.decomposition.NMF`

    Returns
    -------
    components: np.ndarray [shape=(..., n_features, n_components)]
        matrix of components (basis elements).
    activations: np.ndarray [shape=(n_components, n_samples)]
        transformed matrix/activation matrix

    Raises
    ------
    ParameterError
        if ``fit`` is False and no ``transformer`` object is provided.

        if the input array is multi-channel and ``sort=True`` is specified.

    See Also
    --------
    sklearn.decomposition : SciKit-Learn matrix decomposition modules

    Examples
    --------
    Decompose a magnitude spectrogram into 16 components with NMF

    >>> y, sr = librosa.load(librosa.ex('pistachio'), duration=5)
    >>> S = np.abs(librosa.stft(y))
    >>> comps, acts = librosa.decompose.decompose(S, n_components=16)

    Sort components by ascending peak frequency

    >>> comps, acts = librosa.decompose.decompose(S, n_components=16,
    ...                                           sort=True)

    Or with sparse dictionary learning

    >>> import sklearn.decomposition
    >>> T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=16)
    >>> scomps, sacts = librosa.decompose.decompose(S, transformer=T, sort=True)

    >>> import matplotlib.pyplot as plt
    >>> layout = [list(".AAAA"), list("BCCCC"), list(".DDDD")]
    >>> fig, ax = plt.subplot_mosaic(layout, constrained_layout=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax['A'])
    >>> ax['A'].set(title='Input spectrogram')
    >>> ax['A'].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(comps,
    >>>                                                  ref=np.max),
    >>>                          y_axis='log', ax=ax['B'])
    >>> ax['B'].set(title='Components')
    >>> ax['B'].label_outer()
    >>> ax['B'].sharey(ax['A'])
    >>> librosa.display.specshow(acts, x_axis='time', ax=ax['C'], cmap='gray_r')
    >>> ax['C'].set(ylabel='Components', title='Activations')
    >>> ax['C'].sharex(ax['A'])
    >>> ax['C'].label_outer()
    >>> S_approx = comps.dot(acts)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S_approx,
    >>>                                                        ref=np.max),
    >>>                                y_axis='log', x_axis='time', ax=ax['D'])
    >>> ax['D'].set(title='Reconstructed spectrogram')
    >>> ax['D'].sharex(ax['A'])
    >>> ax['D'].sharey(ax['A'])
    >>> ax['D'].label_outer()
    >>> fig.colorbar(img, ax=list(ax.values()), format="%+2.f dB")
    """

    # Do a swapaxes and unroll
    orig_shape = list(S.shape)

    if S.ndim > 2 and sort:
        raise ParameterError(
            "Parameter sort=True is unsupported for input with more than two dimensions"
        )

    # Transpose S and unroll feature dimensions
    # Use order='F' here to preserve the temporal ordering
    S = S.T.reshape((S.shape[-1], -1), order="F")

    if n_components is None:
        n_components = S.shape[-1]

    if transformer is None:
        if fit is False:
            raise ParameterError("fit must be True if transformer is None")

        transformer = sklearn.decomposition.NMF(n_components=n_components, **kwargs)

    if fit:
        activations = transformer.fit_transform(S).T
    else:
        activations = transformer.transform(S).T

    components = transformer.components_
    component_shape = orig_shape[:-1] + [-1]
    # use order='F' here to preserve component ordering
    components = components.reshape(component_shape[::-1], order="F").T

    if sort:
        components, idx = util.axis_sort(components, index=True)
        activations = activations[idx]

    return components, activations


@cache(level=30)
@deprecate_positional_args
def hpss(S, *, kernel_size=31, power=2.0, mask=False, margin=1.0):
    """Median-filtering harmonic percussive source separation (HPSS).

    If ``margin = 1.0``, decomposes an input spectrogram ``S = H + P``
    where ``H`` contains the harmonic components,
    and ``P`` contains the percussive components.

    If ``margin > 1.0``, decomposes an input spectrogram ``S = H + P + R``
    where ``R`` contains residual components not included in ``H`` or ``P``.

    This implementation is based upon the algorithm described by [#]_ and [#]_.

    .. [#] Fitzgerald, Derry.
        "Harmonic/percussive separation using median filtering."
        13th International Conference on Digital Audio Effects (DAFX10),
        Graz, Austria, 2010.

    .. [#] Driedger, Müller, Disch.
        "Extending harmonic-percussive separation of audio."
        15th International Society for Music Information Retrieval Conference (ISMIR 2014),
        Taipei, Taiwan, 2014.

    Parameters
    ----------
    S : np.ndarray [shape=(..., d, n)]
        input spectrogram. May be real (magnitude) or complex.
        Multi-channel is supported.

    kernel_size : int or tuple (kernel_harmonic, kernel_percussive)
        kernel size(s) for the median filters.

        - If scalar, the same size is used for both harmonic and percussive.
        - If tuple, the first value specifies the width of the
          harmonic filter, and the second value specifies the width
          of the percussive filter.

    power : float > 0 [scalar]
        Exponent for the Wiener filter when constructing soft mask matrices.

    mask : bool
        Return the masking matrices instead of components.

        Masking matrices contain non-negative real values that
        can be used to measure the assignment of energy from ``S``
        into harmonic or percussive components.

        Components can be recovered by multiplying ``S * mask_H``
        or ``S * mask_P``.

    margin : float or tuple (margin_harmonic, margin_percussive)
        margin size(s) for the masks (as described in [2]_)

        - If scalar, the same size is used for both harmonic and percussive.
        - If tuple, the first value specifies the margin of the
          harmonic mask, and the second value specifies the margin
          of the percussive mask.

    Returns
    -------
    harmonic : np.ndarray [shape=(..., d, n)]
        harmonic component (or mask)
    percussive : np.ndarray [shape=(..., d, n)]
        percussive component (or mask)

    See Also
    --------
    librosa.util.softmask

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Separate into harmonic and percussive

    >>> y, sr = librosa.load(librosa.ex('choice'), duration=5)
    >>> D = librosa.stft(y)
    >>> H, P = librosa.decompose.hpss(D)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(D),
    ...                                                        ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Full power spectrogram')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(np.abs(H),
    ...                                                  ref=np.max(np.abs(D))),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Harmonic power spectrogram')
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(np.abs(P),
    ...                                                  ref=np.max(np.abs(D))),
    ...                          y_axis='log', x_axis='time', ax=ax[2])
    >>> ax[2].set(title='Percussive power spectrogram')
    >>> fig.colorbar(img, ax=ax, format='%+2.0f dB')

    Or with a narrower horizontal filter

    >>> H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))

    Just get harmonic/percussive masks, not the spectra

    >>> mask_H, mask_P = librosa.decompose.hpss(D, mask=True)
    >>> mask_H
    array([[1.853e-03, 1.701e-04, ..., 9.922e-01, 1.000e+00],
           [2.316e-03, 2.127e-04, ..., 9.989e-01, 1.000e+00],
           ...,
           [8.195e-05, 6.939e-05, ..., 3.105e-04, 4.231e-04],
           [3.159e-05, 4.156e-05, ..., 6.216e-04, 6.188e-04]],
          dtype=float32)
    >>> mask_P
    array([[9.981e-01, 9.998e-01, ..., 7.759e-03, 3.201e-05],
           [9.977e-01, 9.998e-01, ..., 1.122e-03, 4.451e-06],
           ...,
           [9.999e-01, 9.999e-01, ..., 9.997e-01, 9.996e-01],
           [1.000e+00, 1.000e+00, ..., 9.994e-01, 9.994e-01]],
          dtype=float32)

    Separate into harmonic/percussive/residual components by using a margin > 1.0

    >>> H, P = librosa.decompose.hpss(D, margin=3.0)
    >>> R = D - (H+P)
    >>> y_harm = librosa.istft(H)
    >>> y_perc = librosa.istft(P)
    >>> y_resi = librosa.istft(R)

    Get a more isolated percussive component by widening its margin

    >>> H, P = librosa.decompose.hpss(D, margin=(1.0,5.0))

    """

    if np.iscomplexobj(S):
        S, phase = core.magphase(S)
    else:
        phase = 1

    if np.isscalar(kernel_size):
        win_harm = kernel_size
        win_perc = kernel_size
    else:
        win_harm = kernel_size[0]
        win_perc = kernel_size[1]

    if np.isscalar(margin):
        margin_harm = margin
        margin_perc = margin
    else:
        margin_harm = margin[0]
        margin_perc = margin[1]

    # margin minimum is 1.0
    if margin_harm < 1 or margin_perc < 1:
        raise ParameterError(
            "Margins must be >= 1.0. " "A typical range is between 1 and 10."
        )

    # shape for kernels
    harm_shape = [1 for _ in S.shape]
    harm_shape[-1] = win_harm

    perc_shape = [1 for _ in S.shape]
    perc_shape[-2] = win_perc

    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = np.empty_like(S)
    harm[:] = median_filter(S, size=harm_shape, mode="reflect")

    perc = np.empty_like(S)
    perc[:] = median_filter(S, size=perc_shape, mode="reflect")

    split_zeros = margin_harm == 1 and margin_perc == 1

    mask_harm = util.softmask(
        harm, perc * margin_harm, power=power, split_zeros=split_zeros
    )

    mask_perc = util.softmask(
        perc, harm * margin_perc, power=power, split_zeros=split_zeros
    )

    if mask:
        return mask_harm, mask_perc

    return ((S * mask_harm) * phase, (S * mask_perc) * phase)


@cache(level=30)
@deprecate_positional_args
def nn_filter(S, *, rec=None, aggregate=None, axis=-1, **kwargs):
    """Filtering by nearest-neighbors.

    Each data point (e.g, spectrogram column) is replaced
    by aggregating its nearest neighbors in feature space.

    This can be useful for de-noising a spectrogram or feature matrix.

    The non-local means method [#]_ can be recovered by providing a
    weighted recurrence matrix as input and specifying ``aggregate=np.average``.

    Similarly, setting ``aggregate=np.median`` produces sparse de-noising
    as in REPET-SIM [#]_.

    .. [#] Buades, A., Coll, B., & Morel, J. M.
        (2005, June). A non-local algorithm for image denoising.
        In Computer Vision and Pattern Recognition, 2005.
        CVPR 2005. IEEE Computer Society Conference on (Vol. 2, pp. 60-65). IEEE.

    .. [#] Rafii, Z., & Pardo, B.
        (2012, October).  "Music/Voice Separation Using the Similarity Matrix."
        International Society for Music Information Retrieval Conference, 2012.

    Parameters
    ----------
    S : np.ndarray
        The input data (spectrogram) to filter. Multi-channel is supported.

    rec : (optional) scipy.sparse.spmatrix or np.ndarray
        Optionally, a pre-computed nearest-neighbor matrix
        as provided by `librosa.segment.recurrence_matrix`

    aggregate : function
        aggregation function (default: `np.mean`)

        If ``aggregate=np.average``, then a weighted average is
        computed according to the (per-row) weights in ``rec``.

        For all other aggregation functions, all neighbors
        are treated equally.

    axis : int
        The axis along which to filter (by default, columns)

    **kwargs
        Additional keyword arguments provided to
        `librosa.segment.recurrence_matrix` if ``rec`` is not provided

    Returns
    -------
    S_filtered : np.ndarray
        The filtered data, with shape equivalent to the input ``S``.

    Raises
    ------
    ParameterError
        if ``rec`` is provided and its shape is incompatible with ``S``.

    See Also
    --------
    decompose
    hpss
    librosa.segment.recurrence_matrix

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    De-noise a chromagram by non-local median filtering.
    By default this would use euclidean distance to select neighbors,
    but this can be overridden directly by setting the ``metric`` parameter.

    >>> y, sr = librosa.load(librosa.ex('brahms'),
    ...                      offset=30, duration=10)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> chroma_med = librosa.decompose.nn_filter(chroma,
    ...                                          aggregate=np.median,
    ...                                          metric='cosine')

    To use non-local means, provide an affinity matrix and ``aggregate=np.average``.

    >>> rec = librosa.segment.recurrence_matrix(chroma, mode='affinity',
    ...                                         metric='cosine', sparse=True)
    >>> chroma_nlm = librosa.decompose.nn_filter(chroma, rec=rec,
    ...                                          aggregate=np.average)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=5, sharex=True, sharey=True, figsize=(10, 10))
    >>> librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Unfiltered')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(chroma_med, y_axis='chroma', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Median-filtered')
    >>> ax[1].label_outer()
    >>> imgc = librosa.display.specshow(chroma_nlm, y_axis='chroma', x_axis='time', ax=ax[2])
    >>> ax[2].set(title='Non-local means')
    >>> ax[2].label_outer()
    >>> imgr1 = librosa.display.specshow(chroma - chroma_med,
    ...                          y_axis='chroma', x_axis='time', ax=ax[3])
    >>> ax[3].set(title='Original - median')
    >>> ax[3].label_outer()
    >>> imgr2 = librosa.display.specshow(chroma - chroma_nlm,
    ...                          y_axis='chroma', x_axis='time', ax=ax[4])
    >>> ax[4].label_outer()
    >>> ax[4].set(title='Original - NLM')
    >>> fig.colorbar(imgc, ax=ax[:3])
    >>> fig.colorbar(imgr1, ax=[ax[3]])
    >>> fig.colorbar(imgr2, ax=[ax[4]])
    """

    if aggregate is None:
        aggregate = np.mean

    if rec is None:
        kwargs = dict(kwargs)
        kwargs["sparse"] = True
        rec = segment.recurrence_matrix(S, axis=axis, **kwargs)
    elif not scipy.sparse.issparse(rec):
        rec = scipy.sparse.csc_matrix(rec)

    if rec.shape[0] != S.shape[axis] or rec.shape[0] != rec.shape[1]:
        raise ParameterError(
            "Invalid self-similarity matrix shape "
            "rec.shape={} for S.shape={}".format(rec.shape, S.shape)
        )

    return __nn_filter_helper(
        rec.data, rec.indices, rec.indptr, S.swapaxes(0, axis), aggregate
    ).swapaxes(0, axis)


def __nn_filter_helper(R_data, R_indices, R_ptr, S, aggregate):
    """Nearest-neighbor filter helper function.

    This is an internal function, not for use outside of the decompose module.

    It applies the nearest-neighbor filter to S, assuming that the first index
    corresponds to observations.

    Parameters
    ----------
    R_data, R_indices, R_ptr : np.ndarrays
        The ``data``, ``indices``, and ``indptr`` of a scipy.sparse matrix
    S : np.ndarray
        The observation data to filter
    aggregate : callable
        The aggregation operator

    Returns
    -------
    S_out : np.ndarray like S
        The filtered data array
    """
    s_out = np.empty_like(S)

    for i in range(len(R_ptr) - 1):

        # Get the non-zeros out of the recurrence matrix
        targets = R_indices[R_ptr[i] : R_ptr[i + 1]]

        if not len(targets):
            s_out[i] = S[i]
            continue

        neighbors = np.take(S, targets, axis=0)

        if aggregate is np.average:
            weights = R_data[R_ptr[i] : R_ptr[i + 1]]
            s_out[i] = aggregate(neighbors, axis=0, weights=weights)
        else:
            s_out[i] = aggregate(neighbors, axis=0)

    return s_out
