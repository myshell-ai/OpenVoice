#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Constant-Q transforms"""
import warnings
import numpy as np
from numba import jit

from . import audio
from .fft import get_fftlib
from .convert import cqt_frequencies, note_to_hz
from .spectrum import stft, istft
from .pitch import estimate_tuning
from .._cache import cache
from .. import filters
from .. import util
from ..util.exceptions import ParameterError
from ..util.decorators import deprecate_positional_args

__all__ = ["cqt", "hybrid_cqt", "pseudo_cqt", "icqt", "griffinlim_cqt", "vqt"]

# TODO: ivqt, griffinlim_vqt


@deprecate_positional_args
@cache(level=20)
def cqt(
    y,
    *,
    sr=22050,
    hop_length=512,
    fmin=None,
    n_bins=84,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    norm=1,
    sparsity=0.01,
    window="hann",
    scale=True,
    pad_mode="constant",
    res_type=None,
    dtype=None,
):
    """Compute the constant-Q transform of an audio signal.

    This implementation is based on the recursive sub-sampling method
    described by [#]_.

    .. [#] Schoerkhuber, Christian, and Anssi Klapuri.
        "Constant-Q transform toolbox for music processing."
        7th Sound and Music Computing Conference, Barcelona, Spain. 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at ``fmin``

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float
        Tuning offset in fractions of a bin.

        If ``None``, tuning will be automatically estimated from the signal.

        The minimum frequency of the resulting CQT will be modified to
        ``fmin * 2**(tuning / bins_per_octave)``.

    filter_scale : float > 0
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the CQT response by square-root the length of
        each channel's filter.  This is analogous to ``norm='ortho'`` in FFT.

        If ``False``, do not scale the CQT. This is analogous to
        ``norm=None`` in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.stft` and `numpy.pad`.

    res_type : string [optional]
        The resampling mode for recursive downsampling.

        By default, `cqt` will adaptively select a resampling mode
        which trades off accuracy at high frequencies for efficiency at low frequencies.

        You can override this by specifying a resampling mode as supported by
        `librosa.resample`.  For example, ``res_type='fft'`` will use a high-quality,
        but potentially slow FFT-based down-sampling, while ``res_type='polyphase'`` will
        use a fast, but potentially inaccurate down-sampling.

    dtype : np.dtype
        The (complex) data type of the output array.  By default, this is inferred to match
        the numerical precision of the input signal.

    Returns
    -------
    CQT : np.ndarray [shape=(..., n_bins, t)]
        Constant-Q value each frequency at each time.

    See Also
    --------
    vqt
    librosa.resample
    librosa.util.normalize

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Generate and plot a constant-Q power spectrum

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> C = np.abs(librosa.cqt(y, sr=sr))
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
    ...                                sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
    >>> ax.set_title('Constant-Q power spectrum')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")

    Limit the frequency range

    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
    ...                 n_bins=60))
    >>> C
    array([[6.830e-04, 6.361e-04, ..., 7.362e-09, 9.102e-09],
           [5.366e-04, 4.818e-04, ..., 8.953e-09, 1.067e-08],
           ...,
           [4.288e-02, 4.580e-01, ..., 1.529e-05, 5.572e-06],
           [2.965e-03, 1.508e-01, ..., 8.965e-06, 1.455e-05]])

    Using a higher frequency resolution

    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
    ...                 n_bins=60 * 2, bins_per_octave=12 * 2))
    >>> C
    array([[5.468e-04, 5.382e-04, ..., 5.911e-09, 6.105e-09],
           [4.118e-04, 4.014e-04, ..., 7.788e-09, 8.160e-09],
           ...,
           [2.780e-03, 1.424e-01, ..., 4.225e-06, 2.388e-05],
           [5.147e-02, 6.959e-02, ..., 1.694e-05, 5.811e-06]])
    """

    # CQT is the special case of VQT with gamma=0
    return vqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        gamma=0,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        norm=norm,
        sparsity=sparsity,
        window=window,
        scale=scale,
        pad_mode=pad_mode,
        res_type=res_type,
        dtype=dtype,
    )


@deprecate_positional_args
@cache(level=20)
def hybrid_cqt(
    y,
    *,
    sr=22050,
    hop_length=512,
    fmin=None,
    n_bins=84,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    norm=1,
    sparsity=0.01,
    window="hann",
    scale=True,
    pad_mode="constant",
    res_type=None,
    dtype=None,
):
    """Compute the hybrid constant-Q transform of an audio signal.

    Here, the hybrid CQT uses the pseudo CQT for higher frequencies where
    the hop_length is longer than half the filter length and the full CQT
    for lower frequencies.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at ``fmin``

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float
        Tuning offset in fractions of a bin.

        If ``None``, tuning will be automatically estimated from the signal.

        The minimum frequency of the resulting CQT will be modified to
        ``fmin * 2**(tuning / bins_per_octave)``.

    filter_scale : float > 0
        Filter filter_scale factor. Larger values use longer windows.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the CQT response by square-root the length of
        each channel's filter.  This is analogous to ``norm='ortho'`` in FFT.

        If ``False``, do not scale the CQT. This is analogous to
        ``norm=None`` in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.stft` and `numpy.pad`.

    res_type : string
        Resampling mode.  See `librosa.cqt` for details.

    dtype : np.dtype, optional
        The complex dtype to use for computing the CQT.
        By default, this is inferred to match the precision of
        the input signal.

    Returns
    -------
    CQT : np.ndarray [shape=(..., n_bins, t), dtype=np.float]
        Constant-Q energy for each frequency at each time.

    See Also
    --------
    cqt
    pseudo_cqt

    Notes
    -----
    This function caches at level 20.

    """

    if fmin is None:
        # C1 by default
        fmin = note_to_hz("C1")

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # Get all CQT frequencies
    freqs = cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    # Compute an alpha parameter, just in case we need it
    alpha = __bpo_to_alpha(bins_per_octave)

    # Compute the length of each constant-Q basis function
    lengths, _ = filters.wavelet_lengths(
        freqs=freqs, sr=sr, filter_scale=filter_scale, window=window, alpha=alpha
    )

    # Determine which filters to use with Pseudo CQT
    # These are the ones that fit within 2 hop lengths after padding
    pseudo_filters = 2.0 ** np.ceil(np.log2(lengths)) < 2 * hop_length

    n_bins_pseudo = int(np.sum(pseudo_filters))

    n_bins_full = n_bins - n_bins_pseudo
    cqt_resp = []

    if n_bins_pseudo > 0:
        fmin_pseudo = np.min(freqs[pseudo_filters])

        cqt_resp.append(
            pseudo_cqt(
                y,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin_pseudo,
                n_bins=n_bins_pseudo,
                bins_per_octave=bins_per_octave,
                filter_scale=filter_scale,
                norm=norm,
                sparsity=sparsity,
                window=window,
                scale=scale,
                pad_mode=pad_mode,
                dtype=dtype,
            )
        )

    if n_bins_full > 0:
        cqt_resp.append(
            np.abs(
                cqt(
                    y,
                    sr=sr,
                    hop_length=hop_length,
                    fmin=fmin,
                    n_bins=n_bins_full,
                    bins_per_octave=bins_per_octave,
                    filter_scale=filter_scale,
                    norm=norm,
                    sparsity=sparsity,
                    window=window,
                    scale=scale,
                    pad_mode=pad_mode,
                    res_type=res_type,
                    dtype=dtype,
                )
            )
        )

    # Propagate dtype from the last component
    return __trim_stack(cqt_resp, n_bins, cqt_resp[-1].dtype)


@deprecate_positional_args
@cache(level=20)
def pseudo_cqt(
    y,
    *,
    sr=22050,
    hop_length=512,
    fmin=None,
    n_bins=84,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    norm=1,
    sparsity=0.01,
    window="hann",
    scale=True,
    pad_mode="constant",
    dtype=None,
):
    """Compute the pseudo constant-Q transform of an audio signal.

    This uses a single fft size that is the smallest power of 2 that is greater
    than or equal to the max of:

        1. The longest CQT filter
        2. 2x the hop_length

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at ``fmin``

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float
        Tuning offset in fractions of a bin.

        If ``None``, tuning will be automatically estimated from the signal.

        The minimum frequency of the resulting CQT will be modified to
        ``fmin * 2**(tuning / bins_per_octave)``.

    filter_scale : float > 0
        Filter filter_scale factor. Larger values use longer windows.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the CQT response by square-root the length of
        each channel's filter.  This is analogous to ``norm='ortho'`` in FFT.

        If ``False``, do not scale the CQT. This is analogous to
        ``norm=None`` in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.stft` and `numpy.pad`.

    dtype : np.dtype, optional
        The complex data type for CQT calculations.
        By default, this is inferred to match the precision of the input signal.

    Returns
    -------
    CQT : np.ndarray [shape=(..., n_bins, t), dtype=np.float]
        Pseudo Constant-Q energy for each frequency at each time.

    Notes
    -----
    This function caches at level 20.

    """

    if fmin is None:
        # C1 by default
        fmin = note_to_hz("C1")

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)

    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    freqs = cqt_frequencies(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)

    alpha = __bpo_to_alpha(bins_per_octave)

    lengths, _ = filters.wavelet_lengths(
        freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, alpha=alpha
    )

    fft_basis, n_fft, _ = __vqt_filter_fft(
        sr,
        freqs,
        filter_scale,
        norm,
        sparsity,
        hop_length=hop_length,
        window=window,
        dtype=dtype,
        alpha=alpha,
    )

    fft_basis = np.abs(fft_basis)

    # Compute the magnitude-only CQT response
    C = __cqt_response(
        y,
        n_fft,
        hop_length,
        fft_basis,
        pad_mode,
        window="hann",
        dtype=dtype,
        phase=False,
    )

    if scale:
        C /= np.sqrt(n_fft)
    else:
        # reshape lengths to match dimension properly
        lengths = util.expand_to(lengths, ndim=C.ndim, axes=-2)

        C *= np.sqrt(lengths / n_fft)

    return C


@deprecate_positional_args
@cache(level=40)
def icqt(
    C,
    *,
    sr=22050,
    hop_length=512,
    fmin=None,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    norm=1,
    sparsity=0.01,
    window="hann",
    scale=True,
    length=None,
    res_type="fft",
    dtype=None,
):
    """Compute the inverse constant-Q transform.

    Given a constant-Q transform representation ``C`` of an audio signal ``y``,
    this function produces an approximation ``y_hat``.

    Parameters
    ----------
    C : np.ndarray, [shape=(..., n_bins, n_frames)]
        Constant-Q representation as produced by `cqt`

    sr : number > 0 [scalar]
        sampling rate of the signal

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float [scalar]
        Tuning offset in fractions of a bin.

        The minimum frequency of the CQT will be modified to
        ``fmin * 2**(tuning / bins_per_octave)``.

    filter_scale : float > 0 [scalar]
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the CQT response by square-root the length
        of each channel's filter. This is analogous to ``norm='ortho'`` in FFT.

        If ``False``, do not scale the CQT. This is analogous to ``norm=None``
        in FFT.

    length : int > 0, optional
        If provided, the output ``y`` is zero-padded or clipped to exactly
        ``length`` samples.

    res_type : string
        Resampling mode.  By default, this uses ``'fft'`` mode for high-quality
        reconstruction, but this may be slow depending on your signal duration.
        See `librosa.resample` for supported modes.

    dtype : numeric type
        Real numeric type for ``y``.  Default is inferred to match the numerical
        precision of the input CQT.

    Returns
    -------
    y : np.ndarray, [shape=(..., n_samples), dtype=np.float]
        Audio time-series reconstructed from the CQT representation.

    See Also
    --------
    cqt
    librosa.resample

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    Using default parameters

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> C = librosa.cqt(y=y, sr=sr)
    >>> y_hat = librosa.icqt(C=C, sr=sr)

    Or with a different hop length and frequency resolution:

    >>> hop_length = 256
    >>> bins_per_octave = 12 * 3
    >>> C = librosa.cqt(y=y, sr=sr, hop_length=256, n_bins=7*bins_per_octave,
    ...                 bins_per_octave=bins_per_octave)
    >>> y_hat = librosa.icqt(C=C, sr=sr, hop_length=hop_length,
    ...                 bins_per_octave=bins_per_octave)
    """
    if fmin is None:
        fmin = note_to_hz("C1")

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # Get the top octave of frequencies
    n_bins = C.shape[-2]

    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))

    # truncate the cqt to max frames if helpful
    freqs = cqt_frequencies(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    alpha = __bpo_to_alpha(bins_per_octave)

    lengths, f_cutoff = filters.wavelet_lengths(
        freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, alpha=alpha
    )

    # Trim the CQT to only what's necessary for reconstruction
    if length is not None:
        n_frames = int(np.ceil((length + max(lengths)) / hop_length))
        C = C[..., :n_frames]

    C_scale = np.sqrt(lengths)

    # This shape array will be used for broadcasting the basis scale
    # we'll have to adapt this per octave within the loop
    y = None

    # Assume the top octave is at the full rate
    srs = [sr]
    hops = [hop_length]

    for i in range(n_octaves - 1):
        if hops[0] % 2 == 0:
            # We can downsample:
            srs.insert(0, srs[0] * 0.5)
            hops.insert(0, hops[0] // 2)
        else:
            # We're out of downsamplings, carry forward
            srs.insert(0, srs[0])
            hops.insert(0, hops[0])

    for i, (my_sr, my_hop) in enumerate(zip(srs, hops)):

        # How many filters are in this octave?
        n_filters = min(bins_per_octave, n_bins - bins_per_octave * i)

        # Slice out the current octave
        sl = slice(bins_per_octave * i, bins_per_octave * i + n_filters)

        fft_basis, n_fft, _ = __vqt_filter_fft(
            my_sr,
            freqs[sl],
            filter_scale,
            norm,
            sparsity,
            window=window,
            dtype=dtype,
            alpha=alpha,
        )

        # Transpose the basis
        inv_basis = fft_basis.H.todense()

        # Compute each filter's frequency-domain power
        freq_power = 1 / np.sum(np.abs(np.asarray(inv_basis)) ** 2, axis=0)

        # Compensate for length normalization in the forward transform
        freq_power *= n_fft / lengths[sl]

        # Inverse-project the basis for each octave
        if scale:
            # scale=True ==> re-scale by sqrt(lengths)
            D_oct = np.einsum(
                "fc,c,c,...ct->...ft",
                inv_basis,
                C_scale[sl],
                freq_power,
                C[..., sl, :],
                optimize=True,
            )
        else:
            D_oct = np.einsum(
                "fc,c,...ct->...ft", inv_basis, freq_power, C[..., sl, :], optimize=True
            )

        y_oct = istft(D_oct, window="ones", hop_length=my_hop, dtype=dtype)

        y_oct = audio.resample(
            y_oct,
            orig_sr=1,
            target_sr=sr // my_sr,
            res_type=res_type,
            scale=False,
            fix=False,
        )

        if y is None:
            y = y_oct
        else:
            y[..., : y_oct.shape[-1]] += y_oct
    if length:
        y = util.fix_length(y, size=length)

    return y


@deprecate_positional_args
@cache(level=20)
def vqt(
    y,
    *,
    sr=22050,
    hop_length=512,
    fmin=None,
    n_bins=84,
    gamma=None,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    norm=1,
    sparsity=0.01,
    window="hann",
    scale=True,
    pad_mode="constant",
    res_type=None,
    dtype=None,
):
    """Compute the variable-Q transform of an audio signal.

    This implementation is based on the recursive sub-sampling method
    described by [#]_.

    .. [#] Schörkhuber, Christian, Anssi Klapuri, Nicki Holighaus, and Monika Dörfler.
        "A Matlab toolbox for efficient perfect reconstruction time-frequency
        transforms with log-frequency resolution."
        In Audio Engineering Society Conference: 53rd International Conference: Semantic Audio.
        Audio Engineering Society, 2014.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    hop_length : int > 0 [scalar]
        number of samples between successive VQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at ``fmin``

    gamma : number > 0 [scalar]
        Bandwidth offset for determining filter lengths.

        If ``gamma=0``, produces the constant-Q transform.

        If 'gamma=None', gamma will be calculated such that filter bandwidths are equal to a
        constant fraction of the equivalent rectangular bandwidths (ERB). This is accomplished
        by solving for the gamma which gives::

            B_k = alpha * f_k + gamma = C * ERB(f_k),

        where ``B_k`` is the bandwidth of filter ``k`` with center frequency ``f_k``, alpha
        is the inverse of what would be the constant Q-factor, and ``C = alpha / 0.108`` is the
        constant fraction across all filters.

        Here we use ``ERB(f_k) = 24.7 + 0.108 * f_k``, the best-fit curve derived
        from experimental data in [#]_.

        .. [#] Glasberg, Brian R., and Brian CJ Moore.
            "Derivation of auditory filter shapes from notched-noise data."
            Hearing research 47.1-2 (1990): 103-138.

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float
        Tuning offset in fractions of a bin.

        If ``None``, tuning will be automatically estimated from the signal.

        The minimum frequency of the resulting VQT will be modified to
        ``fmin * 2**(tuning / bins_per_octave)``.

    filter_scale : float > 0
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the VQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the VQT response by square-root the length of
        each channel's filter.  This is analogous to ``norm='ortho'`` in FFT.

        If ``False``, do not scale the VQT. This is analogous to
        ``norm=None`` in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.stft` and `numpy.pad`.

    res_type : string [optional]
        The resampling mode for recursive downsampling.

        By default, `vqt` will adaptively select a resampling mode
        which trades off accuracy at high frequencies for efficiency at low frequencies.

        You can override this by specifying a resampling mode as supported by
        `librosa.resample`.  For example, ``res_type='fft'`` will use a high-quality,
        but potentially slow FFT-based down-sampling, while ``res_type='polyphase'`` will
        use a fast, but potentially inaccurate down-sampling.

    dtype : np.dtype
        The dtype of the output array.  By default, this is inferred to match the
        numerical precision of the input signal.

    Returns
    -------
    VQT : np.ndarray [shape=(..., n_bins, t), dtype=np.complex]
        Variable-Q value each frequency at each time.

    See Also
    --------
    cqt

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Generate and plot a variable-Q power spectrum

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=5)
    >>> C = np.abs(librosa.cqt(y, sr=sr))
    >>> V = np.abs(librosa.vqt(y, sr=sr))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
    ...                          sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[0])
    >>> ax[0].set(title='Constant-Q power spectrum', xlabel=None)
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(V, ref=np.max),
    ...                                sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[1])
    >>> ax[1].set_title('Variable-Q power spectrum')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """

    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    if fmin is None:
        # C1 by default
        fmin = note_to_hz("C1")

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)

    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    freqs_top = freqs[-bins_per_octave:]

    fmax_t = np.max(freqs_top)
    alpha = __bpo_to_alpha(bins_per_octave)

    lengths, filter_cutoff = filters.wavelet_lengths(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
        alpha=alpha,
    )

    # Determine required resampling quality
    nyquist = sr / 2.0

    if filter_cutoff > nyquist:
        raise ParameterError(
            f"Wavelet basis with max frequency={fmax_t} would exceed the Nyquist frequency={nyquist}. "
            "Try reducing the number of frequency bins."
        )

    auto_resample = False
    if not res_type:
        auto_resample = True
        if filter_cutoff < audio.BW_FASTEST * nyquist:
            res_type = "kaiser_fast"
        else:
            res_type = "kaiser_best"

    y, sr, hop_length = __early_downsample(
        y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale
    )

    vqt_resp = []

    # Skip this block for now
    oct_start = 0
    if auto_resample and res_type != "kaiser_fast":

        # Do the top octave before resampling to allow for fast resampling
        freqs_top = freqs[-n_filters:]

        fft_basis, n_fft, _ = __vqt_filter_fft(
            sr,
            freqs_top,
            filter_scale,
            norm,
            sparsity,
            window=window,
            gamma=gamma,
            dtype=dtype,
            alpha=alpha,
        )

        # Compute the VQT filter response and append it to the stack
        vqt_resp.append(
            __cqt_response(y, n_fft, hop_length, fft_basis, pad_mode, dtype=dtype)
        )

        oct_start = 1

        res_type = "kaiser_fast"

    # Iterate down the octaves
    my_y, my_sr, my_hop = y, sr, hop_length

    for i in range(oct_start, n_octaves):

        # Slice out the current octave of filters
        if i == 0:
            sl = slice(-n_filters, None)
        else:
            sl = slice(-n_filters * (i + 1), -n_filters * i)

        # This may be incorrect with early downsampling
        freqs_oct = freqs[sl]

        fft_basis, n_fft, _ = __vqt_filter_fft(
            my_sr,
            freqs_oct,
            filter_scale,
            norm,
            sparsity,
            window=window,
            gamma=gamma,
            dtype=dtype,
            alpha=alpha,
        )

        # Re-scale the filters to compensate for downsampling
        fft_basis[:] *= np.sqrt(sr / my_sr)

        # Compute the vqt filter response and append to the stack
        vqt_resp.append(
            __cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode, dtype=dtype)
        )

        if my_hop % 2 == 0:
            my_hop //= 2
            my_sr /= 2.0
            my_y = audio.resample(
                my_y, orig_sr=2, target_sr=1, res_type=res_type, scale=True
            )

    V = __trim_stack(vqt_resp, n_bins, dtype)

    if scale:
        # Recompute lengths here because early downsampling may have changed
        # our sampling rate
        lengths, _ = filters.wavelet_lengths(
            freqs=freqs,
            sr=sr,
            window=window,
            filter_scale=filter_scale,
            gamma=gamma,
            alpha=alpha,
        )

        # reshape lengths to match V shape
        lengths = util.expand_to(lengths, ndim=V.ndim, axes=-2)
        V /= np.sqrt(lengths)

    return V


@cache(level=10)
def __vqt_filter_fft(
    sr,
    freqs,
    filter_scale,
    norm,
    sparsity,
    hop_length=None,
    window="hann",
    gamma=0.0,
    dtype=np.complex64,
    alpha=None,
):
    """Generate the frequency domain variable-Q filter basis."""

    basis, lengths = filters.wavelet(
        freqs=freqs,
        sr=sr,
        filter_scale=filter_scale,
        norm=norm,
        pad_fft=True,
        window=window,
        gamma=gamma,
        alpha=alpha,
    )

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):
        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, np.newaxis] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft = get_fftlib()
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

    # sparsify the basis
    fft_basis = util.sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)

    return fft_basis, n_fft, lengths


def __trim_stack(cqt_resp, n_bins, dtype):
    """Helper function to trim and stack a collection of CQT responses"""

    max_col = min(c_i.shape[-1] for c_i in cqt_resp)
    # Grab any leading dimensions
    shape = list(cqt_resp[0].shape)
    shape[-2] = n_bins
    shape[-1] = max_col
    cqt_out = np.empty(shape, dtype=dtype, order="F")

    # Copy per-octave data into output array
    end = n_bins
    for c_i in cqt_resp:
        # By default, take the whole octave
        n_oct = c_i.shape[-2]
        # If the whole octave is more than we can fit,
        # take the highest bins from c_i
        if end < n_oct:
            cqt_out[..., :end, :] = c_i[..., -end:, :max_col]
        else:
            cqt_out[..., end - n_oct : end, :] = c_i[..., :max_col]

        end -= n_oct

    return cqt_out


def __cqt_response(
    y, n_fft, hop_length, fft_basis, mode, window="ones", phase=True, dtype=None
):
    """Compute the filter response with a target STFT hop."""

    # Compute the STFT matrix
    D = stft(
        y, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=mode, dtype=dtype
    )

    if not phase:
        D = np.abs(D)

    # Reshape D to Dr
    Dr = D.reshape((-1, D.shape[-2], D.shape[-1]))
    output_flat = np.empty(
        (Dr.shape[0], fft_basis.shape[0], Dr.shape[-1]), dtype=D.dtype
    )

    # iterate over channels
    #   project fft_basis.dot(Dr[i])
    for i in range(Dr.shape[0]):
        output_flat[i] = fft_basis.dot(Dr[i])

    # reshape Dr to match D's leading dimensions again
    shape = list(D.shape)
    shape[-2] = fft_basis.shape[0]
    return output_flat.reshape(shape)


def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    """Compute the number of early downsampling operations"""

    downsample_count1 = max(
        0, int(np.ceil(np.log2(audio.BW_FASTEST * nyquist / filter_cutoff)) - 1) - 1
    )

    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def __early_downsample(
    y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale
):
    """Perform early downsampling on an audio signal, if it applies."""

    downsample_count = __early_downsample_count(
        nyquist, filter_cutoff, hop_length, n_octaves
    )

    if downsample_count > 0 and res_type == "kaiser_fast":
        downsample_factor = 2 ** (downsample_count)

        hop_length //= downsample_factor

        if y.shape[-1] < downsample_factor:
            raise ParameterError(
                "Input signal length={:d} is too short for "
                "{:d}-octave CQT".format(len(y), n_octaves)
            )

        new_sr = sr / float(downsample_factor)
        y = audio.resample(
            y, orig_sr=sr, target_sr=new_sr, res_type=res_type, scale=True
        )

        # If we're not going to length-scale after CQT, we
        # need to compensate for the downsampling factor here
        if not scale:
            y *= np.sqrt(downsample_factor)

        sr = new_sr

    return y, sr, hop_length


@jit(nopython=True, cache=True)
def __num_two_factors(x):
    """Return how many times integer x can be evenly divided by 2.

    Returns 0 for non-positive integers.
    """
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos


@deprecate_positional_args
def griffinlim_cqt(
    C,
    *,
    n_iter=32,
    sr=22050,
    hop_length=512,
    fmin=None,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    norm=1,
    sparsity=0.01,
    window="hann",
    scale=True,
    pad_mode="constant",
    res_type="kaiser_fast",
    dtype=None,
    length=None,
    momentum=0.99,
    init="random",
    random_state=None,
):
    """Approximate constant-Q magnitude spectrogram inversion using the "fast" Griffin-Lim
    algorithm.

    Given the magnitude of a constant-Q spectrogram (``C``), the algorithm randomly initializes
    phase estimates, and then alternates forward- and inverse-CQT operations. [#]_

    This implementation is based on the (fast) Griffin-Lim method for Short-time Fourier Transforms, [#]_
    but adapted for use with constant-Q spectrograms.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    .. [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    Parameters
    ----------
    C : np.ndarray [shape=(..., n_bins, n_frames)]
        The constant-Q magnitude spectrogram

    n_iter : int > 0
        The number of iterations to run

    sr : number > 0
        Audio sampling rate

    hop_length : int > 0
        The hop length of the CQT

    fmin : number > 0
        Minimum frequency for the CQT.

        If not provided, it defaults to `C1`.

    bins_per_octave : int > 0
        Number of bins per octave

    tuning : float
        Tuning deviation from A440, in fractions of a bin

    filter_scale : float > 0
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the CQT response by square-root the length
        of each channel's filter.  This is analogous to ``norm='ortho'``
        in FFT.

        If ``False``, do not scale the CQT. This is analogous to ``norm=None``
        in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.stft` and `numpy.pad`.

    res_type : string
        The resampling mode for recursive downsampling.

        By default, CQT uses an adaptive mode selection to
        trade accuracy at high frequencies for efficiency at low
        frequencies.

        Griffin-Lim uses the efficient (fast) resampling mode by default.

        See ``librosa.resample`` for a list of available options.

    dtype : numeric type
        Real numeric type for ``y``.  Default is inferred to match the precision
        of the input CQT.

    length : int > 0, optional
        If provided, the output ``y`` is zero-padded or clipped to exactly
        ``length`` samples.

    momentum : float > 0
        The momentum parameter for fast Griffin-Lim.
        Setting this to 0 recovers the original Griffin-Lim method.
        Values near 1 can lead to faster convergence, but above 1 may not converge.

    init : None or 'random' [default]
        If 'random' (the default), then phase values are initialized randomly
        according to ``random_state``.  This is recommended when the input ``C`` is
        a magnitude spectrogram with no initial phase estimates.

        If ``None``, then the phase is initialized from ``C``.  This is useful when
        an initial guess for phase can be provided, or when you want to resume
        Griffin-Lim from a previous output.

    random_state : None, int, or np.random.RandomState
        If int, random_state is the seed used by the random number generator
        for phase initialization.

        If `np.random.RandomState` instance, the random number generator itself.

        If ``None``, defaults to the current `np.random` object.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        time-domain signal reconstructed from ``C``

    See Also
    --------
    cqt
    icqt
    griffinlim
    filters.get_window
    resample

    Examples
    --------
    A basis CQT inverse example

    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), sr=None)
    >>> # Get the CQT magnitude, 7 octaves at 36 bins per octave
    >>> C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=36, n_bins=7*36))
    >>> # Invert using Griffin-Lim
    >>> y_inv = librosa.griffinlim_cqt(C, sr=sr, bins_per_octave=36)
    >>> # And invert without estimating phase
    >>> y_icqt = librosa.icqt(C, sr=sr, bins_per_octave=36)

    Wave-plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> librosa.display.waveshow(y, sr=sr, color='b', ax=ax[0])
    >>> ax[0].set(title='Original', xlabel=None)
    >>> ax[0].label_outer()
    >>> librosa.display.waveshow(y_inv, sr=sr, color='g', ax=ax[1])
    >>> ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
    >>> ax[1].label_outer()
    >>> librosa.display.waveshow(y_icqt, sr=sr, color='r', ax=ax[2])
    >>> ax[2].set(title='Magnitude-only icqt reconstruction')
    """
    if fmin is None:
        fmin = note_to_hz("C1")

    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    if momentum > 1:
        warnings.warn(
            "Griffin-Lim with momentum={} > 1 can be unstable. "
            "Proceed with caution!".format(momentum),
            stacklevel=2,
        )
    elif momentum < 0:
        raise ParameterError(
            "griffinlim_cqt() called with momentum={} < 0".format(momentum)
        )

    # using complex64 will keep the result to minimal necessary precision
    angles = np.empty(C.shape, dtype=np.complex64)
    eps = util.tiny(angles)

    if init == "random":
        # randomly initialize the phase
        angles[:] = np.exp(2j * np.pi * rng.rand(*C.shape))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise ParameterError("init={} must either None or 'random'".format(init))

    # And initialize the previous iterate to 0
    rebuilt = 0.0

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = icqt(
            C * angles,
            sr=sr,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            fmin=fmin,
            tuning=tuning,
            filter_scale=filter_scale,
            window=window,
            length=length,
            res_type=res_type,
            norm=norm,
            scale=scale,
            sparsity=sparsity,
            dtype=dtype,
        )

        # Rebuild the spectrogram
        rebuilt = cqt(
            inverse,
            sr=sr,
            bins_per_octave=bins_per_octave,
            n_bins=C.shape[-2],
            hop_length=hop_length,
            fmin=fmin,
            tuning=tuning,
            filter_scale=filter_scale,
            window=window,
            norm=norm,
            scale=scale,
            sparsity=sparsity,
            pad_mode=pad_mode,
            res_type=res_type,
        )

        # Update our phase estimates
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + eps

    # Return the final phase estimates
    return icqt(
        C * angles,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        fmin=fmin,
        window=window,
        length=length,
        res_type=res_type,
        norm=norm,
        scale=scale,
        sparsity=sparsity,
        dtype=dtype,
    )


def __bpo_to_alpha(bins_per_octave):
    """Compute the alpha coefficient for a given number of bins per octave

    Parameters
    ----------
    bins_per_octave : int

    Returns
    -------
    alpha : number > 0
    """

    r = 2 ** (1 / bins_per_octave)
    return (r ** 2 - 1) / (r ** 2 + 1)
