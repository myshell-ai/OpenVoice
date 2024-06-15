#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filters
=======

Filter bank construction
------------------------
.. autosummary::
    :toctree: generated/

    mel
    chroma
    wavelet
    semitone_filterbank

Window functions
----------------
.. autosummary::
    :toctree: generated/

    window_bandwidth
    get_window

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    wavelet_lengths
    cq_to_chroma
    mr_frequencies
    window_sumsquare
    diagonal_filter

Deprecated
----------
.. autosummary::
    :toctree: generated/

    constant_q
    constant_q_lengths

"""
import warnings

import numpy as np
import scipy
import scipy.signal
import scipy.ndimage

from numba import jit

from ._cache import cache
from . import util
from .util.exceptions import ParameterError
from .util.decorators import deprecated, deprecate_positional_args

from .core.convert import note_to_hz, hz_to_midi, midi_to_hz, hz_to_octs
from .core.convert import fft_frequencies, mel_frequencies

__all__ = [
    "mel",
    "chroma",
    "constant_q",
    "constant_q_lengths",
    "cq_to_chroma",
    "window_bandwidth",
    "get_window",
    "mr_frequencies",
    "semitone_filterbank",
    "window_sumsquare",
    "diagonal_filter",
    "wavelet",
    "wavelet_lengths",
]

# Dictionary of window function bandwidths

WINDOW_BANDWIDTHS = {
    "bart": 1.3334961334912805,
    "barthann": 1.4560255965133932,
    "bartlett": 1.3334961334912805,
    "bkh": 2.0045975283585014,
    "black": 1.7269681554262326,
    "blackharr": 2.0045975283585014,
    "blackman": 1.7269681554262326,
    "blackmanharris": 2.0045975283585014,
    "blk": 1.7269681554262326,
    "bman": 1.7859588613860062,
    "bmn": 1.7859588613860062,
    "bohman": 1.7859588613860062,
    "box": 1.0,
    "boxcar": 1.0,
    "brt": 1.3334961334912805,
    "brthan": 1.4560255965133932,
    "bth": 1.4560255965133932,
    "cosine": 1.2337005350199792,
    "flat": 2.7762255046484143,
    "flattop": 2.7762255046484143,
    "flt": 2.7762255046484143,
    "halfcosine": 1.2337005350199792,
    "ham": 1.3629455320350348,
    "hamm": 1.3629455320350348,
    "hamming": 1.3629455320350348,
    "han": 1.50018310546875,
    "hann": 1.50018310546875,
    "hanning": 1.50018310546875,
    "nut": 1.9763500280946082,
    "nutl": 1.9763500280946082,
    "nuttall": 1.9763500280946082,
    "ones": 1.0,
    "par": 1.9174603174603191,
    "parz": 1.9174603174603191,
    "parzen": 1.9174603174603191,
    "rect": 1.0,
    "rectangular": 1.0,
    "tri": 1.3331706523555851,
    "triang": 1.3331706523555851,
    "triangle": 1.3331706523555851,
}


@deprecate_positional_args
@cache(level=10)
def mel(
    *,
    sr,
    n_fft,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm="slaney",
    dtype=np.float32,
):
    """Create a Mel filter-bank.

    This produces a linear transformation matrix to project
    FFT bins onto Mel-frequency bins.

    Parameters
    ----------
    sr : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft : int > 0 [scalar]
        number of FFT components

    n_mels : int > 0 [scalar]
        number of Mel bands to generate

    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``

    htk : bool [scalar]
        use HTK formula instead of Slaney

    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization).

        If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
        See `librosa.util.normalize` for a full description of supported norm values
        (including `+-np.inf`).

        Otherwise, leave all the triangles aiming for a peak value of 1.0

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    M : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    See Also
    --------
    librosa.util.normalize

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> melfb = librosa.filters.mel(sr=22050, n_fft=2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])

    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(sr=22050, n_fft=2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)
    >>> ax.set(ylabel='Mel filter', title='Mel filter bank')
    >>> fig.colorbar(img, ax=ax)
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    else:
        weights = util.normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels.",
            stacklevel=2,
        )

    return weights


@deprecate_positional_args
@cache(level=10)
def chroma(
    *,
    sr,
    n_fft,
    n_chroma=12,
    tuning=0.0,
    ctroct=5.0,
    octwidth=2,
    norm=2,
    base_c=True,
    dtype=np.float32,
):
    """Create a chroma filter bank.

    This creates a linear transformation matrix to project
    FFT bins onto chroma bins (i.e. pitch classes).

    Parameters
    ----------
    sr : number > 0 [scalar]
        audio sampling rate

    n_fft : int > 0 [scalar]
        number of FFT bins

    n_chroma : int > 0 [scalar]
        number of chroma bins

    tuning : float
        Tuning deviation from A440 in fractions of a chroma bin.

    ctroct : float > 0 [scalar]

    octwidth : float > 0 or None [scalar]
        ``ctroct`` and ``octwidth`` specify a dominance window:
        a Gaussian weighting centered on ``ctroct`` (in octs, A0 = 27.5Hz)
        and with a gaussian half-width of ``octwidth``.

        Set ``octwidth`` to `None` to use a flat weighting.

    norm : float > 0 or np.inf
        Normalization factor for each filter

    base_c : bool
        If True, the filter bank will start at 'C'.
        If False, the filter bank will start at 'A'.

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    wts : ndarray [shape=(n_chroma, 1 + n_fft / 2)]
        Chroma filter matrix

    See Also
    --------
    librosa.util.normalize
    librosa.feature.chroma_stft

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Build a simple chroma filter bank

    >>> chromafb = librosa.filters.chroma(sr=22050, n_fft=4096)
    array([[  1.689e-05,   3.024e-04, ...,   4.639e-17,   5.327e-17],
           [  1.716e-05,   2.652e-04, ...,   2.674e-25,   3.176e-25],
    ...,
           [  1.578e-05,   3.619e-04, ...,   8.577e-06,   9.205e-06],
           [  1.643e-05,   3.355e-04, ...,   1.474e-10,   1.636e-10]])

    Use quarter-tones instead of semitones

    >>> librosa.filters.chroma(sr=22050, n_fft=4096, n_chroma=24)
    array([[  1.194e-05,   2.138e-04, ...,   6.297e-64,   1.115e-63],
           [  1.206e-05,   2.009e-04, ...,   1.546e-79,   2.929e-79],
    ...,
           [  1.162e-05,   2.372e-04, ...,   6.417e-38,   9.923e-38],
           [  1.180e-05,   2.260e-04, ...,   4.697e-50,   7.772e-50]])

    Equally weight all octaves

    >>> librosa.filters.chroma(sr=22050, n_fft=4096, octwidth=None)
    array([[  3.036e-01,   2.604e-01, ...,   2.445e-16,   2.809e-16],
           [  3.084e-01,   2.283e-01, ...,   1.409e-24,   1.675e-24],
    ...,
           [  2.836e-01,   3.116e-01, ...,   4.520e-05,   4.854e-05],
           [  2.953e-01,   2.888e-01, ...,   7.768e-10,   8.629e-10]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(chromafb, x_axis='linear', ax=ax)
    >>> ax.set(ylabel='Chroma filter', title='Chroma filter bank')
    >>> fig.colorbar(img, ax=ax)
    """

    wts = np.zeros((n_chroma, n_fft))

    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    frqbins = n_chroma * hz_to_octs(
        frequencies, tuning=tuning, bins_per_octave=n_chroma
    )

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))

    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1], 1.0), [1]))

    D = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype="d")).T

    n_chroma2 = np.round(float(n_chroma) / 2)

    # Project into range -n_chroma/2 .. n_chroma/2
    # add on fixed offset of 10*n_chroma to ensure all values passed to
    # rem are positive
    D = np.remainder(D + n_chroma2 + 10 * n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2 * D / np.tile(binwidthbins, (n_chroma, 1))) ** 2)

    # normalize each column
    wts = util.normalize(wts, norm=norm, axis=0)

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((frqbins / n_chroma - ctroct) / octwidth) ** 2)),
            (n_chroma, 1),
        )

    if base_c:
        wts = np.roll(wts, -3 * (n_chroma // 12), axis=0)

    # remove aliasing columns, copy to ensure row-contiguity
    return np.ascontiguousarray(wts[:, : int(1 + n_fft / 2)], dtype=dtype)


def __float_window(window_spec):
    """Decorator function for windows with fractional input.

    This function guarantees that for fractional ``x``, the following hold:

    1. ``__float_window(window_function)(x)`` has length ``np.ceil(x)``
    2. all values from ``np.floor(x)`` are set to 0.

    For integer-valued ``x``, there should be no change in behavior.
    """

    def _wrap(n, *args, **kwargs):
        """The wrapped window"""
        n_min, n_max = int(np.floor(n)), int(np.ceil(n))

        window = get_window(window_spec, n_min)

        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))], mode="constant")

        window[n_min:] = 0.0

        return window

    return _wrap


@deprecated(version="0.9.0", version_removed="1.0")
@deprecate_positional_args
def constant_q(
    *,
    sr,
    fmin=None,
    n_bins=84,
    bins_per_octave=12,
    window="hann",
    filter_scale=1,
    pad_fft=True,
    norm=1,
    dtype=np.complex64,
    gamma=0,
    **kwargs,
):
    r"""Construct a constant-Q basis.

    This function constructs a filter bank similar to Morlet wavelets,
    where complex exponentials are windowed to different lengths
    such that the number of cycles remains fixed for all frequencies.

    By default, a Hann window (rather than the Gaussian window of Morlet wavelets)
    is used, but this can be controlled by the ``window`` parameter.

    Frequencies are spaced geometrically, increasing by a factor of
    ``(2**(1./bins_per_octave))`` at each successive band.

    .. warning:: This function is deprecated as of v0.9 and will be removed in 1.0.
        See `librosa.filters.wavelet`.

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin. Defaults to `C1 ~= 32.70`

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    window : string, tuple, number, or function
        Windowing function to apply to filters.

    filter_scale : float > 0 [scalar]
        Scale of filter windows.
        Small values (<1) use shorter windows for higher temporal resolution.

    pad_fft : boolean
        Center-pad all filters up to the nearest integral power of 2.

        By default, padding is done with zeros, but this can be overridden
        by setting the ``mode=`` field in *kwargs*.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See librosa.util.normalize

    gamma : number >= 0
        Bandwidth offset for variable-Q transforms.
        ``gamma=0`` produces a constant-Q filterbank.

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 64-bit (single precision) complex floating point.

    **kwargs : additional keyword arguments
        Arguments to `np.pad()` when ``pad==True``.

    Returns
    -------
    filters : np.ndarray, ``len(filters) == n_bins``
        ``filters[i]`` is ``i``\ th time-domain CQT basis filter
    lengths : np.ndarray, ``len(lengths) == n_bins``
        The (fractional) length of each filter

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    wavelet
    constant_q_lengths
    librosa.cqt
    librosa.vqt
    librosa.util.normalize

    Examples
    --------
    Use a shorter window for each filter

    >>> basis, lengths = librosa.filters.constant_q(sr=22050, filter_scale=0.5)

    Plot one octave of filters in time and frequency

    >>> import matplotlib.pyplot as plt
    >>> basis, lengths = librosa.filters.constant_q(sr=22050)
    >>> fig, ax = plt.subplots(nrows=2, figsize=(10, 6))
    >>> notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
    >>> for i, (f, n) in enumerate(zip(basis, notes[:12])):
    ...     f_scale = librosa.util.normalize(f) / 2
    ...     ax[0].plot(i + f_scale.real)
    ...     ax[0].plot(i + f_scale.imag, linestyle=':')
    >>> ax[0].set(yticks=np.arange(len(notes[:12])), yticklabels=notes[:12],
    ...           ylabel='CQ filters',
    ...           title='CQ filters (one octave, time domain)',
    ...           xlabel='Time (samples at 22050 Hz)')
    >>> ax[0].legend(['Real', 'Imaginary'])
    >>> F = np.abs(np.fft.fftn(basis, axes=[-1]))
    >>> # Keep only the positive frequencies
    >>> F = F[:, :(1 + F.shape[1] // 2)]
    >>> librosa.display.specshow(F, x_axis='linear', y_axis='cqt_note', ax=ax[1])
    >>> ax[1].set(ylabel='CQ filters', title='CQ filter magnitudes (frequency domain)')
    """

    if fmin is None:
        fmin = note_to_hz("C1")

    # Pass-through parameters to get the filter lengths
    lengths = constant_q_lengths(
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
    )

    freqs = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = np.exp(
            np.arange(-ilen // 2, ilen // 2, dtype=float) * 1j * 2 * np.pi * freq / sr
        )

        # Apply the windowing function
        sig = sig * __float_window(window)(len(sig))

        # Normalize
        sig = util.normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0 ** (np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray(
        [util.pad_center(filt, size=max_len, **kwargs) for filt in filters], dtype=dtype
    )

    return filters, np.asarray(lengths)


@deprecated(version="0.9.0", version_removed="1.0")
@deprecate_positional_args
@cache(level=10)
def constant_q_lengths(
    *, sr, fmin, n_bins=84, bins_per_octave=12, window="hann", filter_scale=1, gamma=0
):
    r"""Return length of each filter in a constant-Q basis.

    .. warning:: This function is deprecated as of v0.9 and will be removed in 1.0.
        See `librosa.filters.wavelet_lengths`.

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    fmin : float > 0 [scalar]
        Minimum frequency bin.
    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).
    bins_per_octave : int > 0 [scalar]
        Number of bins per octave
    window : str or callable
        Window function to use on filters
    filter_scale : float > 0 [scalar]
        Resolution of filter windows. Larger values use longer windows.
    gamma : number >= 0
        Bandwidth offset for variable-Q transforms.
        ``gamma=0`` produces a constant-Q filterbank.

    Returns
    -------
    lengths : np.ndarray
        The length of each filter.

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    wavelet_lengths
    """

    if fmin <= 0:
        raise ParameterError("fmin must be strictly positive")

    if bins_per_octave <= 0:
        raise ParameterError("bins_per_octave must be positive")

    if filter_scale <= 0:
        raise ParameterError("filter_scale must be positive")

    if n_bins <= 0 or not isinstance(n_bins, (int, np.integer)):
        raise ParameterError("n_bins must be a positive integer")

    # Compute the frequencies
    freq = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    #
    # Balance filter bandwidths
    alpha = (2.0 ** (2 / bins_per_octave) - 1) / (2.0 ** (2 / bins_per_octave) + 1)
    Q = float(filter_scale) / alpha

    if max(freq * (1 + 0.5 * window_bandwidth(window) / Q)) > sr / 2.0:
        raise ParameterError(
            f"Maximum filter frequency={max(freq):.2f} would exceed Nyquist={sr/2}"
        )

    # Convert frequencies to filter lengths
    lengths = Q * sr / (freq + gamma / alpha)

    return lengths


@deprecate_positional_args
@cache(level=10)
def wavelet_lengths(
    *, freqs, sr=22050, window="hann", filter_scale=1, gamma=0, alpha=None
):
    """Return length of each filter in a wavelet basis.

    Parameters
    ----------
    freqs : np.ndarray (positive)
        Center frequencies of the filters (in Hz).
        Must be in ascending order.

    sr : number > 0 [scalar]
        Audio sampling rate

    window : str or callable
        Window function to use on filters

    filter_scale : float > 0 [scalar]
        Resolution of filter windows. Larger values use longer windows.

    gamma : number >= 0 [scalar, optional]
        Bandwidth offset for determining filter lengths, as used in
        Variable-Q transforms.

        Bandwidth for the k'th filter is determined by::

            B[k] = alpha[k] * freqs[k] + gamma

        ``alpha[k]`` is twice the relative difference between ``freqs[k+1]`` and ``freqs[k-1]``::

            alpha[k] = (freqs[k+1]-freqs[k-1]) / (freqs[k+1]+freqs[k-1])

        If ``freqs`` follows a geometric progression (as in CQT and VQT), the vector
        ``alpha`` is constant and such that::

            (1 + alpha) * freqs[k-1] = (1 - alpha) * freqs[k+1]

        Furthermore, if ``gamma=0`` (default), ``alpha`` is such that even-``k`` and
        odd-``k`` filters are interleaved::

            freqs[k-1] + B[k-1] = freqs[k+1] - B[k+1]

        If ``gamma=None`` is specified, then ``gamma`` is computed such
        that each filter has bandwidth proportional to the equivalent
        rectangular bandwidth (ERB) at frequency ``freqs[k]``::

            gamma[k] = 24.7 * alpha[k] / 0.108

        as derived by [#]_.

        .. [#] Glasberg, Brian R., and Brian CJ Moore.
            "Derivation of auditory filter shapes from notched-noise data."
            Hearing research 47.1-2 (1990): 103-138.

    alpha : number > 0 [optional]
        If only one frequency is provided (``len(freqs)==1``), then filter bandwidth
        cannot be computed.  In that case, the ``alpha`` parameter described above
        can be explicitly specified here.

        If two or more frequencies are provided, this parameter is ignored.

    Returns
    -------
    lengths : np.ndarray
        The length of each filter.
    f_cutoff : float
        The lowest frequency at which all filters' main lobes have decayed by
        at least 3dB.

        This second output serves in cqt and vqt to ensure that all wavelet
        bands remain below the Nyquist frequency.

    Notes
    -----
    This function caches at level 10.

    Raises
    ------
    ParameterError
        - If ``filter_scale`` is not strictly positive

        - If ``gamma`` is a negative number

        - If any frequencies are <= 0

        - If the frequency array is not sorted in ascending order
    """
    freqs = np.asarray(freqs)
    if filter_scale <= 0:
        raise ParameterError(f"filter_scale={filter_scale} must be positive")

    if gamma is not None and gamma < 0:
        raise ParameterError(f"gamma={gamma} must be non-negative")

    if np.any(freqs <= 0):
        raise ParameterError("frequencies must be strictly positive")

    if len(freqs) > 1 and np.any(freqs[:-1] > freqs[1:]):
        raise ParameterError(
            f"Frequency array={freqs} must be in strictly ascending order"
        )

    # We need at least 2 frequencies to infer alpha
    if len(freqs) > 1:
        # Approximate the local octave resolution
        bpo = np.empty(len(freqs))
        logf = np.log2(freqs)
        bpo[0] = 1 / (logf[1] - logf[0])
        bpo[-1] = 1 / (logf[-1] - logf[-2])
        bpo[1:-1] = 2 / (logf[2:] - logf[:-2])

        alpha = (2.0 ** (2 / bpo) - 1) / (2.0 ** (2 / bpo) + 1)
    elif alpha is None:
        raise ParameterError(
            "Cannot construct a wavelet basis for a single frequency if alpha is not provided"
        )

    if gamma is None:
        gamma = alpha * 24.7 / 0.108

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / alpha

    # How far up does our highest frequency reach?
    f_cutoff = max(freqs * (1 + 0.5 * window_bandwidth(window) / Q) + 0.5 * gamma)

    # Convert frequencies to filter lengths
    lengths = Q * sr / (freqs + gamma / alpha)

    return lengths, f_cutoff


@deprecate_positional_args
@cache(level=10)
def wavelet(
    *,
    freqs,
    sr=22050,
    window="hann",
    filter_scale=1,
    pad_fft=True,
    norm=1,
    dtype=np.complex64,
    gamma=0,
    alpha=None,
    **kwargs,
):
    """Construct a wavelet basis using windowed complex sinusoids.

    This function constructs a wavelet filterbank at a specified set of center
    frequencies.

    Parameters
    ----------
    freqs : np.ndarray (positive)
        Center frequencies of the filters (in Hz).
        Must be in ascending order.

    sr : number > 0 [scalar]
        Audio sampling rate

    window : string, tuple, number, or function
        Windowing function to apply to filters.

    filter_scale : float > 0 [scalar]
        Scale of filter windows.
        Small values (<1) use shorter windows for higher temporal resolution.

    pad_fft : boolean
        Center-pad all filters up to the nearest integral power of 2.

        By default, padding is done with zeros, but this can be overridden
        by setting the ``mode=`` field in *kwargs*.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See librosa.util.normalize

    gamma : number >= 0
        Bandwidth offset for variable-Q transforms.

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 64-bit (single precision) complex floating point.

    alpha : number > 0 [optional]
        If only one frequency is provided (``len(freqs)==1``), then filter bandwidth
        cannot be computed.  In that case, the ``alpha`` parameter described above
        can be explicitly specified here.

        If two or more frequencies are provided, this parameter is ignored.

    **kwargs : additional keyword arguments
        Arguments to `np.pad()` when ``pad==True``.

    Returns
    -------
    filters : np.ndarray, ``len(filters) == n_bins``
        each ``filters[i]`` is a (complex) time-domain filter
    lengths : np.ndarray, ``len(lengths) == n_bins``
        The (fractional) length of each filter in samples

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    wavelet_lengths
    librosa.cqt
    librosa.vqt
    librosa.util.normalize

    Examples
    --------
    Create a constant-Q basis

    >>> freqs = librosa.cqt_frequencies(n_bins=84, fmin=librosa.note_to_hz('C1'))
    >>> basis, lengths = librosa.filters.wavelet(freqs=freqs, sr=22050)

    Plot one octave of filters in time and frequency

    >>> import matplotlib.pyplot as plt
    >>> basis, lengths = librosa.filters.wavelet(freqs=freqs, sr=22050)
    >>> fig, ax = plt.subplots(nrows=2, figsize=(10, 6))
    >>> notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
    >>> for i, (f, n) in enumerate(zip(basis, notes[:12])):
    ...     f_scale = librosa.util.normalize(f) / 2
    ...     ax[0].plot(i + f_scale.real)
    ...     ax[0].plot(i + f_scale.imag, linestyle=':')
    >>> ax[0].set(yticks=np.arange(len(notes[:12])), yticklabels=notes[:12],
    ...           ylabel='CQ filters',
    ...           title='CQ filters (one octave, time domain)',
    ...           xlabel='Time (samples at 22050 Hz)')
    >>> ax[0].legend(['Real', 'Imaginary'])
    >>> F = np.abs(np.fft.fftn(basis, axes=[-1]))
    >>> # Keep only the positive frequencies
    >>> F = F[:, :(1 + F.shape[1] // 2)]
    >>> librosa.display.specshow(F, x_axis='linear', y_axis='cqt_note', ax=ax[1])
    >>> ax[1].set(ylabel='CQ filters', title='CQ filter magnitudes (frequency domain)')
    """

    # Pass-through parameters to get the filter lengths
    lengths, _ = wavelet_lengths(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
        alpha=alpha,
    )

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = np.exp(
            np.arange(-ilen // 2, ilen // 2, dtype=float) * 1j * 2 * np.pi * freq / sr
        )

        # Apply the windowing function
        sig *= __float_window(window)(len(sig))

        # Normalize
        sig = util.normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0 ** (np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray(
        [util.pad_center(filt, size=max_len, **kwargs) for filt in filters], dtype=dtype
    )

    return filters, lengths


@deprecate_positional_args
@cache(level=10)
def cq_to_chroma(
    n_input,
    *,
    bins_per_octave=12,
    n_chroma=12,
    fmin=None,
    window=None,
    base_c=True,
    dtype=np.float32,
):
    """Construct a linear transformation matrix to map Constant-Q bins
    onto chroma bins (i.e., pitch classes).

    Parameters
    ----------
    n_input : int > 0 [scalar]
        Number of input components (CQT bins)
    bins_per_octave : int > 0 [scalar]
        How many bins per octave in the CQT
    n_chroma : int > 0 [scalar]
        Number of output bins (per octave) in the chroma
    fmin : None or float > 0
        Center frequency of the first constant-Q channel.
        Default: 'C1' ~= 32.7 Hz
    window : None or np.ndarray
        If provided, the cq_to_chroma filter bank will be
        convolved with ``window``.
    base_c : bool
        If True, the first chroma bin will start at 'C'
        If False, the first chroma bin will start at 'A'
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    cq_to_chroma : np.ndarray [shape=(n_chroma, n_input)]
        Transformation matrix: ``Chroma = np.dot(cq_to_chroma, CQT)``

    Raises
    ------
    ParameterError
        If ``n_input`` is not an integer multiple of ``n_chroma``

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Get a CQT, and wrap bins to chroma

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> CQT = np.abs(librosa.cqt(y, sr=sr))
    >>> chroma_map = librosa.filters.cq_to_chroma(CQT.shape[0])
    >>> chromagram = chroma_map.dot(CQT)
    >>> # Max-normalize each time step
    >>> chromagram = librosa.util.normalize(chromagram, axis=0)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> imgcq = librosa.display.specshow(librosa.amplitude_to_db(CQT,
    ...                                                         ref=np.max),
    ...                                  y_axis='cqt_note', x_axis='time',
    ...                                  ax=ax[0])
    >>> ax[0].set(title='CQT Power')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time',
    ...                          ax=ax[1])
    >>> ax[1].set(title='Chroma (wrapped CQT)')
    >>> ax[1].label_outer()
    >>> chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    >>> imgchroma = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[2])
    >>> ax[2].set(title='librosa.feature.chroma_stft')
    """

    # How many fractional bins are we merging?
    n_merge = float(bins_per_octave) / n_chroma

    if fmin is None:
        fmin = note_to_hz("C1")

    if np.mod(n_merge, 1) != 0:
        raise ParameterError(
            "Incompatible CQ merge: "
            "input bins must be an "
            "integer multiple of output bins."
        )

    # Tile the identity to merge fractional bins
    cq_to_ch = np.repeat(np.eye(n_chroma), n_merge, axis=1)

    # Roll it left to center on the target bin
    cq_to_ch = np.roll(cq_to_ch, -int(n_merge // 2), axis=1)

    # How many octaves are we repeating?
    n_octaves = np.ceil(float(n_input) / bins_per_octave)

    # Repeat and trim
    cq_to_ch = np.tile(cq_to_ch, int(n_octaves))[:, :n_input]

    # What's the note number of the first bin in the CQT?
    # midi uses 12 bins per octave here
    midi_0 = np.mod(hz_to_midi(fmin), 12)

    if base_c:
        # rotate to C
        roll = midi_0
    else:
        # rotate to A
        roll = midi_0 - 9

    # Adjust the roll in terms of how many chroma we want out
    # We need to be careful with rounding here
    roll = int(np.round(roll * (n_chroma / 12.0)))

    # Apply the roll
    cq_to_ch = np.roll(cq_to_ch, roll, axis=0).astype(dtype)

    if window is not None:
        cq_to_ch = scipy.signal.convolve(cq_to_ch, np.atleast_2d(window), mode="same")

    return cq_to_ch


@cache(level=10)
def window_bandwidth(window, n=1000):
    """Get the equivalent noise bandwidth of a window function.

    Parameters
    ----------
    window : callable or string
        A window function, or the name of a window function.
        Examples:
        - scipy.signal.hann
        - 'boxcar'
    n : int > 0
        The number of coefficients to use in estimating the
        window bandwidth

    Returns
    -------
    bandwidth : float
        The equivalent noise bandwidth (in FFT bins) of the
        given window function

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    get_window
    """

    if hasattr(window, "__name__"):
        key = window.__name__
    else:
        key = window

    if key not in WINDOW_BANDWIDTHS:
        win = get_window(window, n)
        WINDOW_BANDWIDTHS[key] = n * np.sum(win ** 2) / np.sum(np.abs(win)) ** 2

    return WINDOW_BANDWIDTHS[key]


@deprecate_positional_args
@cache(level=10)
def get_window(window, Nx, *, fftbins=True):
    """Compute a window function.

    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        The window specification:

        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`

    Nx : int > 0
        The length of the window

    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.

    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`

    See Also
    --------
    scipy.signal.get_window

    Notes
    -----
    This function caches at level 10.

    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.
    """
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError(
            "Window size mismatch: " "{:d} != {:d}".format(len(window), Nx)
        )
    else:
        raise ParameterError("Invalid window specification: {}".format(window))


@cache(level=10)
def _multirate_fb(
    center_freqs=None,
    sample_rates=None,
    Q=25.0,
    passband_ripple=1,
    stopband_attenuation=50,
    ftype="ellip",
    flayout="sos",
):
    r"""Helper function to construct a multirate filterbank.

     A filter bank consists of multiple band-pass filters which divide the input signal
     into subbands. In the case of a multirate filter bank, the band-pass filters
     operate with resampled versions of the input signal, e.g. to keep the length
     of a filter constant while shifting its center frequency.

     This implementation uses `scipy.signal.iirdesign` to design the filters.

    Parameters
    ----------
    center_freqs : np.ndarray [shape=(n,), dtype=float]
        Center frequencies of the filter kernels.
        Also defines the number of filters in the filterbank.

    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Samplerate for each filter (used for multirate filterbank).

    Q : float
        Q factor (influences the filter bandwidth).

    passband_ripple : float
        The maximum loss in the passband (dB)
        See `scipy.signal.iirdesign` for details.

    stopband_attenuation : float
        The minimum attenuation in the stopband (dB)
        See `scipy.signal.iirdesign` for details.

    ftype : str
        The type of IIR filter to design
        See `scipy.signal.iirdesign` for details.

    flayout : string
        Valid `output` argument for `scipy.signal.iirdesign`.

        - If `ba`, returns numerators/denominators of the transfer functions,
          used for filtering with `scipy.signal.filtfilt`.
          Can be unstable for high-order filters.

        - If `sos`, returns a series of second-order filters,
          used for filtering with `scipy.signal.sosfiltfilt`.
          Minimizes numerical precision errors for high-order filters, but is slower.

        - If `zpk`, returns zeros, poles, and system gains of the transfer functions.

    Returns
    -------
    filterbank : list [shape=(n,), dtype=float]
        Each list entry comprises the filter coefficients for a single filter.
    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Samplerate for each filter.

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    scipy.signal.iirdesign

    Raises
    ------
    ParameterError
        If ``center_freqs`` is ``None``.
        If ``sample_rates`` is ``None``.
        If ``center_freqs.shape`` does not match ``sample_rates.shape``.
    """

    if center_freqs is None:
        raise ParameterError("center_freqs must be provided.")

    if sample_rates is None:
        raise ParameterError("sample_rates must be provided.")

    if center_freqs.shape != sample_rates.shape:
        raise ParameterError(
            "Number of provided center_freqs and sample_rates must be equal."
        )

    nyquist = 0.5 * sample_rates
    filter_bandwidths = center_freqs / float(Q)

    filterbank = []

    for cur_center_freq, cur_nyquist, cur_bw in zip(
        center_freqs, nyquist, filter_bandwidths
    ):
        passband_freqs = [
            cur_center_freq - 0.5 * cur_bw,
            cur_center_freq + 0.5 * cur_bw,
        ] / cur_nyquist
        stopband_freqs = [
            cur_center_freq - cur_bw,
            cur_center_freq + cur_bw,
        ] / cur_nyquist

        cur_filter = scipy.signal.iirdesign(
            passband_freqs,
            stopband_freqs,
            passband_ripple,
            stopband_attenuation,
            analog=False,
            ftype=ftype,
            output=flayout,
        )

        filterbank.append(cur_filter)

    return filterbank, sample_rates


@cache(level=10)
def mr_frequencies(tuning):
    r"""Helper function for generating center frequency and sample rate pairs.

    This function will return center frequency and corresponding sample rates
    to obtain similar pitch filterbank settings as described in [#]_.
    Instead of starting with MIDI pitch `A0`, we start with `C0`.

    .. [#] Müller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.

    Parameters
    ----------
    tuning : float [scalar]
        Tuning deviation from A440, measure as a fraction of the equally
        tempered semitone (1/12 of an octave).

    Returns
    -------
    center_freqs : np.ndarray [shape=(n,), dtype=float]
        Center frequencies of the filter kernels.
        Also defines the number of filters in the filterbank.
    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Sample rate for each filter, used for multirate filterbank.

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    librosa.filters.semitone_filterbank
    """

    center_freqs = midi_to_hz(np.arange(24 + tuning, 109 + tuning))

    sample_rates = np.asarray(
        len(np.arange(0, 36))
        * [
            882,
        ]
        + len(np.arange(36, 70))
        * [
            4410,
        ]
        + len(np.arange(70, 85))
        * [
            22050,
        ]
    )

    return center_freqs, sample_rates


@deprecate_positional_args
def semitone_filterbank(
    *, center_freqs=None, tuning=0.0, sample_rates=None, flayout="ba", **kwargs
):
    r"""Construct a multi-rate bank of infinite-impulse response (IIR)
    band-pass filters at user-defined center frequencies and sample rates.

    By default, these center frequencies are set equal to the 88 fundamental
    frequencies of the grand piano keyboard, according to a pitch tuning standard
    of A440, that is, note A above middle C set to 440 Hz. The center frequencies
    are tuned to the twelve-tone equal temperament, which means that they grow
    exponentially at a rate of 2**(1/12), that is, twelve notes per octave.

    The A440 tuning can be changed by the user while keeping twelve-tone equal
    temperament. While A440 is currently the international standard in the music
    industry (ISO 16), some orchestras tune to A441-A445, whereas baroque musicians
    tune to A415.

    See [#]_ for details.

    .. [#] Müller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.

    Parameters
    ----------
    center_freqs : np.ndarray [shape=(n,), dtype=float]
        Center frequencies of the filter kernels.
        Also defines the number of filters in the filterbank.
    tuning : float [scalar]
        Tuning deviation from A440 as a fraction of a semitone (1/12 of an octave
        in equal temperament).
    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Sample rates of each filter in the multirate filterbank.
    flayout : string
        - If `ba`, the standard difference equation is used for filtering with `scipy.signal.filtfilt`.
          Can be unstable for high-order filters.
        - If `sos`, a series of second-order filters is used for filtering with `scipy.signal.sosfiltfilt`.
          Minimizes numerical precision errors for high-order filters, but is slower.
    **kwargs : additional keyword arguments
        Additional arguments to the private function `_multirate_fb()`.

    Returns
    -------
    filterbank : list [shape=(n,), dtype=float]
        Each list entry contains the filter coefficients for a single filter.
    fb_sample_rates : np.ndarray [shape=(n,), dtype=float]
        Sample rate for each filter.

    See Also
    --------
    librosa.cqt
    librosa.iirt
    librosa.filters.mr_frequencies
    scipy.signal.iirdesign

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import scipy.signal
    >>> semitone_filterbank, sample_rates = librosa.filters.semitone_filterbank()
    >>> fig, ax = plt.subplots()
    >>> for cur_sr, cur_filter in zip(sample_rates, semitone_filterbank):
    ...    w, h = scipy.signal.freqz(cur_filter[0], cur_filter[1], worN=2000)
    ...    ax.semilogx((cur_sr / (2 * np.pi)) * w, 20 * np.log10(abs(h)))
    >>> ax.set(xlim=[20, 10e3], ylim=[-60, 3], title='Magnitude Responses of the Pitch Filterbank',
    ...        xlabel='Log-Frequency (Hz)', ylabel='Magnitude (dB)')
    """

    if (center_freqs is None) and (sample_rates is None):
        center_freqs, sample_rates = mr_frequencies(tuning)

    filterbank, fb_sample_rates = _multirate_fb(
        center_freqs=center_freqs, sample_rates=sample_rates, flayout=flayout, **kwargs
    )

    return filterbank, fb_sample_rates


@jit(nopython=True, cache=True)
def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    """Helper function for window sum-square calculation."""

    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]


@deprecate_positional_args
def window_sumsquare(
    *,
    window,
    n_frames,
    hop_length=512,
    win_length=None,
    n_fft=2048,
    dtype=np.float32,
    norm=None,
):
    """Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing observations
    in short-time Fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches ``n_fft``.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    norm : {np.inf, -np.inf, 0, float > 0, None}
        Normalization mode used in window construction.
        Note that this does not affect the squaring operation.

    Returns
    -------
    wss : np.ndarray, shape=``(n_fft + hop_length * (n_frames - 1))``
        The sum-squared envelope of the window function

    Examples
    --------
    For a fixed frame length (2048), compare modulation effects for a Hann window
    at different hop lengths:

    >>> n_frames = 50
    >>> wss_256 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=256)
    >>> wss_512 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=512)
    >>> wss_1024 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=1024)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharey=True)
    >>> ax[0].plot(wss_256)
    >>> ax[0].set(title='hop_length=256')
    >>> ax[1].plot(wss_512)
    >>> ax[1].set(title='hop_length=512')
    >>> ax[2].plot(wss_1024)
    >>> ax[2].set(title='hop_length=1024')
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = util.normalize(win_sq, norm=norm) ** 2
    win_sq = util.pad_center(win_sq, size=n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x


@deprecate_positional_args
@cache(level=10)
def diagonal_filter(window, n, *, slope=1.0, angle=None, zero_mean=False):
    """Build a two-dimensional diagonal filter.

    This is primarily used for smoothing recurrence or self-similarity matrices.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        The window function to use for the filter.

        See `get_window` for details.

        Note that the window used here should be non-negative.

    n : int > 0
        the length of the filter

    slope : float
        The slope of the diagonal filter to produce

    angle : float or None
        If given, the slope parameter is ignored,
        and angle directly sets the orientation of the filter (in radians).
        Otherwise, angle is inferred as `arctan(slope)`.

    zero_mean : bool
        If True, a zero-mean filter is used.
        Otherwise, a non-negative averaging filter is used.

        This should be enabled if you want to enhance paths and suppress
        blocks.

    Returns
    -------
    kernel : np.ndarray, shape=[(m, m)]
        The 2-dimensional filter kernel

    Notes
    -----
    This function caches at level 10.
    """

    if angle is None:
        angle = np.arctan(slope)

    win = np.diag(get_window(window, n, fftbins=False))

    if not np.isclose(angle, np.pi / 4):
        win = scipy.ndimage.rotate(
            win, 45 - angle * 180 / np.pi, order=5, prefilter=False
        )

    np.clip(win, 0, None, out=win)
    win /= win.sum()

    if zero_mean:
        win -= win.mean()

    return win
