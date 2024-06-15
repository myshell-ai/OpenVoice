#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pitch-tracking and tuning estimation"""

import warnings
import numpy as np
import scipy


from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from ..util.decorators import deprecate_positional_args

__all__ = ["estimate_tuning", "pitch_tuning", "piptrack", "yin", "pyin"]


@deprecate_positional_args
def estimate_tuning(
    *,
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    resolution=0.01,
    bins_per_octave=12,
    **kwargs,
):
    """Estimate the tuning of an audio time series or spectrogram input.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio signal. Multi-channel is supported..
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        magnitude or power spectrogram
    n_fft : int > 0 [scalar] or None
        number of FFT bins to use, if ``y`` is provided.
    resolution : float in `(0, 1)`
        Resolution of the tuning as a fraction of a bin.
        0.01 corresponds to measurements in cents.
    bins_per_octave : int > 0 [scalar]
        How many frequency bins per octave
    **kwargs : additional keyword arguments
        Additional arguments passed to `piptrack`

    Returns
    -------
    tuning: float in `[-0.5, 0.5)`
        estimated tuning deviation (fractions of a bin).

        Note that if multichannel input is provided, a single tuning estimate is provided spanning all
        channels.

    See Also
    --------
    piptrack : Pitch tracking by parabolic interpolation

    Examples
    --------
    With time-series input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.estimate_tuning(y=y, sr=sr)
    -0.08000000000000002

    In tenths of a cent

    >>> librosa.estimate_tuning(y=y, sr=sr, resolution=1e-3)
    -0.016000000000000014

    Using spectrogram input

    >>> S = np.abs(librosa.stft(y))
    >>> librosa.estimate_tuning(S=S, sr=sr)
    -0.08000000000000002

    Using pass-through arguments to `librosa.piptrack`

    >>> librosa.estimate_tuning(y=y, sr=sr, n_fft=8192,
    ...                         fmax=librosa.note_to_hz('G#9'))
    -0.08000000000000002
    """

    pitch, mag = piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kwargs)

    # Only count magnitude where frequency is > 0
    pitch_mask = pitch > 0

    if pitch_mask.any():
        threshold = np.median(mag[pitch_mask])
    else:
        threshold = 0.0

    return pitch_tuning(
        pitch[(mag >= threshold) & pitch_mask],
        resolution=resolution,
        bins_per_octave=bins_per_octave,
    )


@deprecate_positional_args
def pitch_tuning(frequencies, *, resolution=0.01, bins_per_octave=12):
    """Given a collection of pitches, estimate its tuning offset
    (in fractions of a bin) relative to A440=440.0Hz.

    Parameters
    ----------
    frequencies : array-like, float
        A collection of frequencies detected in the signal.
        See `piptrack`
    resolution : float in `(0, 1)`
        Resolution of the tuning as a fraction of a bin.
        0.01 corresponds to cents.
    bins_per_octave : int > 0 [scalar]
        How many frequency bins per octave

    Returns
    -------
    tuning: float in `[-0.5, 0.5)`
        estimated tuning deviation (fractions of a bin)

    See Also
    --------
    estimate_tuning : Estimating tuning from time-series or spectrogram input

    Examples
    --------
    >>> # Generate notes at +25 cents
    >>> freqs = librosa.cqt_frequencies(n_bins=24, fmin=55, tuning=0.25)
    >>> librosa.pitch_tuning(freqs)
    0.25

    >>> # Track frequencies from a real spectrogram
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> freqs, times, mags = librosa.reassigned_spectrogram(y, sr=sr,
    ...                                                     fill_nan=True)
    >>> # Select out pitches with high energy
    >>> freqs = freqs[mags > np.median(mags)]
    >>> librosa.pitch_tuning(freqs)
    -0.07

    """

    frequencies = np.atleast_1d(frequencies)

    # Trim out any DC components
    frequencies = frequencies[frequencies > 0]

    if not np.any(frequencies):
        warnings.warn(
            "Trying to estimate tuning from empty frequency set.", stacklevel=2
        )
        return 0.0

    # Compute the residual relative to the number of bins
    residual = np.mod(bins_per_octave * convert.hz_to_octs(frequencies), 1.0)

    # Are we on the wrong side of the semitone?
    # A residual of 0.95 is more likely to be a deviation of -0.05
    # from the next tone up.
    residual[residual >= 0.5] -= 1.0

    bins = np.linspace(-0.5, 0.5, int(np.ceil(1.0 / resolution)) + 1)

    counts, tuning = np.histogram(residual, bins)

    # return the histogram peak
    return tuning[np.argmax(counts)]


@deprecate_positional_args
@cache(level=30)
def piptrack(
    *,
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=None,
    fmin=150.0,
    fmax=4000.0,
    threshold=0.1,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant",
    ref=None,
):
    """Pitch tracking on thresholded parabolically-interpolated STFT.

    This implementation uses the parabolic interpolation method described by [#]_.

    .. [#] https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio signal. Multi-channel is supported..

    sr : number > 0 [scalar]
        audio sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        magnitude or power spectrogram

    n_fft : int > 0 [scalar] or None
        number of FFT bins to use, if ``y`` is provided.

    hop_length : int > 0 [scalar] or None
        number of samples to hop

    threshold : float in `(0, 1)`
        A bin in spectrum ``S`` is considered a pitch when it is greater than
        ``threshold * ref(S)``.

        By default, ``ref(S)`` is taken to be ``max(S, axis=0)`` (the maximum value in
        each column).

    fmin : float > 0 [scalar]
        lower frequency cutoff.

    fmax : float > 0 [scalar]
        upper frequency cutoff.

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero-padding.

        See also: `np.pad`.

    ref : scalar or callable [default=np.max]
        If scalar, the reference value against which ``S`` is compared for determining
        pitches.

        If callable, the reference value is computed as ``ref(S, axis=0)``.

    Returns
    -------
    pitches, magnitudes : np.ndarray [shape=(..., d, t)]
        Where ``d`` is the subset of FFT bins within ``fmin`` and ``fmax``.

        ``pitches[..., f, t]`` contains instantaneous frequency at bin
        ``f``, time ``t``

        ``magnitudes[..., f, t]`` contains the corresponding magnitudes.

        Both ``pitches`` and ``magnitudes`` take value 0 at bins
        of non-maximal magnitude.

    Notes
    -----
    This function caches at level 30.

    One of ``S`` or ``y`` must be provided.
    If ``S`` is not given, it is computed from ``y`` using
    the default parameters of `librosa.stft`.

    Examples
    --------
    Computing pitches from a waveform input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    Or from a spectrogram input

    >>> S = np.abs(librosa.stft(y))
    >>> pitches, magnitudes = librosa.piptrack(S=S, sr=sr)

    Or with an alternate reference value for pitch detection, where
    values above the mean spectral energy in each frame are counted as pitches

    >>> pitches, magnitudes = librosa.piptrack(S=S, sr=sr, threshold=1,
    ...                                        ref=np.mean)

    """

    # Check that we received an audio time series or STFT
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Make sure we're dealing with magnitudes
    S = np.abs(S)

    # Truncate to feasible region
    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)

    fft_freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = 0.5 * (S[..., 2:, :] - S[..., :-2, :])

    shift = 2 * S[..., 1:-1, :] - S[..., 2:, :] - S[..., :-2, :]

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift = avg / (shift + (np.abs(shift) < util.tiny(shift)))

    # Pad back up to the same shape as S
    padding = [(0, 0) for _ in S.shape]
    padding[-2] = (1, 1)
    avg = np.pad(avg, padding, mode="constant")
    shift = np.pad(shift, padding, mode="constant")

    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)

    # Clip to the viable frequency range
    freq_mask = (fmin <= fft_freqs) & (fft_freqs < fmax)
    freq_mask = util.expand_to(freq_mask, ndim=S.ndim, axes=-2)

    # Compute the column-wise local max of S after thresholding
    # Find the argmax coordinates
    if ref is None:
        ref = np.max

    if callable(ref):
        ref_value = threshold * ref(S, axis=-2)
        # Reinsert the frequency axis here, in case the callable doesn't
        # support keepdims=True
        ref_value = np.expand_dims(ref_value, -2)
    else:
        ref_value = np.abs(ref)

    # Store pitch and magnitude
    idx = np.nonzero(freq_mask & util.localmax(S * (S > ref_value), axis=-2))
    pitches[idx] = (idx[-2] + shift[idx]) * float(sr) / n_fft
    mags[idx] = S[idx] + dskew[idx]

    return pitches, mags


def _cumulative_mean_normalized_difference(
    y_frames, frame_length, win_length, min_period, max_period
):
    """Cumulative mean normalized difference function (equation 8 in [#]_)

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.
    frame_length : int > 0 [scalar]
        length of the frames in samples.
    win_length : int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
    min_period : int > 0 [scalar]
        minimum period.
    max_period : int > 0 [scalar]
        maximum period.

    Returns
    -------
    yin_frames : np.ndarray [shape=(max_period-min_period+1,n_frames)]
        Cumulative mean normalized difference function for each frame.
    """
    # Autocorrelation.
    a = np.fft.rfft(y_frames, frame_length, axis=-2)
    b = np.fft.rfft(y_frames[..., win_length:0:-1, :], frame_length, axis=-2)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=-2)[..., win_length:, :]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0

    # Energy terms.
    energy_frames = np.cumsum(y_frames ** 2, axis=-2)
    energy_frames = (
        energy_frames[..., win_length:, :] - energy_frames[..., :-win_length, :]
    )
    energy_frames[np.abs(energy_frames) < 1e-6] = 0

    # Difference function.
    yin_frames = energy_frames[..., :1, :] + energy_frames - 2 * acf_frames

    # Cumulative mean normalized difference function.
    yin_numerator = yin_frames[..., min_period : max_period + 1, :]
    # broadcast this shape to have leading ones
    tau_range = util.expand_to(
        np.arange(1, max_period + 1), ndim=yin_frames.ndim, axes=-2
    )

    cumulative_mean = (
        np.cumsum(yin_frames[..., 1 : max_period + 1, :], axis=-2) / tau_range
    )
    yin_denominator = cumulative_mean[..., min_period - 1 : max_period, :]
    yin_frames = yin_numerator / (yin_denominator + util.tiny(yin_denominator))
    return yin_frames


def _parabolic_interpolation(y_frames):
    """Piecewise parabolic interpolation for yin and pyin.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.

    Returns
    -------
    parabolic_shifts : np.ndarray [shape=(frame_length, n_frames)]
        position of the parabola optima
    """

    parabolic_shifts = np.zeros_like(y_frames)
    parabola_a = (
        y_frames[..., :-2, :] + y_frames[..., 2:, :] - 2 * y_frames[..., 1:-1, :]
    ) / 2
    parabola_b = (y_frames[..., 2:, :] - y_frames[..., :-2, :]) / 2
    parabolic_shifts[..., 1:-1, :] = -parabola_b / (
        2 * parabola_a + util.tiny(parabola_a)
    )
    parabolic_shifts[np.abs(parabolic_shifts) > 1] = 0
    return parabolic_shifts


@deprecate_positional_args
def yin(
    y,
    *,
    fmin,
    fmax,
    sr=22050,
    frame_length=2048,
    win_length=None,
    hop_length=None,
    trough_threshold=0.1,
    center=True,
    pad_mode="constant",
):
    """Fundamental frequency (F0) estimation using the YIN algorithm.

    YIN is an autocorrelation based method for fundamental frequency estimation [#]_.
    First, a normalized difference function is computed over short (overlapping) frames of audio.
    Next, the first minimum in the difference function below ``trough_threshold`` is selected as
    an estimate of the signal's period.
    Finally, the estimated period is refined using parabolic interpolation before converting
    into the corresponding frequency.

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported..
    fmin : number > 0 [scalar]
        minimum frequency in Hertz.
        The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
        though lower values may be feasible.
    fmax : number > 0 [scalar]
        maximum frequency in Hertz.
        The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
        though higher values may be feasible.
    sr : number > 0 [scalar]
        sampling rate of ``y`` in Hertz.
    frame_length : int > 0 [scalar]
        length of the frames in samples.
        By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
        a sampling rate of 22050 Hz.
    win_length : None or int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
        If ``None``, defaults to ``frame_length // 2``
    hop_length : None or int > 0 [scalar]
        number of audio samples between adjacent YIN predictions.
        If ``None``, defaults to ``frame_length // 4``.
    trough_threshold : number > 0 [scalar]
        absolute threshold for peak estimation.
    center : boolean
        If ``True``, the signal `y` is padded so that frame
        ``D[:, t]`` is centered at `y[t * hop_length]`.
        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of ``librosa.core.frames_to_samples``.
    pad_mode : string or function
        If ``center=True``, this argument is passed to ``np.pad`` for padding
        the edges of the signal ``y``. By default (``pad_mode="constant"``),
        ``y`` is padded on both sides with zeros.
        If ``center=False``,  this argument is ignored.
        .. see also:: `np.pad`

    Returns
    -------
    f0: np.ndarray [shape=(..., n_frames)]
        time series of fundamental frequencies in Hertz.

        If multi-channel input is provided, f0 curves are estimated separately for each channel.

    See Also
    --------
    librosa.pyin :
        Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).

    Examples
    --------
    Computing a fundamental frequency (F0) curve from an audio input

    >>> y = librosa.chirp(fmin=440, fmax=880, duration=5.0)
    >>> librosa.yin(y, fmin=440, fmax=880)
    array([442.66354675, 441.95299983, 441.58010963, ...,
        871.161732  , 873.99001454, 877.04297681])
    """

    if fmin is None or fmax is None:
        raise ParameterError('both "fmin" and "fmax" must be provided')

    # Set the default window length if it is not already specified.
    if win_length is None:
        win_length = frame_length // 2

    if win_length >= frame_length:
        raise ParameterError(
            "win_length={} cannot exceed given frame_length={}".format(
                win_length, frame_length
            )
        )

    # Set the default hop if it is not already specified.
    if hop_length is None:
        hop_length = frame_length // 4

    # Check that audio is valid.
    util.valid_audio(y, mono=False)

    # Pad the time series so that frames are centered
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (frame_length // 2, frame_length // 2)
        y = np.pad(y, padding, mode=pad_mode)

    # Frame audio.
    y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sr / fmax)), 1)
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period
    )

    # Parabolic interpolation.
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find local minima.
    is_trough = util.localmin(yin_frames, axis=-2)
    is_trough[..., 0, :] = yin_frames[..., 0, :] < yin_frames[..., 1, :]

    # Find minima below peak threshold.
    is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)

    # Absolute threshold.
    # "The solution we propose is to set an absolute threshold and choose the
    # smallest value of tau that gives a minimum of d' deeper than
    # this threshold. If none is found, the global minimum is chosen instead."
    target_shape = list(yin_frames.shape)
    target_shape[-2] = 1

    global_min = np.argmin(yin_frames, axis=-2)
    yin_period = np.argmax(is_threshold_trough, axis=-2)

    global_min = global_min.reshape(target_shape)
    yin_period = yin_period.reshape(target_shape)

    no_trough_below_threshold = np.all(~is_threshold_trough, axis=-2, keepdims=True)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

    # Refine peak by parabolic interpolation.

    yin_period = (
        min_period
        + yin_period
        + np.take_along_axis(parabolic_shifts, yin_period, axis=-2)
    )[..., 0, :]

    # Convert period to fundamental frequency.
    f0 = sr / yin_period
    return f0


@deprecate_positional_args
def pyin(
    y,
    *,
    fmin,
    fmax,
    sr=22050,
    frame_length=2048,
    win_length=None,
    hop_length=None,
    n_thresholds=100,
    beta_parameters=(2, 18),
    boltzmann_parameter=2,
    resolution=0.1,
    max_transition_rate=35.92,
    switch_prob=0.01,
    no_trough_prob=0.01,
    fill_na=np.nan,
    center=True,
    pad_mode="constant",
):
    """Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).

    pYIN [#]_ is a modificatin of the YIN algorithm [#]_ for fundamental frequency (F0) estimation.
    In the first step of pYIN, F0 candidates and their probabilities are computed using the YIN algorithm.
    In the second step, Viterbi decoding is used to estimate the most likely F0 sequence and voicing flags.

    .. [#] Mauch, Matthias, and Simon Dixon.
        "pYIN: A fundamental frequency estimator using probabilistic threshold distributions."
        2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    fmin : number > 0 [scalar]
        minimum frequency in Hertz.
        The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
        though lower values may be feasible.
    fmax : number > 0 [scalar]
        maximum frequency in Hertz.
        The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
        though higher values may be feasible.
    sr : number > 0 [scalar]
        sampling rate of ``y`` in Hertz.
    frame_length : int > 0 [scalar]
        length of the frames in samples.
        By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
        a sampling rate of 22050 Hz.
    win_length : None or int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
        If ``None``, defaults to ``frame_length // 2``
    hop_length : None or int > 0 [scalar]
        number of audio samples between adjacent pYIN predictions.
        If ``None``, defaults to ``frame_length // 4``.
    n_thresholds : int > 0 [scalar]
        number of thresholds for peak estimation.
    beta_parameters : tuple
        shape parameters for the beta distribution prior over thresholds.
    boltzmann_parameter : number > 0 [scalar]
        shape parameter for the Boltzmann distribution prior over troughs.
        Larger values will assign more mass to smaller periods.
    resolution : float in `(0, 1)`
        Resolution of the pitch bins.
        0.01 corresponds to cents.
    max_transition_rate : float > 0
        maximum pitch transition rate in octaves per second.
    switch_prob : float in ``(0, 1)``
        probability of switching from voiced to unvoiced or vice versa.
    no_trough_prob : float in ``(0, 1)``
        maximum probability to add to global minimum if no trough is below threshold.
    fill_na : None, float, or ``np.nan``
        default value for unvoiced frames of ``f0``.
        If ``None``, the unvoiced frames will contain a best guess value.
    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.
        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of ``librosa.core.frames_to_samples``.
    pad_mode : string or function
        If ``center=True``, this argument is passed to ``np.pad`` for padding
        the edges of the signal ``y``. By default (``pad_mode="constant"``),
        ``y`` is padded on both sides with zeros.
        If ``center=False``,  this argument is ignored.
        .. see also:: `np.pad`

    Returns
    -------
    f0: np.ndarray [shape=(..., n_frames)]
        time series of fundamental frequencies in Hertz.
    voiced_flag: np.ndarray [shape=(..., n_frames)]
        time series containing boolean flags indicating whether a frame is voiced or not.
    voiced_prob: np.ndarray [shape=(..., n_frames)]
        time series containing the probability that a frame is voiced.
    .. note:: If multi-channel input is provided, f0 and voicing are estimated separately for each channel.

    See Also
    --------
    librosa.yin :
        Fundamental frequency (F0) estimation using the YIN algorithm.

    Examples
    --------
    Computing a fundamental frequency (F0) curve from an audio input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> f0, voiced_flag, voiced_probs = librosa.pyin(y,
    ...                                              fmin=librosa.note_to_hz('C2'),
    ...                                              fmax=librosa.note_to_hz('C7'))
    >>> times = librosa.times_like(f0)

    Overlay F0 over a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    >>> ax.set(title='pYIN fundamental frequency estimation')
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    >>> ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    >>> ax.legend(loc='upper right')
    """

    if fmin is None or fmax is None:
        raise ParameterError('both "fmin" and "fmax" must be provided')

    # Set the default window length if it is not already specified.
    if win_length is None:
        win_length = frame_length // 2

    if win_length >= frame_length:
        raise ParameterError(
            "win_length={} cannot exceed given frame_length={}".format(
                win_length, frame_length
            )
        )

    # Set the default hop if it is not already specified.
    if hop_length is None:
        hop_length = frame_length // 4

    # Check that audio is valid.
    util.valid_audio(y, mono=False)

    # Pad the time series so that frames are centered
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (frame_length // 2, frame_length // 2)
        y = np.pad(y, padding, mode=pad_mode)

    # Frame audio.
    y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sr / fmax)), 1)
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period
    )

    # Parabolic interpolation.
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find Yin candidates and probabilities.
    # The implementation here follows the official pYIN software which
    # differs from the method described in the paper.
    # 1. Define the prior over the thresholds.
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)

    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    def _helper(a, b):
        return __pyin_helper(
            a,
            b,
            sr,
            thresholds,
            boltzmann_parameter,
            beta_probs,
            no_trough_prob,
            min_period,
            fmin,
            n_pitch_bins,
            n_bins_per_semitone,
        )

    helper = np.vectorize(_helper, signature="(f,t),(k,t)->(1,d,t),(j,t)")
    observation_probs, voiced_prob = helper(yin_frames, parabolic_shifts)

    # Construct transition matrix.
    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    # Construct the within voicing transition probabilities
    transition = sequence.transition_local(
        n_pitch_bins, transition_width, window="triangle", wrap=False
    )

    # Include across voicing transition probabilities
    t_switch = sequence.transition_loop(2, 1 - switch_prob)
    transition = np.kron(t_switch, transition)

    p_init = np.zeros(2 * n_pitch_bins)
    p_init[n_pitch_bins:] = 1 / n_pitch_bins

    states = sequence.viterbi(observation_probs, transition, p_init=p_init)

    # Find f0 corresponding to each decoded pitch bin.
    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins

    if fill_na is not None:
        f0[~voiced_flag] = fill_na

    return f0[..., 0, :], voiced_flag[..., 0, :], voiced_prob[..., 0, :]


def __pyin_helper(
    yin_frames,
    parabolic_shifts,
    sr,
    thresholds,
    boltzmann_parameter,
    beta_probs,
    no_trough_prob,
    min_period,
    fmin,
    n_pitch_bins,
    n_bins_per_semitone,
):

    yin_probs = np.zeros_like(yin_frames)

    for i, yin_frame in enumerate(yin_frames.T):
        # 2. For each frame find the troughs.
        is_trough = util.localmin(yin_frame)

        is_trough[0] = yin_frame[0] < yin_frame[1]
        (trough_index,) = np.nonzero(is_trough)

        if len(trough_index) == 0:
            continue

        # 3. Find the troughs below each threshold.
        # these are the local minima of the frame, could get them directly without the trough index
        trough_heights = yin_frame[trough_index]
        trough_thresholds = np.less.outer(trough_heights, thresholds[1:])

        # 4. Define the prior over the troughs.
        # Smaller periods are weighted more.
        trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)

        trough_prior = scipy.stats.boltzmann.pmf(
            trough_positions, boltzmann_parameter, n_troughs
        )

        trough_prior[~trough_thresholds] = 0

        # 5. For each threshold add probability to global minimum if no trough is below threshold,
        # else add probability to each trough below threshold biased by prior.

        probs = trough_prior.dot(beta_probs)

        global_min = np.argmin(trough_heights)
        n_thresholds_below_min = np.count_nonzero(~trough_thresholds[global_min, :])
        probs[global_min] += no_trough_prob * np.sum(
            beta_probs[:n_thresholds_below_min]
        )

        yin_probs[trough_index, i] = probs

    yin_period, frame_index = np.nonzero(yin_probs)

    # Refine peak by parabolic interpolation.
    period_candidates = min_period + yin_period
    period_candidates = period_candidates + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates

    # Find pitch bin corresponding to each f0 candidate.
    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)

    # Observation probabilities.
    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]

    voiced_prob = np.clip(
        np.sum(observation_probs[:n_pitch_bins, :], axis=0, keepdims=True), 0, 1
    )
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob) / n_pitch_bins

    return observation_probs[np.newaxis], voiced_prob
