#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Beat and tempo
==============
.. autosummary::
   :toctree: generated/

   beat_track
   plp
   tempo
"""

import numpy as np
import scipy
import scipy.stats

from ._cache import cache
from . import core
from . import onset
from . import util
from .feature import tempogram, fourier_tempogram
from .util.exceptions import ParameterError
from .util.decorators import deprecate_positional_args

__all__ = ["beat_track", "tempo", "plp"]


@deprecate_positional_args
def beat_track(
    *,
    y=None,
    sr=22050,
    onset_envelope=None,
    hop_length=512,
    start_bpm=120.0,
    tightness=100,
    trim=True,
    bpm=None,
    prior=None,
    units="frames",
):
    r"""Dynamic programming beat tracker.

    Beats are detected in three stages, following the method of [#]_:

      1. Measure onset strength
      2. Estimate tempo from onset correlation
      3. Pick peaks in onset strength approximately consistent with estimated
         tempo

    .. [#] Ellis, Daniel PW. "Beat tracking by dynamic programming."
           Journal of New Music Research 36.1 (2007): 51-60.
           http://labrosa.ee.columbia.edu/projects/beattrack/

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series
    sr : number > 0 [scalar]
        sampling rate of ``y``
    onset_envelope : np.ndarray [shape=(n,)] or None
        (optional) pre-computed onset strength envelope.
    hop_length : int > 0 [scalar]
        number of audio samples between successive ``onset_envelope`` values
    start_bpm : float > 0 [scalar]
        initial guess for the tempo estimator (in beats per minute)
    tightness : float [scalar]
        tightness of beat distribution around tempo
    trim : bool [scalar]
        trim leading/trailing beats with weak onsets
    bpm : float [scalar]
        (optional) If provided, use ``bpm`` as the tempo instead of
        estimating it from ``onsets``.
    prior : scipy.stats.rv_continuous [optional]
        An optional prior distribution over tempo.
        If provided, ``start_bpm`` will be ignored.
    units : {'frames', 'samples', 'time'}
        The units to encode detected beat events in.
        By default, 'frames' are used.

    Returns
    -------
    tempo : float [scalar, non-negative]
        estimated global tempo (in beats per minute)
    beats : np.ndarray [shape=(m,)]
        estimated beat event locations in the specified units
        (default is frame indices)
    .. note::
        If no onset strength could be detected, beat_tracker estimates 0 BPM
        and returns an empty list.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onset_envelope`` are provided,
        or if ``units`` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    librosa.onset.onset_strength

    Examples
    --------
    Track beats using time series input

    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)

    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> tempo
    135.99917763157896

    Print the frames corresponding to beats

    >>> beats
    array([  3,  21,  40,  59,  78,  96, 116, 135, 154, 173, 192, 211,
           230, 249, 268, 287, 306, 325, 344, 363])

    Or print them as timestamps

    >>> librosa.frames_to_time(beats, sr=sr)
    array([0.07 , 0.488, 0.929, 1.37 , 1.811, 2.229, 2.694, 3.135,
           3.576, 4.017, 4.458, 4.899, 5.341, 5.782, 6.223, 6.664,
           7.105, 7.546, 7.988, 8.429])

    Track beats using a pre-computed onset envelope

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          aggregate=np.median)
    >>> tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,
    ...                                        sr=sr)
    >>> tempo
    135.99917763157896
    >>> beats
    array([  3,  21,  40,  59,  78,  96, 116, 135, 154, 173, 192, 211,
           230, 249, 268, 287, 306, 325, 344, 363])

    Plot the beat events against the onset strength envelope

    >>> import matplotlib.pyplot as plt
    >>> hop_length = 512
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    >>> M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    >>> librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
    ...                          y_axis='mel', x_axis='time', hop_length=hop_length,
    ...                          ax=ax[0])
    >>> ax[0].label_outer()
    >>> ax[0].set(title='Mel spectrogram')
    >>> ax[1].plot(times, librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='Beats')
    >>> ax[1].legend()
    """

    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError("y or onset_envelope must be provided")

        onset_envelope = onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length, aggregate=np.median
        )

    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return (0, np.array([], dtype=int))

    # Estimate BPM if one was not provided
    if bpm is None:
        bpm = tempo(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=hop_length,
            start_bpm=start_bpm,
            prior=prior,
        )[0]

    # Then, run the tracker
    beats = __beat_tracker(onset_envelope, bpm, float(sr) / hop_length, tightness, trim)

    if units == "frames":
        pass
    elif units == "samples":
        beats = core.frames_to_samples(beats, hop_length=hop_length)
    elif units == "time":
        beats = core.frames_to_time(beats, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError("Invalid unit type: {}".format(units))

    return (bpm, beats)


@cache(level=30)
@deprecate_positional_args
def tempo(
    *,
    y=None,
    sr=22050,
    onset_envelope=None,
    hop_length=512,
    start_bpm=120,
    std_bpm=1.0,
    ac_size=8.0,
    max_tempo=320.0,
    aggregate=np.mean,
    prior=None,
):
    """Estimate the tempo (beats per minute)

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of the time series
    onset_envelope : np.ndarray [shape=(..., n)]
        pre-computed onset strength envelope
    hop_length : int > 0 [scalar]
        hop length of the time series
    start_bpm : float [scalar]
        initial guess of the BPM
    std_bpm : float > 0 [scalar]
        standard deviation of tempo distribution
    ac_size : float > 0 [scalar]
        length (in seconds) of the auto-correlation window
    max_tempo : float > 0 [scalar, optional]
        If provided, only estimate tempo below this threshold
    aggregate : callable [optional]
        Aggregation function for estimating global tempo.
        If `None`, then tempo is estimated independently for each frame.
    prior : scipy.stats.rv_continuous [optional]
        A prior distribution over tempo (in beats per minute).
        By default, a pseudo-log-normal prior is used.
        If given, ``start_bpm`` and ``std_bpm`` will be ignored.

    Returns
    -------
    tempo : np.ndarray
        estimated tempo (beats per minute).
        If input is multi-channel, one tempo estimate per channel is provided.

    See Also
    --------
    librosa.onset.onset_strength
    librosa.feature.tempogram

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> # Estimate a static tempo
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=30)
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    >>> tempo
    array([143.555])

    >>> # Or a static tempo with a uniform prior instead
    >>> import scipy.stats
    >>> prior = scipy.stats.uniform(30, 300)  # uniform over 30-300 BPM
    >>> utempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, prior=prior)
    >>> utempo
    array([161.499])

    >>> # Or a dynamic tempo
    >>> dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
    ...                             aggregate=None)
    >>> dtempo
    array([ 89.103,  89.103,  89.103, ..., 123.047, 123.047, 123.047])

    >>> # Dynamic tempo with a proper log-normal prior
    >>> prior_lognorm = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    >>> dtempo_lognorm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
    ...                                     aggregate=None,
    ...                                     prior=prior_lognorm)
    >>> dtempo_lognorm
    array([ 89.103,  89.103,  89.103, ..., 123.047, 123.047, 123.047])

    Plot the estimated tempo against the onset autocorrelation

    >>> import matplotlib.pyplot as plt
    >>> # Convert to scalar
    >>> tempo = tempo.item()
    >>> utempo = utempo.item()
    >>> # Compute 2-second windowed autocorrelation
    >>> hop_length = 512
    >>> ac = librosa.autocorrelate(onset_env, max_size=2 * sr // hop_length)
    >>> freqs = librosa.tempo_frequencies(len(ac), sr=sr,
    ...                                   hop_length=hop_length)
    >>> # Plot on a BPM axis.  We skip the first (0-lag) bin.
    >>> fig, ax = plt.subplots()
    >>> ax.semilogx(freqs[1:], librosa.util.normalize(ac)[1:],
    ...              label='Onset autocorrelation', base=2)
    >>> ax.axvline(tempo, 0, 1, alpha=0.75, linestyle='--', color='r',
    ...             label='Tempo (default prior): {:.2f} BPM'.format(tempo))
    >>> ax.axvline(utempo, 0, 1, alpha=0.75, linestyle=':', color='g',
    ...             label='Tempo (uniform prior): {:.2f} BPM'.format(utempo))
    >>> ax.set(xlabel='Tempo (BPM)', title='Static tempo estimation')
    >>> ax.grid(True)
    >>> ax.legend()

    Plot dynamic tempo estimates over a tempogram

    >>> fig, ax = plt.subplots()
    >>> tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
    ...                                hop_length=hop_length)
    >>> librosa.display.specshow(tg, x_axis='time', y_axis='tempo', cmap='magma', ax=ax)
    >>> ax.plot(librosa.times_like(dtempo), dtempo,
    ...          color='c', linewidth=1.5, label='Tempo estimate (default prior)')
    >>> ax.plot(librosa.times_like(dtempo_lognorm), dtempo_lognorm,
    ...          color='c', linewidth=1.5, linestyle='--',
    ...          label='Tempo estimate (lognorm prior)')
    >>> ax.set(title='Dynamic tempo estimation')
    >>> ax.legend()
    """

    if start_bpm <= 0:
        raise ParameterError("start_bpm must be strictly positive")

    win_length = core.time_to_frames(ac_size, sr=sr, hop_length=hop_length).item()

    tg = tempogram(
        y=y,
        sr=sr,
        onset_envelope=onset_envelope,
        hop_length=hop_length,
        win_length=win_length,
    )

    # Eventually, we want this to work for time-varying tempo
    if aggregate is not None:
        tg = aggregate(tg, axis=-1, keepdims=True)

    # Get the BPM values for each bin, skipping the 0-lag bin
    bpms = core.tempo_frequencies(tg.shape[-2], hop_length=hop_length, sr=sr)

    # Weight the autocorrelation by a log-normal distribution
    if prior is None:
        logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2
    else:
        logprior = prior.logpdf(bpms)

    # Kill everything above the max tempo
    if max_tempo is not None:
        max_idx = np.argmax(bpms < max_tempo)
        logprior[:max_idx] = -np.inf
    # explicit axis expansion
    logprior = util.expand_to(logprior, ndim=tg.ndim, axes=-2)

    # Get the maximum, weighted by the prior
    # Using log1p here for numerical stability
    best_period = np.argmax(np.log1p(1e6 * tg) + logprior, axis=-2)

    return np.take(bpms, best_period)


@deprecate_positional_args
def plp(
    *,
    y=None,
    sr=22050,
    onset_envelope=None,
    hop_length=512,
    win_length=384,
    tempo_min=30,
    tempo_max=300,
    prior=None,
):
    """Predominant local pulse (PLP) estimation. [#]_

    The PLP method analyzes the onset strength envelope in the frequency domain
    to find a locally stable tempo for each frame.  These local periodicities
    are used to synthesize local half-waves, which are combined such that peaks
    coincide with rhythmically salient frames (e.g. onset events on a musical time grid).
    The local maxima of the pulse curve can be taken as estimated beat positions.

    This method may be preferred over the dynamic programming method of `beat_track`
    when either the tempo is expected to vary significantly over time.  Additionally,
    since `plp` does not require the entire signal to make predictions, it may be
    preferable when beat-tracking long recordings in a streaming setting.

    .. [#] Grosche, P., & Muller, M. (2011).
        "Extracting predominant local pulse information from music recordings."
        IEEE Transactions on Audio, Speech, and Language Processing, 19(6), 1688-1701.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    onset_envelope : np.ndarray [shape=(..., n)] or None
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        number of audio samples between successive ``onset_envelope`` values

    win_length : int > 0 [scalar]
        number of frames to use for tempogram analysis.
        By default, 384 frames (at ``sr=22050`` and ``hop_length=512``) corresponds
        to about 8.9 seconds.

    tempo_min, tempo_max : numbers > 0 [scalar], optional
        Minimum and maximum permissible tempo values.  ``tempo_max`` must be at least
        ``tempo_min``.

        Set either (or both) to `None` to disable this constraint.

    prior : scipy.stats.rv_continuous [optional]
        A prior distribution over tempo (in beats per minute).
        By default, a uniform prior over ``[tempo_min, tempo_max]`` is used.

    Returns
    -------
    pulse : np.ndarray, shape=[(..., n)]
        The estimated pulse curve.  Maxima correspond to rhythmically salient
        points of time.

        If input is multi-channel, one pulse curve per channel is computed.

    See Also
    --------
    beat_track
    librosa.onset.onset_strength
    librosa.feature.fourier_tempogram

    Examples
    --------
    Visualize the PLP compared to an onset strength envelope.
    Both are normalized here to make comparison easier.

    >>> y, sr = librosa.load(librosa.ex('brahms'))
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    >>> # Or compute pulse with an alternate prior, like log-normal
    >>> import scipy.stats
    >>> prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    >>> pulse_lognorm = librosa.beat.plp(onset_envelope=onset_env, sr=sr,
    ...                                  prior=prior)
    >>> melspec = librosa.feature.melspectrogram(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.specshow(librosa.power_to_db(melspec,
    ...                                              ref=np.max),
    ...                          x_axis='time', y_axis='mel', ax=ax[0])
    >>> ax[0].set(title='Mel spectrogram')
    >>> ax[0].label_outer()
    >>> ax[1].plot(librosa.times_like(onset_env),
    ...          librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[1].plot(librosa.times_like(pulse),
    ...          librosa.util.normalize(pulse),
    ...          label='Predominant local pulse (PLP)')
    >>> ax[1].set(title='Uniform tempo prior [30, 300]')
    >>> ax[1].label_outer()
    >>> ax[2].plot(librosa.times_like(onset_env),
    ...          librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[2].plot(librosa.times_like(pulse_lognorm),
    ...          librosa.util.normalize(pulse_lognorm),
    ...          label='Predominant local pulse (PLP)')
    >>> ax[2].set(title='Log-normal tempo prior, mean=120', xlim=[5, 20])
    >>> ax[2].legend()

    PLP local maxima can be used as estimates of beat positions.

    >>> tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
    >>> beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> times = librosa.times_like(onset_env, sr=sr)
    >>> ax[0].plot(times, librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[0].vlines(times[beats], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='Beats')
    >>> ax[0].legend()
    >>> ax[0].set(title='librosa.beat.beat_track')
    >>> ax[0].label_outer()
    >>> # Limit the plot to a 15-second window
    >>> times = librosa.times_like(pulse, sr=sr)
    >>> ax[1].plot(times, librosa.util.normalize(pulse),
    ...          label='PLP')
    >>> ax[1].vlines(times[beats_plp], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='PLP Beats')
    >>> ax[1].legend()
    >>> ax[1].set(title='librosa.beat.plp', xlim=[5, 20])
    >>> ax[1].xaxis.set_major_formatter(librosa.display.TimeFormatter())

    """

    # Step 1: get the onset envelope
    if onset_envelope is None:
        onset_envelope = onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length, aggregate=np.median
        )

    if tempo_min is not None and tempo_max is not None and tempo_max <= tempo_min:
        raise ParameterError(
            "tempo_max={} must be larger than tempo_min={}".format(tempo_max, tempo_min)
        )

    # Step 2: get the fourier tempogram
    ftgram = fourier_tempogram(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
    )

    # Step 3: pin to the feasible tempo range
    tempo_frequencies = core.fourier_tempo_frequencies(
        sr=sr, hop_length=hop_length, win_length=win_length
    )

    if tempo_min is not None:
        ftgram[..., tempo_frequencies < tempo_min, :] = 0
    if tempo_max is not None:
        ftgram[..., tempo_frequencies > tempo_max, :] = 0

    # reshape lengths to match dimension properly
    tempo_frequencies = util.expand_to(tempo_frequencies, ndim=ftgram.ndim, axes=-2)

    # Step 3: Discard everything below the peak
    ftmag = np.log1p(1e6 * np.abs(ftgram))
    if prior is not None:
        ftmag += prior.logpdf(tempo_frequencies)

    peak_values = ftmag.max(axis=-2, keepdims=True)
    ftgram[ftmag < peak_values] = 0

    # Normalize to keep only phase information
    ftgram /= util.tiny(ftgram) ** 0.5 + np.abs(ftgram.max(axis=-2, keepdims=True))

    # Step 5: invert the Fourier tempogram to get the pulse
    pulse = core.istft(
        ftgram, hop_length=1, n_fft=win_length, length=onset_envelope.shape[-1]
    )

    # Step 6: retain only the positive part of the pulse cycle
    pulse = np.clip(pulse, 0, None, pulse)

    # Return the normalized pulse
    return util.normalize(pulse, axis=-1)


def __beat_tracker(onset_envelope, bpm, fft_res, tightness, trim):
    """Internal function that tracks beats in an onset strength envelope.

    Parameters
    ----------
    onset_envelope : np.ndarray [shape=(n,)]
        onset strength envelope
    bpm : float [scalar]
        tempo estimate
    fft_res : float [scalar]
        resolution of the fft (sr / hop_length)
    tightness : float [scalar]
        how closely do we adhere to bpm?
    trim : bool [scalar]
        trim leading/trailing beats with weak onsets?

    Returns
    -------
    beats : np.ndarray [shape=(n,)]
        frame numbers of beat events
    """

    if bpm <= 0:
        raise ParameterError("bpm must be strictly positive")

    # convert bpm to a sample period for searching
    period = round(60.0 * fft_res / bpm)

    # localscore is a smoothed version of AGC'd onset envelope
    localscore = __beat_local_score(onset_envelope, period)

    # run the DP
    backlink, cumscore = __beat_track_dp(localscore, period, tightness)

    # get the position of the last beat
    beats = [__last_beat(cumscore)]

    # Reconstruct the beat path from backlinks
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])

    # Put the beats in ascending order
    # Convert into an array of frame numbers
    beats = np.array(beats[::-1], dtype=int)

    # Discard spurious trailing beats
    beats = __trim_beats(localscore, beats, trim)

    return beats


# -- Helper functions for beat tracking
def __normalize_onsets(onsets):
    """Maps onset strength function into the range [0, 1]"""

    norm = onsets.std(ddof=1)
    if norm > 0:
        onsets = onsets / norm
    return onsets


def __beat_local_score(onset_envelope, period):
    """Construct the local score for an onset envlope and given period"""

    window = np.exp(-0.5 * (np.arange(-period, period + 1) * 32.0 / period) ** 2)
    return scipy.signal.convolve(__normalize_onsets(onset_envelope), window, "same")


def __beat_track_dp(localscore, period, tightness):
    """Core dynamic program for beat tracking"""

    backlink = np.zeros_like(localscore, dtype=int)
    cumscore = np.zeros_like(localscore)

    # Search range for previous beat
    window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)

    # Make a score window, which begins biased toward start_bpm and skewed
    if tightness <= 0:
        raise ParameterError("tightness must be strictly positive")

    txwt = -tightness * (np.log(-window / period) ** 2)

    # Are we on the first beat?
    first_beat = True
    for i, score_i in enumerate(localscore):

        # Are we reaching back before time 0?
        z_pad = np.maximum(0, min(-window[0], len(window)))

        # Search over all possible predecessors
        candidates = txwt.copy()
        candidates[z_pad:] = candidates[z_pad:] + cumscore[window[z_pad:]]

        # Find the best preceding beat
        beat_location = np.argmax(candidates)

        # Add the local score
        cumscore[i] = score_i + candidates[beat_location]

        # Special case the first onset.  Stop if the localscore is small
        if first_beat and score_i < 0.01 * localscore.max():
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location]
            first_beat = False

        # Update the time range
        window = window + 1

    return backlink, cumscore


def __last_beat(cumscore):
    """Get the last beat from the cumulative score array"""

    maxes = util.localmax(cumscore)
    med_score = np.median(cumscore[np.argwhere(maxes)])

    # The last of these is the last beat (since score generally increases)
    return np.argwhere((cumscore * maxes * 2 > med_score)).max()


def __trim_beats(localscore, beats, trim):
    """Final post-processing: throw out spurious leading/trailing beats"""

    smooth_boe = scipy.signal.convolve(localscore[beats], scipy.signal.hann(5), "same")

    if trim:
        threshold = 0.5 * ((smooth_boe ** 2).mean() ** 0.5)
    else:
        threshold = 0.0

    valid = np.argwhere(smooth_boe > threshold)

    return beats[valid.min() : valid.max()]
