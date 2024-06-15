#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Onset detection
===============
.. autosummary::
    :toctree: generated/

    onset_detect
    onset_backtrack
    onset_strength
    onset_strength_multi
"""

import numpy as np
import scipy

from ._cache import cache
from . import core
from . import util
from .util.exceptions import ParameterError
from .util.decorators import deprecate_positional_args

from .feature.spectral import melspectrogram

__all__ = ["onset_detect", "onset_strength", "onset_strength_multi", "onset_backtrack"]


@deprecate_positional_args
def onset_detect(
    *,
    y=None,
    sr=22050,
    onset_envelope=None,
    hop_length=512,
    backtrack=False,
    energy=None,
    units="frames",
    normalize=True,
    **kwargs,
):
    """Locate note onset events by picking peaks in an onset strength envelope.

    The `peak_pick` parameters were chosen by large-scale hyper-parameter
    optimization over the dataset provided by [#]_.

    .. [#] https://github.com/CPJKU/onset_db

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series, must be monophonic

    sr : number > 0 [scalar]
        sampling rate of ``y``

    onset_envelope : np.ndarray [shape=(m,)]
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length (in samples)

    units : {'frames', 'samples', 'time'}
        The units to encode detected onset events in.
        By default, 'frames' are used.

    backtrack : bool
        If ``True``, detected onset events are backtracked to the nearest
        preceding minimum of ``energy``.

        This is primarily useful when using onsets as slice points for segmentation.

    energy : np.ndarray [shape=(m,)] (optional)
        An energy function to use for backtracking detected onset events.
        If none is provided, then ``onset_envelope`` is used.

    normalize : bool
        If ``True`` (default), normalize the onset envelope to have minimum of 0 and
        maximum of 1 prior to detection.  This is helpful for standardizing the
        parameters of `librosa.util.peak_pick`.

        Otherwise, the onset envelope is left unnormalized.

    **kwargs : additional keyword arguments
        Additional parameters for peak picking.

        See `librosa.util.peak_pick` for details.

    Returns
    -------
    onsets : np.ndarray [shape=(n_onsets,)]
        estimated positions of detected onsets, in whichever units
        are specified.  By default, frame indices.

        .. note::
            If no onset strength could be detected, onset_detect returns
            an empty list.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onsets`` are provided

        or if ``units`` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    onset_strength : compute onset strength per-frame
    onset_backtrack : backtracking onset events
    librosa.util.peak_pick : pick peaks from a time series

    Examples
    --------
    Get onset times from a signal

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.onset.onset_detect(y=y, sr=sr, units='time')
    array([0.07 , 0.232, 0.395, 0.604, 0.743, 0.929, 1.045, 1.115,
           1.416, 1.672, 1.881, 2.043, 2.206, 2.368, 2.554, 3.019])

    Or use a pre-computed onset envelope

    >>> o_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> times = librosa.times_like(o_env, sr=sr)
    >>> onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> D = np.abs(librosa.stft(y))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          x_axis='time', y_axis='log', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, o_env, label='Onset strength')
    >>> ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
    ...            linestyle='--', label='Onsets')
    >>> ax[1].legend()
    """

    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError("y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Shift onset envelope up to be non-negative
    # (a common normalization step to make the threshold more consistent)
    if normalize:
        # Normalize onset strength function to [0, 1] range
        onset_envelope = onset_envelope - onset_envelope.min()
        # Max-scale with safe division
        onset_envelope /= np.max(onset_envelope) + util.tiny(onset_envelope)

    # Do we have any onsets to grab?
    if not onset_envelope.any() or not np.all(np.isfinite(onset_envelope)):
        onsets = np.array([], dtype=int)

    else:
        # These parameter settings found by large-scale search
        kwargs.setdefault("pre_max", 0.03 * sr // hop_length)  # 30ms
        kwargs.setdefault("post_max", 0.00 * sr // hop_length + 1)  # 0ms
        kwargs.setdefault("pre_avg", 0.10 * sr // hop_length)  # 100ms
        kwargs.setdefault("post_avg", 0.10 * sr // hop_length + 1)  # 100ms
        kwargs.setdefault("wait", 0.03 * sr // hop_length)  # 30ms
        kwargs.setdefault("delta", 0.07)

        # Peak pick the onset envelope
        onsets = util.peak_pick(onset_envelope, **kwargs)

        # Optionally backtrack the events
        if backtrack:
            if energy is None:
                energy = onset_envelope

            onsets = onset_backtrack(onsets, energy)

    if units == "frames":
        pass
    elif units == "samples":
        onsets = core.frames_to_samples(onsets, hop_length=hop_length)
    elif units == "time":
        onsets = core.frames_to_time(onsets, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError("Invalid unit type: {}".format(units))

    return onsets


@deprecate_positional_args
def onset_strength(
    *,
    y=None,
    sr=22050,
    S=None,
    lag=1,
    max_size=1,
    ref=None,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    **kwargs,
):
    """Compute a spectral flux onset strength envelope.

    Onset strength at time ``t`` is determined by::

        mean_f max(0, S[f, t] - ref[f, t - lag])

    where ``ref`` is ``S`` after local max filtering along the frequency
    axis [#]_.

    By default, if a time series ``y`` is provided, S will be the
    log-power Mel spectrogram.

    .. [#] Böck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, m)]
        pre-computed (log-power) spectrogram

    lag : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.

    ref : None or np.ndarray [shape=(..., d, m)]
        An optional pre-computed reference spectrum, of the same shape as ``S``.
        If not provided, it will be computed from ``S``.
        If provided, it will override any local max filtering governed by ``max_size``.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.

    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``

    aggregate : function
        Aggregation function to use when combining onsets
        at different frequency bins.

        Default: `np.mean`

    **kwargs : additional keyword arguments
        Additional parameters to ``feature()``, if ``S`` is not provided.

    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., m,)]
        vector containing the onset strength envelope.
        If the input contains multiple channels, then onset envelope is computed for each channel.

    Raises
    ------
    ParameterError
        if neither ``(y, sr)`` nor ``S`` are provided

        or if ``lag`` or ``max_size`` are not positive integers

    See Also
    --------
    onset_detect
    onset_strength_multi

    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> D = np.abs(librosa.stft(y))
    >>> times = librosa.times_like(D)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()

    Construct a standard onset function

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> ax[1].plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,
    ...            label='Mean (mel)')

    Median aggregation, and custom mel options

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          aggregate=np.median,
    ...                                          fmax=8000, n_mels=256)
    >>> ax[1].plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,
    ...            label='Median (custom mel)')

    Constant-Q spectrogram instead of Mel

    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> onset_env = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
    >>> ax[1].plot(times, onset_env / onset_env.max(), alpha=0.8,
    ...          label='Mean (CQT)')
    >>> ax[1].legend()
    >>> ax[1].set(ylabel='Normalized strength', yticks=[])
    """

    if aggregate is False:
        raise ParameterError(
            "aggregate={} cannot be False when computing full-spectrum onset strength."
        )

    odf_all = onset_strength_multi(
        y=y,
        sr=sr,
        S=S,
        lag=lag,
        max_size=max_size,
        ref=ref,
        detrend=detrend,
        center=center,
        feature=feature,
        aggregate=aggregate,
        channels=None,
        **kwargs,
    )

    return odf_all[..., 0, :]


def onset_backtrack(events, energy):
    """Backtrack detected onset events to the nearest preceding local
    minimum of an energy function.

    This function can be used to roll back the timing of detected onsets
    from a detected peak amplitude to the preceding minimum.

    This is most useful when using onsets to determine slice points for
    segmentation, as described by [#]_.

    .. [#] Jehan, Tristan.
           "Creating music by listening"
           Doctoral dissertation
           Massachusetts Institute of Technology, 2005.

    Parameters
    ----------
    events : np.ndarray, dtype=int
        List of onset event frame indices, as computed by `onset_detect`
    energy : np.ndarray, shape=(m,)
        An energy function

    Returns
    -------
    events_backtracked : np.ndarray, shape=events.shape
        The input events matched to nearest preceding minima of ``energy``.

    Examples
    --------
    Backtrack the events using the onset envelope

    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr)
    >>> times = librosa.times_like(oenv)
    >>> # Detect events without backtracking
    >>> onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,
    ...                                        backtrack=False)
    >>> onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)

    Backtrack the events using the RMS values

    >>> S = np.abs(librosa.stft(y=y))
    >>> rms = librosa.feature.rms(S=S)
    >>> onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])

    Plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, oenv, label='Onset strength')
    >>> ax[1].vlines(librosa.frames_to_time(onset_raw), 0, oenv.max(), label='Raw onsets')
    >>> ax[1].vlines(librosa.frames_to_time(onset_bt), 0, oenv.max(), label='Backtracked', color='r')
    >>> ax[1].legend()
    >>> ax[1].label_outer()
    >>> ax[2].plot(times, rms[0], label='RMS')
    >>> ax[2].vlines(librosa.frames_to_time(onset_bt_rms), 0, rms.max(), label='Backtracked (RMS)', color='r')
    >>> ax[2].legend()
    """

    # Find points where energy is non-increasing
    # all points:  energy[i] <= energy[i-1]
    # tail points: energy[i] < energy[i+1]
    minima = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:]))

    # Pad on a 0, just in case we have onsets with no preceding minimum
    # Shift by one to account for slicing in minima detection
    minima = util.fix_frames(1 + minima, x_min=0)

    # Only match going left from the detected events
    return minima[util.match_events(events, minima, right=False)]


@deprecate_positional_args
@cache(level=30)
def onset_strength_multi(
    *,
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    lag=1,
    max_size=1,
    ref=None,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    channels=None,
    **kwargs,
):
    """Compute a spectral flux onset strength envelope across multiple channels.

    Onset strength for channel ``i`` at time ``t`` is determined by::

        mean_{f in channels[i]} max(0, S[f, t+1] - S[f, t])

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, m)]
        pre-computed (log-power) spectrogram

    n_fft : int > 0 [scalar]
        FFT window size for use in ``feature()`` if ``S`` is not provided.

    hop_length : int > 0 [scalar]
        hop length for use in ``feature()`` if ``S`` is not provided.

    lag : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.

    ref : None or np.ndarray [shape=(d, m)]
        An optional pre-computed reference spectrum, of the same shape as ``S``.
        If not provided, it will be computed from ``S``.
        If provided, it will override any local max filtering governed by ``max_size``.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.

    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``

        Must support arguments: ``y, sr, n_fft, hop_length``

    aggregate : function or False
        Aggregation function to use when combining onsets
        at different frequency bins.

        If ``False``, then no aggregation is performed.

        Default: `np.mean`

    channels : list or None
        Array of channel boundaries or slice objects.
        If `None`, then a single channel is generated to span all bands.

    **kwargs : additional keyword arguments
        Additional parameters to ``feature()``, if ``S`` is not provided.

    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., n_channels, m)]
        array containing the onset strength envelope for each specified channel

    Raises
    ------
    ParameterError
        if neither ``(y, sr)`` nor ``S`` are provided

    See Also
    --------
    onset_strength

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=5)
    >>> D = np.abs(librosa.stft(y))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img1 = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> fig.colorbar(img1, ax=[ax[0]], format="%+2.f dB")

    Construct a standard onset function over four sub-bands

    >>> onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,
    ...                                                     channels=[0, 32, 64, 96, 128])
    >>> img2 = librosa.display.specshow(onset_subbands, x_axis='time', ax=ax[1])
    >>> ax[1].set(ylabel='Sub-bands', title='Sub-band onset strength')
    >>> fig.colorbar(img2, ax=[ax[1]])
    """

    if feature is None:
        feature = melspectrogram
        kwargs.setdefault("fmax", 0.5 * sr)

    if aggregate is None:
        aggregate = np.mean

    if lag < 1 or not isinstance(lag, (int, np.integer)):
        raise ParameterError("lag must be a positive integer")

    if max_size < 1 or not isinstance(max_size, (int, np.integer)):
        raise ParameterError("max_size must be a positive integer")

    # First, compute mel spectrogram
    if S is None:
        S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))

        # Convert to dBs
        S = core.power_to_db(S)

    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute the reference spectrogram.
    # Efficiency hack: skip filtering step and pass by reference
    # if max_size will produce a no-op.
    if ref is None:
        if max_size == 1:
            ref = S
        else:
            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=-2)
    elif ref.shape != S.shape:
        raise ParameterError(
            "Reference spectrum shape {} must match input spectrum {}".format(
                ref.shape, S.shape
            )
        )

    # Compute difference to the reference, spaced by lag
    onset_env = S[..., lag:] - ref[..., :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    if aggregate:
        onset_env = util.sync(
            onset_env, channels, aggregate=aggregate, pad=pad, axis=-2
        )

    # compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    padding = [(0, 0) for _ in onset_env.shape]
    padding[-1] = (int(pad_width), 0)
    onset_env = np.pad(onset_env, padding, mode="constant")

    # remove the DC component
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[..., : S.shape[-1]]

    return onset_env
