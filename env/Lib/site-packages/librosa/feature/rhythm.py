#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rhythmic feature extraction"""

import numpy as np

from .. import util

from ..core.audio import autocorrelate
from ..core.spectrum import stft
from ..util.exceptions import ParameterError
from ..util.decorators import deprecate_positional_args
from ..filters import get_window


__all__ = ["tempogram", "fourier_tempogram"]


# -- Rhythmic features -- #
@deprecate_positional_args
def tempogram(
    *,
    y=None,
    sr=22050,
    onset_envelope=None,
    hop_length=512,
    win_length=384,
    center=True,
    window="hann",
    norm=np.inf,
):
    """Compute the tempogram: local autocorrelation of the onset strength envelope. [#]_

    .. [#] Grosche, Peter, Meinard Müller, and Frank Kurth.
        "Cyclic tempogram - A mid-level tempo representation for music signals."
        ICASSP, 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        Audio time series.  Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    onset_envelope : np.ndarray [shape=(..., n) or (..., m, n)] or None
        Optional pre-computed onset strength envelope as provided by
        `librosa.onset.onset_strength`.

        If multi-dimensional, tempograms are computed independently for each
        band (first dimension).

    hop_length : int > 0
        number of audio samples between successive onset measurements

    win_length : int > 0
        length of the onset autocorrelation window (in frames/onset measurements)
        The default settings (384) corresponds to ``384 * hop_length / sr ~= 8.9s``.

    center : bool
        If `True`, onset autocorrelation windows are centered.
        If `False`, windows are left-aligned.

    window : string, function, number, tuple, or np.ndarray [shape=(win_length,)]
        A window specification as in `stft`.

    norm : {np.inf, -np.inf, 0, float > 0, None}
        Normalization mode.  Set to `None` to disable normalization.

    Returns
    -------
    tempogram : np.ndarray [shape=(..., win_length, n)]
        Localized autocorrelation of the onset strength envelope.

        If given multi-band input (``onset_envelope.shape==(m,n)``) then
        ``tempogram[i]`` is the tempogram of ``onset_envelope[i]``.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onset_envelope`` are provided

        if ``win_length < 1``

    See Also
    --------
    fourier_tempogram
    librosa.onset.onset_strength
    librosa.util.normalize
    librosa.stft

    Examples
    --------
    >>> # Compute local onset autocorrelation
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=30)
    >>> hop_length = 512
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    >>> tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
    ...                                       hop_length=hop_length)
    >>> # Compute global onset autocorrelation
    >>> ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    >>> ac_global = librosa.util.normalize(ac_global)
    >>> # Estimate the global tempo for display purposes
    >>> tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
    ...                            hop_length=hop_length)[0]

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
    >>> times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
    >>> ax[0].plot(times, oenv, label='Onset strength')
    >>> ax[0].label_outer()
    >>> ax[0].legend(frameon=True)
    >>> librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='tempo', cmap='magma',
    ...                          ax=ax[1])
    >>> ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,
    ...             label='Estimated tempo={:g}'.format(tempo))
    >>> ax[1].legend(loc='upper right')
    >>> ax[1].set(title='Tempogram')
    >>> x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
    ...                 num=tempogram.shape[0])
    >>> ax[2].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    >>> ax[2].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    >>> ax[2].set(xlabel='Lag (seconds)')
    >>> ax[2].legend(frameon=True)
    >>> freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    >>> ax[3].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
    ...              label='Mean local autocorrelation', base=2)
    >>> ax[3].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
    ...              label='Global autocorrelation', base=2)
    >>> ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8,
    ...             label='Estimated tempo={:g}'.format(tempo))
    >>> ax[3].legend(frameon=True)
    >>> ax[3].set(xlabel='BPM')
    >>> ax[3].grid(True)
    """

    from ..onset import onset_strength

    if win_length < 1:
        raise ParameterError("win_length must be a positive integer")

    ac_window = get_window(window, win_length, fftbins=True)

    if onset_envelope is None:
        if y is None:
            raise ParameterError("Either y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Center the autocorrelation windows
    n = onset_envelope.shape[-1]

    if center:
        padding = [(0, 0) for _ in onset_envelope.shape]
        padding[-1] = (int(win_length // 2),) * 2
        onset_envelope = np.pad(
            onset_envelope, padding, mode="linear_ramp", end_values=[0, 0]
        )

    # Carve onset envelope into frames
    odf_frame = util.frame(onset_envelope, frame_length=win_length, hop_length=1)

    # Truncate to the length of the original signal
    if center:
        odf_frame = odf_frame[..., :n]

    # explicit broadcast of ac_window
    ac_window = util.expand_to(ac_window, ndim=odf_frame.ndim, axes=-2)

    # Window, autocorrelate, and normalize
    return util.normalize(
        autocorrelate(odf_frame * ac_window, axis=-2), norm=norm, axis=-2
    )


@deprecate_positional_args
def fourier_tempogram(
    *,
    y=None,
    sr=22050,
    onset_envelope=None,
    hop_length=512,
    win_length=384,
    center=True,
    window="hann",
):
    """Compute the Fourier tempogram: the short-time Fourier transform of the
    onset strength envelope. [#]_

    .. [#] Grosche, Peter, Meinard Müller, and Frank Kurth.
        "Cyclic tempogram - A mid-level tempo representation for music signals."
        ICASSP, 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        Audio time series.  Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    onset_envelope : np.ndarray [shape=(..., n)] or None
        Optional pre-computed onset strength envelope as provided by
        ``librosa.onset.onset_strength``.
        Multi-channel is supported.
    hop_length : int > 0
        number of audio samples between successive onset measurements
    win_length : int > 0
        length of the onset window (in frames/onset measurements)
        The default settings (384) corresponds to ``384 * hop_length / sr ~= 8.9s``.
    center : bool
        If `True`, onset windows are centered.
        If `False`, windows are left-aligned.
    window : string, function, number, tuple, or np.ndarray [shape=(win_length,)]
        A window specification as in `stft`.

    Returns
    -------
    tempogram : np.ndarray [shape=(..., win_length // 2 + 1, n)]
        Complex short-time Fourier transform of the onset envelope.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onset_envelope`` are provided

        if ``win_length < 1``

    See Also
    --------
    tempogram
    librosa.onset.onset_strength
    librosa.util.normalize
    librosa.stft

    Examples
    --------
    >>> # Compute local onset autocorrelation
    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 512
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    >>> tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr,
    ...                                               hop_length=hop_length)
    >>> # Compute the auto-correlation tempogram, unnormalized to make comparison easier
    >>> ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
    ...                                          hop_length=hop_length, norm=None)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> ax[0].plot(librosa.times_like(oenv), oenv, label='Onset strength')
    >>> ax[0].legend(frameon=True)
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(np.abs(tempogram), sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='fourier_tempo', cmap='magma',
    ...                          ax=ax[1])
    >>> ax[1].set(title='Fourier tempogram')
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(ac_tempogram, sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='tempo', cmap='magma',
    ...                          ax=ax[2])
    >>> ax[2].set(title='Autocorrelation tempogram')
    """

    from ..onset import onset_strength

    if win_length < 1:
        raise ParameterError("win_length must be a positive integer")

    if onset_envelope is None:
        if y is None:
            raise ParameterError("Either y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Generate the short-time Fourier transform
    return stft(
        onset_envelope, n_fft=win_length, hop_length=1, center=center, window=window
    )
