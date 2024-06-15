#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Harmonic calculations for frequency representations"""

import warnings

import numpy as np
import scipy.interpolate
import scipy.signal
from ..util.exceptions import ParameterError
from ..util import is_unique
from ..util.decorators import deprecate_positional_args

__all__ = ["salience", "interp_harmonics"]


@deprecate_positional_args
def salience(
    S,
    *,
    freqs,
    harmonics,
    weights=None,
    aggregate=None,
    filter_peaks=True,
    fill_value=np.nan,
    kind="linear",
    axis=-2,
):
    """Harmonic salience function.

    Parameters
    ----------
    S : np.ndarray [shape=(..., d, n)]
        input time frequency magnitude representation (e.g. STFT or CQT magnitudes).
        Must be real-valued and non-negative.

    freqs : np.ndarray, shape=(S.shape[axis])
        The frequency values corresponding to S's elements along the
        chosen axis.

    harmonics : list-like, non-negative
        Harmonics to include in salience computation.  The first harmonic (1)
        corresponds to ``S`` itself. Values less than one (e.g., 1/2) correspond
        to sub-harmonics.

    weights : list-like
        The weight to apply to each harmonic in the summation. (default:
        uniform weights). Must be the same length as ``harmonics``.

    aggregate : function
        aggregation function (default: `np.average`)

        If ``aggregate=np.average``, then a weighted average is
        computed per-harmonic according to the specified weights.
        For all other aggregation functions, all harmonics
        are treated equally.

    filter_peaks : bool
        If true, returns harmonic summation only on frequencies of peak
        magnitude. Otherwise returns harmonic summation over the full spectrum.
        Defaults to True.

    fill_value : float
        The value to fill non-peaks in the output representation. (default:
        `np.nan`) Only used if ``filter_peaks == True``.

    kind : str
        Interpolation type for harmonic estimation.
        See `scipy.interpolate.interp1d`.

    axis : int
        The axis along which to compute harmonics

    Returns
    -------
    S_sal : np.ndarray
        ``S_sal`` will have the same shape as ``S``, and measure
        the overall harmonic energy at each frequency.

    See Also
    --------
    interp_harmonics

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> S = np.abs(librosa.stft(y))
    >>> freqs = librosa.fft_frequencies(sr=sr)
    >>> harms = [1, 2, 3, 4]
    >>> weights = [1.0, 0.5, 0.33, 0.25]
    >>> S_sal = librosa.salience(S, freqs=freqs, harmonics=harms, weights=weights, fill_value=0)
    >>> print(S_sal.shape)
    (1025, 115)
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Magnitude spectrogram')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S_sal,
    ...                                                        ref=np.max),
    ...                                sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Salience spectrogram')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """
    if aggregate is None:
        aggregate = np.average

    if weights is None:
        weights = np.ones((len(harmonics),))
    else:
        weights = np.array(weights, dtype=float)

    S_harm = interp_harmonics(S, freqs=freqs, harmonics=harmonics, kind=kind, axis=axis)

    if aggregate is np.average:
        S_sal = aggregate(S_harm, axis=axis - 1, weights=weights)
    else:
        S_sal = aggregate(S_harm, axis=axis - 1)

    if filter_peaks:
        S_peaks = scipy.signal.argrelmax(S, axis=axis)
        S_out = np.empty(S.shape)
        S_out.fill(fill_value)
        S_out[S_peaks] = S_sal[S_peaks]

        S_sal = S_out

    return S_sal


@deprecate_positional_args
def interp_harmonics(x, *, freqs, harmonics, kind="linear", fill_value=0, axis=-2):
    """Compute the energy at harmonics of time-frequency representation.

    Given a frequency-based energy representation such as a spectrogram
    or tempogram, this function computes the energy at the chosen harmonics
    of the frequency axis.  (See examples below.)
    The resulting harmonic array can then be used as input to a salience
    computation.

    Parameters
    ----------
    x : np.ndarray
        The input energy
    freqs : np.ndarray, shape=(X.shape[axis])
        The frequency values corresponding to X's elements along the
        chosen axis.
    harmonics : list-like, non-negative
        Harmonics to compute as ``harmonics[i] * freqs``.
        The first harmonic (1) corresponds to ``freqs``.
        Values less than one (e.g., 1/2) correspond to sub-harmonics.
    kind : str
        Interpolation type.  See `scipy.interpolate.interp1d`.
    fill_value : float
        The value to fill when extrapolating beyond the observed
        frequency range.
    axis : int
        The axis along which to compute harmonics

    Returns
    -------
    x_harm : np.ndarray
        ``x_harm[i]`` will have the same shape as ``x``, and measure
        the energy at the ``harmonics[i]`` harmonic of each frequency.
        A new dimension indexing harmonics will be inserted immediately
        before ``axis``.

    See Also
    --------
    scipy.interpolate.interp1d

    Examples
    --------
    Estimate the harmonics of a time-averaged tempogram

    >>> y, sr = librosa.load(librosa.ex('sweetwaltz'))
    >>> # Compute the time-varying tempogram and average over time
    >>> tempi = np.mean(librosa.feature.tempogram(y=y, sr=sr), axis=1)
    >>> # We'll measure the first five harmonics
    >>> harmonics = [1, 2, 3, 4, 5]
    >>> f_tempo = librosa.tempo_frequencies(len(tempi), sr=sr)
    >>> # Build the harmonic tensor; we only have one axis here (tempo)
    >>> t_harmonics = librosa.interp_harmonics(tempi, freqs=f_tempo, harmonics=harmonics, axis=0)
    >>> print(t_harmonics.shape)
    (5, 384)

    >>> # And plot the results
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(t_harmonics, x_axis='tempo', sr=sr, ax=ax)
    >>> ax.set(yticks=np.arange(len(harmonics)),
    ...        yticklabels=['{:.3g}'.format(_) for _ in harmonics],
    ...        ylabel='Harmonic', xlabel='Tempo (BPM)')

    We can also compute frequency harmonics for spectrograms.
    To calculate sub-harmonic energy, use values < 1.

    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> harmonics = [1./3, 1./2, 1, 2, 3, 4]
    >>> S = np.abs(librosa.stft(y))
    >>> fft_freqs = librosa.fft_frequencies(sr=sr)
    >>> S_harm = librosa.interp_harmonics(S, freqs=fft_freqs, harmonics=harmonics, axis=0)
    >>> print(S_harm.shape)
    (6, 1025, 646)

    >>> fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    >>> for i, _sh in enumerate(S_harm):
    ...     img = librosa.display.specshow(librosa.amplitude_to_db(_sh,
    ...                                                      ref=S.max()),
    ...                              sr=sr, y_axis='log', x_axis='time',
    ...                              ax=ax.flat[i])
    ...     ax.flat[i].set(title='h={:.3g}'.format(harmonics[i]))
    ...     ax.flat[i].label_outer()
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    """

    if freqs.ndim == 1 and len(freqs) == x.shape[axis]:
        # Build the 1-D interpolator.
        # All frames have a common domain, so we only need one interpolator here.

        # First, verify that the input frequencies are unique
        if not is_unique(freqs, axis=0):
            warnings.warn(
                "Frequencies are not unique. This may produce incorrect harmonic interpolations.",
                stacklevel=2,
            )

        f_interp = scipy.interpolate.interp1d(
            freqs,
            x,
            axis=axis,
            bounds_error=False,
            copy=False,
            kind=kind,
            fill_value=fill_value,
        )

        # Set the interpolation points
        f_out = np.multiply.outer(harmonics, freqs)

        # Interpolate
        return f_interp(f_out)

    elif freqs.shape == x.shape:
        if not np.all(is_unique(freqs, axis=axis)):
            warnings.warn(
                "Frequencies are not unique. This may produce incorrect harmonic interpolations.",
                stacklevel=2,
            )

        # If we have time-varying frequencies, then it must match exactly the shape of the input

        # We'll define a frame-wise interpolator helper function that we will vectorize over
        # the entire input array
        def _f_interp(_a, _b):
            interp = scipy.interpolate.interp1d(
                _a, _b, bounds_error=False, copy=False, kind=kind, fill_value=fill_value
            )

            return interp(np.multiply.outer(_a, harmonics))

        # Signature is expanding frequency into a new dimension
        xfunc = np.vectorize(_f_interp, signature="(f),(f)->(f,h)")

        # Rotate the vectorizing axis to the tail so that we get parallelism over frames
        # Afterward, we're swapping (-1, axis-1) instead of (-1,axis)
        # because a new dimension has been inserted
        return (
            xfunc(freqs.swapaxes(axis, -1), x.swapaxes(axis, -1))
            .swapaxes(
                # Return the original target axis to its place
                -2,
                axis,
            )
            .swapaxes(
                # Put the new harmonic axis directly in front of the target axis
                -1,
                axis - 1,
            )
        )
    else:
        raise ParameterError(
            "freqs.shape={} does not match "
            "input shape={}".format(freqs.shape, x.shape)
        )
