#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature manipulation utilities"""

import numpy as np
import scipy.signal
from numba import jit

from .._cache import cache
from ..util.exceptions import ParameterError
from ..util.decorators import deprecate_positional_args

__all__ = ["delta", "stack_memory"]


@deprecate_positional_args
@cache(level=40)
def delta(data, *, width=9, order=1, axis=-1, mode="interp", **kwargs):
    r"""Compute delta features: local estimate of the derivative
    of the input data along the selected axis.

    Delta features are computed Savitsky-Golay filtering.

    Parameters
    ----------
    data : np.ndarray
        the input data matrix (eg, spectrogram)

    width : int, positive, odd [scalar]
        Number of frames over which to compute the delta features.
        Cannot exceed the length of ``data`` along the specified axis.

        If ``mode='interp'``, then ``width`` must be at least ``data.shape[axis]``.

    order : int > 0 [scalar]
        the order of the difference operator.
        1 for first derivative, 2 for second, etc.

    axis : int [scalar]
        the axis along which to compute deltas.
        Default is -1 (columns).

    mode : str, {'interp', 'nearest', 'mirror', 'constant', 'wrap'}
        Padding mode for estimating differences at the boundaries.

    **kwargs : additional keyword arguments
        See `scipy.signal.savgol_filter`

    Returns
    -------
    delta_data : np.ndarray [shape=(..., t)]
        delta matrix of ``data`` at specified order

    Notes
    -----
    This function caches at level 40.

    See Also
    --------
    scipy.signal.savgol_filter

    Examples
    --------
    Compute MFCC deltas, delta-deltas

    >>> y, sr = librosa.load(librosa.ex('libri1'), duration=5)
    >>> mfcc = librosa.feature.mfcc(y=y, sr=sr)
    >>> mfcc_delta = librosa.feature.delta(mfcc)
    >>> mfcc_delta
    array([[-5.713e+02, -5.697e+02, ..., -1.522e+02, -1.224e+02],
           [ 1.104e+01,  1.330e+01, ...,  2.089e+02,  1.698e+02],
           ...,
           [ 2.829e+00,  1.933e+00, ..., -3.149e+00,  2.294e-01],
           [ 2.890e+00,  2.187e+00, ...,  6.959e+00, -1.039e+00]],
          dtype=float32)

    >>> mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    >>> mfcc_delta2
    array([[-1.195, -1.195, ..., -4.328, -4.328],
           [-1.566, -1.566, ..., -9.949, -9.949],
           ...,
           [ 0.707,  0.707, ...,  2.287,  2.287],
           [ 0.655,  0.655, ..., -1.719, -1.719]], dtype=float32)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> img1 = librosa.display.specshow(mfcc, ax=ax[0], x_axis='time')
    >>> ax[0].set(title='MFCC')
    >>> ax[0].label_outer()
    >>> img2 = librosa.display.specshow(mfcc_delta, ax=ax[1], x_axis='time')
    >>> ax[1].set(title=r'MFCC-$\Delta$')
    >>> ax[1].label_outer()
    >>> img3 = librosa.display.specshow(mfcc_delta2, ax=ax[2], x_axis='time')
    >>> ax[2].set(title=r'MFCC-$\Delta^2$')
    >>> fig.colorbar(img1, ax=[ax[0]])
    >>> fig.colorbar(img2, ax=[ax[1]])
    >>> fig.colorbar(img3, ax=[ax[2]])
    """

    data = np.atleast_1d(data)

    if mode == "interp" and width > data.shape[axis]:
        raise ParameterError(
            "when mode='interp', width={} "
            "cannot exceed data.shape[axis]={}".format(width, data.shape[axis])
        )

    if width < 3 or np.mod(width, 2) != 1:
        raise ParameterError("width must be an odd integer >= 3")

    if order <= 0 or not isinstance(order, (int, np.integer)):
        raise ParameterError("order must be a positive integer")

    kwargs.pop("deriv", None)
    kwargs.setdefault("polyorder", order)
    return scipy.signal.savgol_filter(
        data, width, deriv=order, axis=axis, mode=mode, **kwargs
    )


@deprecate_positional_args
@cache(level=40)
def stack_memory(data, *, n_steps=2, delay=1, **kwargs):
    """Short-term history embedding: vertically concatenate a data
    vector or matrix with delayed copies of itself.

    Each column ``data[:, i]`` is mapped to::

        data[..., i] ->  [data[..., i],
                          data[..., i - delay],
                          ...
                          data[..., i - (n_steps-1)*delay]]

    For columns ``i < (n_steps - 1) * delay``, the data will be padded.
    By default, the data is padded with zeros, but this behavior can be
    overridden by supplying additional keyword arguments which are passed
    to `np.pad()`.

    Parameters
    ----------
    data : np.ndarray [shape=(..., d, t)]
        Input data matrix.  If ``data`` is a vector (``data.ndim == 1``),
        it will be interpreted as a row matrix and reshaped to ``(1, t)``.

    n_steps : int > 0 [scalar]
        embedding dimension, the number of steps back in time to stack

    delay : int != 0 [scalar]
        the number of columns to step.

        Positive values embed from the past (previous columns).

        Negative values embed from the future (subsequent columns).

    **kwargs : additional keyword arguments
        Additional arguments to pass to `numpy.pad`

    Returns
    -------
    data_history : np.ndarray [shape=(..., m * d, t)]
        data augmented with lagged copies of itself,
        where ``m == n_steps - 1``.

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    Keep two steps (current and previous)

    >>> data = np.arange(-3, 3)
    >>> librosa.feature.stack_memory(data)
    array([[-3, -2, -1,  0,  1,  2],
           [ 0, -3, -2, -1,  0,  1]])

    Or three steps

    >>> librosa.feature.stack_memory(data, n_steps=3)
    array([[-3, -2, -1,  0,  1,  2],
           [ 0, -3, -2, -1,  0,  1],
           [ 0,  0, -3, -2, -1,  0]])

    Use reflection padding instead of zero-padding

    >>> librosa.feature.stack_memory(data, n_steps=3, mode='reflect')
    array([[-3, -2, -1,  0,  1,  2],
           [-2, -3, -2, -1,  0,  1],
           [-1, -2, -3, -2, -1,  0]])

    Or pad with edge-values, and delay by 2

    >>> librosa.feature.stack_memory(data, n_steps=3, delay=2, mode='edge')
    array([[-3, -2, -1,  0,  1,  2],
           [-3, -3, -3, -2, -1,  0],
           [-3, -3, -3, -3, -3, -2]])

    Stack time-lagged beat-synchronous chroma edge padding

    >>> y, sr = librosa.load(librosa.ex('sweetwaltz'), duration=10)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    >>> beats = librosa.util.fix_frames(beats, x_min=0)
    >>> chroma_sync = librosa.util.sync(chroma, beats)
    >>> chroma_lag = librosa.feature.stack_memory(chroma_sync, n_steps=3,
    ...                                           mode='edge')

    Plot the result

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    >>> librosa.display.specshow(chroma_lag, y_axis='chroma', x_axis='time',
    ...                          x_coords=beat_times, ax=ax)
    >>> ax.text(1.0, 1/6, "Lag=0", transform=ax.transAxes, rotation=-90, ha="left", va="center")
    >>> ax.text(1.0, 3/6, "Lag=1", transform=ax.transAxes, rotation=-90, ha="left", va="center")
    >>> ax.text(1.0, 5/6, "Lag=2", transform=ax.transAxes, rotation=-90, ha="left", va="center")
    >>> ax.set(title='Time-lagged chroma', ylabel="")
    """

    if n_steps < 1:
        raise ParameterError("n_steps must be a positive integer")

    if delay == 0:
        raise ParameterError("delay must be a non-zero integer")

    data = np.atleast_2d(data)
    t = data.shape[-1]

    if t < 1:
        raise ParameterError(
            "Cannot stack memory when input data has "
            "no columns. Given data.shape={}".format(data.shape)
        )
    kwargs.setdefault("mode", "constant")

    if kwargs["mode"] == "constant":
        kwargs.setdefault("constant_values", [0])

    padding = [(0, 0) for _ in range(data.ndim)]

    # Pad the end with zeros, which will roll to the front below
    if delay > 0:
        padding[-1] = (int((n_steps - 1) * delay), 0)
    else:
        padding[-1] = (0, int((n_steps - 1) * -delay))

    data = np.pad(data, padding, **kwargs)

    # Construct the shape of the target array
    shape = list(data.shape)
    shape[-2] = shape[-2] * n_steps
    shape[-1] = t
    shape = tuple(shape)

    # Construct the output array to match layout and dtype of input
    history = np.empty_like(data, shape=shape)

    # Populate the output array
    __stack(history, data, n_steps, delay)

    return history


@jit(nopython=True, cache=True)
def __stack(history, data, n_steps, delay):
    """Memory-stacking helper function.

    Parameters
    ----------
    history : output array (2-dimensional)
    data : pre-padded input array (2-dimensional)
    n_steps : int > 0, the number of steps to stack
    delay : int != 0, the amount of delay between steps

    Returns
    -------
    None
        Output is stored directly in the history array
    """
    # Dimension of each copy of the data
    d = data.shape[-2]

    # Total number of time-steps to output
    t = history.shape[-1]

    if delay > 0:
        for step in range(n_steps):
            q = n_steps - 1 - step
            # nth block is original shifted left by n*delay steps
            history[..., step * d : (step + 1) * d, :] = data[
                ..., q * delay : q * delay + t
            ]
    else:
        # Handle the last block separately to avoid -t:0 empty slices
        history[..., -d:, :] = data[..., -t:]

        for step in range(n_steps - 1):
            # nth block is original shifted right by n*delay steps
            q = n_steps - 1 - step
            history[..., step * d : (step + 1) * d, :] = data[
                ..., -t + q * delay : q * delay
            ]
