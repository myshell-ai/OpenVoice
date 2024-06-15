#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core IO, DSP and utility functions."""

import pathlib
import warnings

import soundfile as sf
import audioread
import numpy as np
import scipy.signal
import resampy

from numba import jit
from .fft import get_fftlib
from .convert import frames_to_samples, time_to_samples
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..util.decorators import deprecate_positional_args

__all__ = [
    "load",
    "stream",
    "to_mono",
    "resample",
    "get_duration",
    "get_samplerate",
    "autocorrelate",
    "lpc",
    "zero_crossings",
    "clicks",
    "tone",
    "chirp",
    "mu_compress",
    "mu_expand",
]

# Resampling bandwidths as percentage of Nyquist
BW_BEST = resampy.filters.get_filter("kaiser_best")[2]
BW_FASTEST = resampy.filters.get_filter("kaiser_fast")[2]


# -- CORE ROUTINES --#
# Load should never be cached, since we cannot verify that the contents of
# 'path' are unchanged across calls.
@deprecate_positional_args
def load(
    path,
    *,
    sr=22050,
    mono=True,
    offset=0.0,
    duration=None,
    dtype=np.float32,
    res_type="kaiser_best",
):
    """Load an audio file as a floating point time series.

    Audio will be automatically resampled to the given rate
    (default ``sr=22050``).

    To preserve the native sampling rate of the file, use ``sr=None``.

    Parameters
    ----------
    path : string, int, pathlib.Path, soundfile.SoundFile or file-like object
        path to the input file.

        Any codec supported by `soundfile` or `audioread` will work.

        Any string file paths, or any object implementing Python's
        file interface (e.g. `pathlib.Path`) are supported as `path`.

        If the codec is supported by `soundfile`, then `path` can also be
        an open file descriptor (int) or an existing `soundfile.SoundFile` object.

        On the contrary, if the codec is not supported by `soundfile`
        (for example, MP3), then `path` must be a file path (string or `pathlib.Path`).

    sr : number > 0 [scalar]
        target sampling rate

        'None' uses the native sampling rate

    mono : bool
        convert signal to mono

    offset : float
        start reading after this time (in seconds)

    duration : float
        only load up to this much audio (in seconds)

    dtype : numeric type
        data type of ``y``

    res_type : str
        resample type (see note)

        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').

            For alternative resampling modes, see `resample`

        .. note::
           `audioread` may truncate the precision of the audio data to 16 bits.

           See :ref:`ioformats` for alternate loading methods.

    Returns
    -------
    y : np.ndarray [shape=(n,) or (..., n)]
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``

    Examples
    --------
    >>> # Load an ogg vorbis file
    >>> filename = librosa.ex('trumpet')
    >>> y, sr = librosa.load(filename)
    >>> y
    array([-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05],
          dtype=float32)
    >>> sr
    22050

    >>> # Load a file and resample to 11 KHz
    >>> filename = librosa.ex('trumpet')
    >>> y, sr = librosa.load(filename, sr=11025)
    >>> y
    array([-8.746e-04, -3.363e-04, ..., -1.301e-05,  0.000e+00],
          dtype=float32)
    >>> sr
    11025

    >>> # Load 5 seconds of a file, starting 15 seconds in
    >>> filename = librosa.ex('brahms')
    >>> y, sr = librosa.load(filename, offset=15.0, duration=5.0)
    >>> y
    array([0.146, 0.144, ..., 0.128, 0.015], dtype=float32)
    >>> sr
    22050

    """

    try:
        if isinstance(path, sf.SoundFile):
            # If the user passed an existing soundfile object,
            # we can use it directly
            context = path
        else:
            # Otherwise, create the soundfile object
            context = sf.SoundFile(path)

        with context as sf_desc:
            sr_native = sf_desc.samplerate
            if offset:
                # Seek to the start of the target read
                sf_desc.seek(int(offset * sr_native))
            if duration is not None:
                frame_duration = int(duration * sr_native)
            else:
                frame_duration = -1

            # Load the target number of frames, and transpose to match librosa form
            y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T

    except RuntimeError as exc:
        # If soundfile failed, try audioread instead
        if isinstance(path, (str, pathlib.PurePath)):
            warnings.warn("PySoundFile failed. Trying audioread instead.", stacklevel=2)
            y, sr_native = __audioread_load(path, offset, duration, dtype)
        else:
            raise (exc)

    # Final cleanup for dtype and contiguity
    if mono:
        y = to_mono(y)

    if sr is not None:
        y = resample(y, orig_sr=sr_native, target_sr=sr, res_type=res_type)

    else:
        sr = sr_native

    return y, sr


def __audioread_load(path, offset, duration, dtype):
    """Load an audio buffer using audioread.

    This loads one block at a time, and then concatenates the results.
    """

    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev) :]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native


@deprecate_positional_args
def stream(
    path,
    *,
    block_length,
    frame_length,
    hop_length,
    mono=True,
    offset=0.0,
    duration=None,
    fill_value=None,
    dtype=np.float32,
):
    """Stream audio in fixed-length buffers.

    This is primarily useful for processing large files that won't
    fit entirely in memory at once.

    Instead of loading the entire audio signal into memory (as
    in `load`, this function produces *blocks* of audio spanning
    a fixed number of frames at a specified frame length and hop
    length.

    While this function strives for similar behavior to `load`,
    there are a few caveats that users should be aware of:

        1. This function does not return audio buffers directly.
           It returns a generator, which you can iterate over
           to produce blocks of audio.  A *block*, in this context,
           refers to a buffer of audio which spans a given number of
           (potentially overlapping) frames.
        2. Automatic sample-rate conversion is not supported.
           Audio will be streamed in its native sample rate,
           so no default values are provided for ``frame_length``
           and ``hop_length``.  It is recommended that you first
           get the sampling rate for the file in question, using
           `get_samplerate`, and set these parameters accordingly.
        3. Many analyses require access to the entire signal
           to behave correctly, such as `resample`, `cqt`, or
           `beat_track`, so these methods will not be appropriate
           for streamed data.
        4. The ``block_length`` parameter specifies how many frames
           of audio will be produced per block.  Larger values will
           consume more memory, but will be more efficient to process
           down-stream.  The best value will ultimately depend on your
           application and other system constraints.
        5. By default, most librosa analyses (e.g., short-time Fourier
           transform) assume centered frames, which requires padding the
           signal at the beginning and end.  This will not work correctly
           when the signal is carved into blocks, because it would introduce
           padding in the middle of the signal.  To disable this feature,
           use ``center=False`` in all frame-based analyses.

    See the examples below for proper usage of this function.

    Parameters
    ----------
    path : string, int, sf.SoundFile, or file-like object
        path to the input file to stream.

        Any codec supported by `soundfile` is permitted here.

        An existing `soundfile.SoundFile` object may also be provided.

    block_length : int > 0
        The number of frames to include in each block.

        Note that at the end of the file, there may not be enough
        data to fill an entire block, resulting in a shorter block
        by default.  To pad the signal out so that blocks are always
        full length, set ``fill_value`` (see below).

    frame_length : int > 0
        The number of samples per frame.

    hop_length : int > 0
        The number of samples to advance between frames.

        Note that by when ``hop_length < frame_length``, neighboring frames
        will overlap.  Similarly, the last frame of one *block* will overlap
        with the first frame of the next *block*.

    mono : bool
        Convert the signal to mono during streaming

    offset : float
        Start reading after this time (in seconds)

    duration : float
        Only load up to this much audio (in seconds)

    fill_value : float [optional]
        If padding the signal to produce constant-length blocks,
        this value will be used at the end of the signal.

        In most cases, ``fill_value=0`` (silence) is expected, but
        you may specify any value here.

    dtype : numeric type
        data type of audio buffers to be produced

    Yields
    ------
    y : np.ndarray
        An audio buffer of (at most)
        ``(block_length-1) * hop_length + frame_length`` samples.

    See Also
    --------
    load
    get_samplerate
    soundfile.blocks

    Examples
    --------
    Apply a short-term Fourier transform to blocks of 256 frames
    at a time.  Note that streaming operation requires left-aligned
    frames, so we must set ``center=False`` to avoid padding artifacts.

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename,
    ...                       block_length=256,
    ...                       frame_length=4096,
    ...                       hop_length=1024)
    >>> for y_block in stream:
    ...     D_block = librosa.stft(y_block, center=False)

    Or compute a mel spectrogram over a stream, using a shorter frame
    and non-overlapping windows

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename,
    ...                         block_length=256,
    ...                         frame_length=2048,
    ...                         hop_length=2048)
    >>> for y_block in stream:
    ...     m_block = librosa.feature.melspectrogram(y=y_block, sr=sr,
    ...                                              n_fft=2048,
    ...                                              hop_length=2048,
    ...                                              center=False)

    """

    if not (np.issubdtype(type(block_length), np.integer) and block_length > 0):
        raise ParameterError("block_length={} must be a positive integer")
    if not (np.issubdtype(type(frame_length), np.integer) and frame_length > 0):
        raise ParameterError("frame_length={} must be a positive integer")
    if not (np.issubdtype(type(hop_length), np.integer) and hop_length > 0):
        raise ParameterError("hop_length={} must be a positive integer")

    if isinstance(path, sf.SoundFile):
        sfo = path
    else:
        sfo = sf.SoundFile(path)

    # Get the sample rate from the file info
    sr = sfo.samplerate

    # Construct the stream
    if offset:
        start = int(offset * sr)
    else:
        start = 0

    if duration:
        frames = int(duration * sr)
    else:
        frames = -1

    # Seek the soundfile object to the starting frame
    sfo.seek(start)

    blocks = sfo.blocks(
        blocksize=frame_length + (block_length - 1) * hop_length,
        overlap=frame_length - hop_length,
        frames=frames,
        dtype=dtype,
        always_2d=False,
        fill_value=fill_value,
    )

    for block in blocks:
        if mono:
            yield to_mono(block.T)
        else:
            yield block.T


@cache(level=20)
def to_mono(y):
    """Convert an audio signal to mono by averaging samples across channels.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.

    Returns
    -------
    y_mono : np.ndarray [shape=(n,)]
        ``y`` as a monophonic time-series

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> y_mono = librosa.to_mono(y)
    >>> y_mono.shape
    (117601,)

    """

    # Validate the buffer.  Stereo is ok here.
    util.valid_audio(y, mono=False)

    if y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim - 1)))

    return y


@deprecate_positional_args
@cache(level=20)
def resample(
    y, *, orig_sr, target_sr, res_type="kaiser_best", fix=True, scale=False, **kwargs
):
    """Resample a time series from orig_sr to target_sr

    By default, this uses a high-quality (but relatively slow) method ('kaiser_best')
    for band-limited sinc interpolation.  The alternate ``res_type`` values listed below
    offer different trade-offs of speed and quality.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series.  Multi-channel is supported.

    orig_sr : number > 0 [scalar]
        original sampling rate of ``y``

    target_sr : number > 0 [scalar]
        target sampling rate

    res_type : str
        resample type

        'kaiser_best' (default)
            `resampy` high-quality mode
        'kaiser_fast'
            `resampy` faster method
        'fft' or 'scipy'
            `scipy.signal.resample` Fourier method.
        'polyphase'
            `scipy.signal.resample_poly` polyphase filtering. (fast)
        'linear'
            `samplerate` linear interpolation. (very fast)
        'zero_order_hold'
            `samplerate` repeat the last value between samples. (very fast)
        'sinc_best', 'sinc_medium' or 'sinc_fastest'
            `samplerate` high-, medium-, and low-quality sinc interpolation.
        'soxr_vhq', 'soxr_hq', 'soxr_mq' or 'soxr_lq'
            `soxr` Very high-, High-, Medium-, Low-quality FFT-based bandlimited interpolation.
            ``'soxr_hq'`` is the default setting of `soxr` (fast)
        'soxr_qq'
            `soxr` Quick cubic interpolation (very fast)

        .. note::
            `samplerate` and `soxr` are not installed with `librosa`.
            To use `samplerate` or `soxr`, they should be installed manually::

                $ pip install samplerate
                $ pip install soxr

        .. note::
            When using ``res_type='polyphase'``, only integer sampling rates are
            supported.

    fix : bool
        adjust the length of the resampled signal to be of size exactly
        ``ceil(target_sr * len(y) / orig_sr)``

    scale : bool
        Scale the resampled signal so that ``y`` and ``y_hat`` have approximately
        equal total energy.

    **kwargs : additional keyword arguments
        If ``fix==True``, additional keyword arguments to pass to
        `librosa.util.fix_length`.

    Returns
    -------
    y_hat : np.ndarray [shape=(..., n * target_sr / orig_sr)]
        ``y`` resampled from ``orig_sr`` to ``target_sr``

    Raises
    ------
    ParameterError
        If ``res_type='polyphase'`` and ``orig_sr`` or ``target_sr`` are not both
        integer-valued.

    See Also
    --------
    librosa.util.fix_length
    scipy.signal.resample
    resampy
    samplerate.converters.resample
    soxr.resample

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Downsample from 22 KHz to 8 KHz

    >>> y, sr = librosa.load(librosa.ex('trumpet'), sr=22050)
    >>> y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
    >>> y.shape, y_8k.shape
    ((117601,), (42668,))
    """

    # First, validate the audio buffer
    util.valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    if res_type in ("scipy", "fft"):
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    elif res_type == "polyphase":
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            raise ParameterError(
                "polyphase resampling is only supported for integer-valued sampling rates."
            )

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd, axis=-1)
    elif res_type in (
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
    ):
        import samplerate

        # We have to transpose here to match libsamplerate
        y_hat = samplerate.resample(y.T, ratio, converter_type=res_type).T
    elif res_type.startswith("soxr"):
        import soxr

        # We have to transpose here to match soxr
        y_hat = soxr.resample(y.T, orig_sr, target_sr, quality=res_type).T
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = util.fix_length(y_hat, size=n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return y_hat.astype(y.dtype)


@deprecate_positional_args
def get_duration(
    *, y=None, sr=22050, S=None, n_fft=2048, hop_length=512, center=True, filename=None
):
    """Compute the duration (in seconds) of an audio time series,
    feature matrix, or filename.

    Examples
    --------
    >>> # Load an example audio file
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.get_duration(y=y, sr=sr)
    5.333378684807256

    >>> # Or directly from an audio file
    >>> librosa.get_duration(filename=librosa.ex('trumpet'))
    5.333378684807256

    >>> # Or compute duration from an STFT matrix
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = librosa.stft(y)
    >>> librosa.get_duration(S=S, sr=sr)
    5.317369614512471

    >>> # Or a non-centered STFT matrix
    >>> S_left = librosa.stft(y, center=False)
    >>> librosa.get_duration(S=S_left, sr=sr)
    5.224489795918367

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        audio sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        STFT matrix, or any STFT-derived matrix (e.g., chromagram
        or mel spectrogram).
        Durations calculated from spectrogram inputs are only accurate
        up to the frame resolution. If high precision is required,
        it is better to use the audio time series directly.

    n_fft : int > 0 [scalar]
        FFT window size for ``S``

    hop_length : int > 0 [ scalar]
        number of audio samples between columns of ``S``

    center : boolean
        - If ``True``, ``S[:, t]`` is centered at ``y[t * hop_length]``
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``

    filename : str
        If provided, all other parameters are ignored, and the
        duration is calculated directly from the audio file.
        Note that this avoids loading the contents into memory,
        and is therefore useful for querying the duration of
        long files.

        As in ``load``, this can also be an integer or open file-handle
        that can be processed by ``soundfile``.

    Returns
    -------
    d : float >= 0
        Duration (in seconds) of the input time series or spectrogram.

    Raises
    ------
    ParameterError
        if none of ``y``, ``S``, or ``filename`` are provided.

    Notes
    -----
    `get_duration` can be applied to a file (``filename``), a spectrogram (``S``),
    or audio buffer (``y, sr``).  Only one of these three options should be
    provided.  If you do provide multiple options (e.g., ``filename`` and ``S``),
    then ``filename`` takes precedence over ``S``, and ``S`` takes precedence over
    ``(y, sr)``.
    """

    if filename is not None:
        try:
            return sf.info(filename).duration
        except RuntimeError:
            with audioread.audio_open(filename) as fdesc:
                return fdesc.duration

    if y is None:
        if S is None:
            raise ParameterError(
                "At least one of (y, sr), S, or filename must be provided"
            )

        n_frames = S.shape[-1]
        n_samples = n_fft + hop_length * (n_frames - 1)

        # If centered, we lose half a window from each end of S
        if center:
            n_samples = n_samples - 2 * int(n_fft // 2)

    else:
        n_samples = y.shape[-1]

    return float(n_samples) / sr


def get_samplerate(path):
    """Get the sampling rate for a given file.

    Parameters
    ----------
    path : string, int, soundfile.SoundFile, or file-like
        The path to the file to be loaded
        As in ``load``, this can also be an integer or open file-handle
        that can be processed by `soundfile`.
        An existing `soundfile.SoundFile` object can also be supplied.

    Returns
    -------
    sr : number > 0
        The sampling rate of the given audio file

    Examples
    --------
    Get the sampling rate for the included audio file

    >>> path = librosa.ex('trumpet')
    >>> librosa.get_samplerate(path)
    22050
    """
    try:
        if isinstance(path, sf.SoundFile):
            return path.samplerate

        return sf.info(path).samplerate
    except RuntimeError:
        with audioread.audio_open(path) as fdesc:
            return fdesc.samplerate


@deprecate_positional_args
@cache(level=20)
def autocorrelate(y, *, max_size=None, axis=-1):
    """Bounded-lag auto-correlation

    Parameters
    ----------
    y : np.ndarray
        array to autocorrelate
    max_size : int > 0 or None
        maximum correlation lag.
        If unspecified, defaults to ``y.shape[axis]`` (unbounded)
    axis : int
        The axis along which to autocorrelate.
        By default, the last axis (-1) is taken.

    Returns
    -------
    z : np.ndarray
        truncated autocorrelation ``y*y`` along the specified axis.
        If ``max_size`` is specified, then ``z.shape[axis]`` is bounded
        to ``max_size``.

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Compute full autocorrelation of ``y``

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.autocorrelate(y)
    array([ 6.899e+02,  6.236e+02, ...,  3.710e-08, -1.796e-08])

    Compute onset strength auto-correlation up to 4 seconds

    >>> import matplotlib.pyplot as plt
    >>> odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    >>> ac = librosa.autocorrelate(odf, max_size=4 * sr // 512)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(ac)
    >>> ax.set(title='Auto-correlation', xlabel='Lag (frames)')
    """

    if max_size is None:
        max_size = y.shape[axis]

    max_size = int(min(max_size, y.shape[axis]))

    # Compute the power spectrum along the chosen axis
    # Pad out the signal to support full-length auto-correlation.
    fft = get_fftlib()
    powspec = np.abs(fft.fft(y, n=2 * y.shape[axis] + 1, axis=axis)) ** 2

    # Convert back to time domain
    autocorr = fft.ifft(powspec, axis=axis)

    # Slice down to max_size
    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)

    autocorr = autocorr[tuple(subslice)]

    if not np.iscomplexobj(y):
        autocorr = autocorr.real

    return autocorr


@deprecate_positional_args
def lpc(y, *, order, axis=-1):
    """Linear Prediction Coefficients via Burg's method

    This function applies Burg's method to estimate coefficients of a linear
    filter on ``y`` of order ``order``.  Burg's method is an extension to the
    Yule-Walker approach, which are both sometimes referred to as LPC parameter
    estimation by autocorrelation.

    It follows the description and implementation approach described in the
    introduction by Marple. [#]_  N.B. This paper describes a different method, which
    is not implemented here, but has been chosen for its clear explanation of
    Burg's technique in its introduction.

    .. [#] Larry Marple.
           A New Autoregressive Spectrum Analysis Algorithm.
           IEEE Transactions on Accoustics, Speech, and Signal Processing
           vol 28, no. 4, 1980.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        Time series to fit. Multi-channel is supported..
    order : int > 0
        Order of the linear filter
    axis : int
        Axis along which to compute the coefficients

    Returns
    -------
    a : np.ndarray [shape=(..., order + 1)]
        LP prediction error coefficients, i.e. filter denominator polynomial.
        Note that the length along the specified ``axis`` will be ``order+1``.

    Raises
    ------
    ParameterError
        - If ``y`` is not valid audio as per `librosa.util.valid_audio`
        - If ``order < 1`` or not integer
    FloatingPointError
        - If ``y`` is ill-conditioned

    See Also
    --------
    scipy.signal.lfilter

    Examples
    --------
    Compute LP coefficients of y at order 16 on entire series

    >>> y, sr = librosa.load(librosa.ex('libri1'))
    >>> librosa.lpc(y, order=16)

    Compute LP coefficients, and plot LP estimate of original series

    >>> import matplotlib.pyplot as plt
    >>> import scipy
    >>> y, sr = librosa.load(librosa.ex('libri1'), duration=0.020)
    >>> a = librosa.lpc(y, order=2)
    >>> b = np.hstack([[0], -1 * a[1:]])
    >>> y_hat = scipy.signal.lfilter(b, [1], y)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(y)
    >>> ax.plot(y_hat, linestyle='--')
    >>> ax.legend(['y', 'y_hat'])
    >>> ax.set_title('LP Model Forward Prediction')

    """
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ParameterError("order must be an integer > 0")

    util.valid_audio(y, mono=False)

    # Move the lpc axis around front, because numba is silly
    y = y.swapaxes(axis, 0)

    dtype = y.dtype

    shape = list(y.shape)
    shape[0] = order + 1

    ar_coeffs = np.zeros(tuple(shape), dtype=dtype)
    ar_coeffs[0] = 1

    ar_coeffs_prev = ar_coeffs.copy()

    shape[0] = 1
    reflect_coeff = np.zeros(shape, dtype=dtype)
    den = reflect_coeff.copy()

    epsilon = util.tiny(den)

    # Call the helper, and swap the results back to the target axis position
    return __lpc(
        y, order, ar_coeffs, ar_coeffs_prev, reflect_coeff, den, epsilon
    ).swapaxes(0, axis)


@jit(nopython=True, cache=True)
def __lpc(y, order, ar_coeffs, ar_coeffs_prev, reflect_coeff, den, epsilon):
    # This implementation follows the description of Burg's algorithm given in
    # section III of Marple's paper referenced in the docstring.
    #
    # We use the Levinson-Durbin recursion to compute AR coefficients for each
    # increasing model order by using those from the last. We maintain two
    # arrays and then flip them each time we increase the model order so that
    # we may use all the coefficients from the previous order while we compute
    # those for the new one. These two arrays hold ar_coeffs for order M and
    # order M-1.  (Corresponding to a_{M,k} and a_{M-1,k} in eqn 5)

    # These two arrays hold the forward and backward prediction error. They
    # correspond to f_{M-1,k} and b_{M-1,k} in eqns 10, 11, 13 and 14 of
    # Marple. First they are used to compute the reflection coefficient at
    # order M from M-1 then are re-used as f_{M,k} and b_{M,k} for each
    # iteration of the below loop
    fwd_pred_error = y[1:]
    bwd_pred_error = y[:-1]

    # DEN_{M} from eqn 16 of Marple.
    den[0] = np.sum(fwd_pred_error ** 2 + bwd_pred_error ** 2, axis=0)

    for i in range(order):
        # can be removed if we keep the epsilon bias
        # if np.any(den <= 0):
        #    raise FloatingPointError("numerical error, input ill-conditioned?")

        # Eqn 15 of Marple, with fwd_pred_error and bwd_pred_error
        # corresponding to f_{M-1,k+1} and b{M-1,k} and the result as a_{M,M}

        reflect_coeff[0] = np.sum(bwd_pred_error * fwd_pred_error, axis=0)
        reflect_coeff[0] *= -2
        reflect_coeff[0] /= den[0] + epsilon

        # Now we use the reflection coefficient and the AR coefficients from
        # the last model order to compute all of the AR coefficients for the
        # current one.  This is the Levinson-Durbin recursion described in
        # eqn 5.
        # Note 1: We don't have to care about complex conjugates as our signals
        # are all real-valued
        # Note 2: j counts 1..order+1, i-j+1 counts order..0
        # Note 3: The first element of ar_coeffs* is always 1, which copies in
        # the reflection coefficient at the end of the new AR coefficient array
        # after the preceding coefficients

        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            # reflection multiply should be broadcast
            ar_coeffs[j] = (
                ar_coeffs_prev[j] + reflect_coeff[0] * ar_coeffs_prev[i - j + 1]
            )

        # Update the forward and backward prediction errors corresponding to
        # eqns 13 and 14.  We start with f_{M-1,k+1} and b_{M-1,k} and use them
        # to compute f_{M,k} and b_{M,k}
        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

        # SNIP - we are now done with order M and advance. M-1 <- M

        # Compute DEN_{M} using the recursion from eqn 17.
        #
        # reflect_coeff = a_{M-1,M-1}      (we have advanced M)
        # den =  DEN_{M-1}                 (rhs)
        # bwd_pred_error = b_{M-1,N-M+1}   (we have advanced M)
        # fwd_pred_error = f_{M-1,k}       (we have advanced M)
        # den <- DEN_{M}                   (lhs)
        #

        q = 1.0 - reflect_coeff[0] ** 2
        den[0] = q * den[0] - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2

        # Shift up forward error.
        #
        # fwd_pred_error <- f_{M-1,k+1}
        # bwd_pred_error <- b_{M-1,k}
        #
        # N.B. We do this after computing the denominator using eqn 17 but
        # before using it in the numerator in eqn 15.
        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]

    return ar_coeffs


@deprecate_positional_args
@cache(level=20)
def zero_crossings(
    y, *, threshold=1e-10, ref_magnitude=None, pad=True, zero_pos=True, axis=-1
):
    """Find the zero-crossings of a signal ``y``: indices ``i`` such that
    ``sign(y[i]) != sign(y[j])``.

    If ``y`` is multi-dimensional, then zero-crossings are computed along
    the specified ``axis``.

    Parameters
    ----------
    y : np.ndarray
        The input array

    threshold : float > 0 or None
        If specified, values where ``-threshold <= y <= threshold`` are
        clipped to 0.

    ref_magnitude : float > 0 or callable
        If numeric, the threshold is scaled relative to ``ref_magnitude``.

        If callable, the threshold is scaled relative to
        ``ref_magnitude(np.abs(y))``.

    pad : boolean
        If ``True``, then ``y[0]`` is considered a valid zero-crossing.

    zero_pos : boolean
        If ``True`` then the value 0 is interpreted as having positive sign.

        If ``False``, then 0, -1, and +1 all have distinct signs.

    axis : int
        Axis along which to compute zero-crossings.

    Returns
    -------
    zero_crossings : np.ndarray [shape=y.shape, dtype=boolean]
        Indicator array of zero-crossings in ``y`` along the selected axis.

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    >>> # Generate a time-series
    >>> y = np.sin(np.linspace(0, 4 * 2 * np.pi, 20))
    >>> y
    array([  0.000e+00,   9.694e-01,   4.759e-01,  -7.357e-01,
            -8.372e-01,   3.247e-01,   9.966e-01,   1.646e-01,
            -9.158e-01,  -6.142e-01,   6.142e-01,   9.158e-01,
            -1.646e-01,  -9.966e-01,  -3.247e-01,   8.372e-01,
             7.357e-01,  -4.759e-01,  -9.694e-01,  -9.797e-16])
    >>> # Compute zero-crossings
    >>> z = librosa.zero_crossings(y)
    >>> z
    array([ True, False, False,  True, False,  True, False, False,
            True, False,  True, False,  True, False, False,  True,
           False,  True, False,  True], dtype=bool)

    >>> # Stack y against the zero-crossing indicator
    >>> librosa.util.stack([y, z], axis=-1)
    array([[  0.000e+00,   1.000e+00],
           [  9.694e-01,   0.000e+00],
           [  4.759e-01,   0.000e+00],
           [ -7.357e-01,   1.000e+00],
           [ -8.372e-01,   0.000e+00],
           [  3.247e-01,   1.000e+00],
           [  9.966e-01,   0.000e+00],
           [  1.646e-01,   0.000e+00],
           [ -9.158e-01,   1.000e+00],
           [ -6.142e-01,   0.000e+00],
           [  6.142e-01,   1.000e+00],
           [  9.158e-01,   0.000e+00],
           [ -1.646e-01,   1.000e+00],
           [ -9.966e-01,   0.000e+00],
           [ -3.247e-01,   0.000e+00],
           [  8.372e-01,   1.000e+00],
           [  7.357e-01,   0.000e+00],
           [ -4.759e-01,   1.000e+00],
           [ -9.694e-01,   0.000e+00],
           [ -9.797e-16,   1.000e+00]])

    >>> # Find the indices of zero-crossings
    >>> np.nonzero(z)
    (array([ 0,  3,  5,  8, 10, 12, 15, 17, 19]),)
    """

    # Clip within the threshold
    if threshold is None:
        threshold = 0.0

    if callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(y))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    if threshold > 0:
        y = y.copy()
        y[np.abs(y) <= threshold] = 0

    # Extract the sign bit
    if zero_pos:
        y_sign = np.signbit(y)
    else:
        y_sign = np.sign(y)

    # Find the change-points by slicing
    slice_pre = [slice(None)] * y.ndim
    slice_pre[axis] = slice(1, None)

    slice_post = [slice(None)] * y.ndim
    slice_post[axis] = slice(-1)

    # Since we've offset the input by one, pad back onto the front
    padding = [(0, 0)] * y.ndim
    padding[axis] = (1, 0)

    return np.pad(
        (y_sign[tuple(slice_post)] != y_sign[tuple(slice_pre)]),
        padding,
        mode="constant",
        constant_values=pad,
    )


@deprecate_positional_args
def clicks(
    *,
    times=None,
    frames=None,
    sr=22050,
    hop_length=512,
    click_freq=1000.0,
    click_duration=0.1,
    click=None,
    length=None,
):
    """Construct a "click track".

    This returns a signal with the signal ``click`` sound placed at
    each specified time.

    Parameters
    ----------
    times : np.ndarray or None
        times to place clicks, in seconds
    frames : np.ndarray or None
        frame indices to place clicks
    sr : number > 0
        desired sampling rate of the output signal
    hop_length : int > 0
        if positions are specified by ``frames``, the number of samples between frames.
    click_freq : float > 0
        frequency (in Hz) of the default click signal.  Default is 1KHz.
    click_duration : float > 0
        duration (in seconds) of the default click signal.  Default is 100ms.
    click : np.ndarray or None
        (optional) click signal sample to use instead of the default click.
        Multi-channel is supported.
    length : int > 0
        desired number of samples in the output signal

    Returns
    -------
    click_signal : np.ndarray
        Synthesized click signal.
        This will be monophonic by default, or match the number of channels to a provided ``click`` signal.

    Raises
    ------
    ParameterError
        - If neither ``times`` nor ``frames`` are provided.
        - If any of ``click_freq``, ``click_duration``, or ``length`` are out of range.

    Examples
    --------
    >>> # Sonify detected beat events
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> y_beats = librosa.clicks(frames=beats, sr=sr)

    >>> # Or generate a signal of the same length as y
    >>> y_beats = librosa.clicks(frames=beats, sr=sr, length=len(y))

    >>> # Or use timing instead of frame indices
    >>> times = librosa.frames_to_time(beats, sr=sr)
    >>> y_beat_times = librosa.clicks(times=times, sr=sr)

    >>> # Or with a click frequency of 880Hz and a 500ms sample
    >>> y_beat_times880 = librosa.clicks(times=times, sr=sr,
    ...                                  click_freq=880, click_duration=0.5)

    Display click waveform next to the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr)
    >>> librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
    ...                          x_axis='time', y_axis='mel', ax=ax[0])
    >>> librosa.display.waveshow(y_beat_times, sr=sr, label='Beat clicks',
    ...                          ax=ax[1])
    >>> ax[1].legend()
    >>> ax[0].label_outer()
    >>> ax[0].set_title(None)
    """

    # Compute sample positions from time or frames
    if times is None:
        if frames is None:
            raise ParameterError('either "times" or "frames" must be provided')

        positions = frames_to_samples(frames, hop_length=hop_length)
    else:
        # Convert times to positions
        positions = time_to_samples(times, sr=sr)

    if click is not None:
        # Check that we have a well-formed audio buffer
        util.valid_audio(click, mono=False)

    else:
        # Create default click signal
        if click_duration <= 0:
            raise ParameterError("click_duration must be strictly positive")

        if click_freq <= 0:
            raise ParameterError("click_freq must be strictly positive")

        angular_freq = 2 * np.pi * click_freq / float(sr)

        click = np.logspace(0, -10, num=int(np.round(sr * click_duration)), base=2.0)

        click *= np.sin(angular_freq * np.arange(len(click)))

    # Set default length
    if length is None:
        length = positions.max() + click.shape[-1]
    else:
        if length < 1:
            raise ParameterError("length must be a positive integer")

        # Filter out any positions past the length boundary
        positions = positions[positions < length]

    # Pre-allocate click signal
    shape = list(click.shape)
    shape[-1] = length
    click_signal = np.zeros(shape, dtype=np.float32)

    # Place clicks
    for start in positions:
        # Compute the end-point of this click
        end = start + click.shape[-1]

        if end >= length:
            click_signal[..., start:] += click[..., : length - start]
        else:
            # Normally, just add a click here
            click_signal[..., start:end] += click

    return click_signal


@deprecate_positional_args
def tone(frequency, *, sr=22050, length=None, duration=None, phi=None):
    """Construct a pure tone (cosine) signal at a given frequency.

    Parameters
    ----------
    frequency : float > 0
        frequency
    sr : number > 0
        desired sampling rate of the output signal
    length : int > 0
        desired number of samples in the output signal.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.
    duration : float > 0
        desired duration in seconds.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.
    phi : float or None
        phase offset, in radians. If unspecified, defaults to ``-np.pi * 0.5``.

    Returns
    -------
    tone_signal : np.ndarray [shape=(length,), dtype=float64]
        Synthesized pure sine tone signal

    Raises
    ------
    ParameterError
        - If ``frequency`` is not provided.
        - If neither ``length`` nor ``duration`` are provided.

    Examples
    --------
    Generate a pure sine tone A4

    >>> tone = librosa.tone(440, duration=1)

    Or generate the same signal using `length`

    >>> tone = librosa.tone(440, sr=22050, length=22050)

    Display spectrogram

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> S = librosa.feature.melspectrogram(y=tone)
    >>> librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
    ...                          x_axis='time', y_axis='mel', ax=ax)
    """

    if frequency is None:
        raise ParameterError('"frequency" must be provided')

    # Compute signal length
    if length is None:
        if duration is None:
            raise ParameterError('either "length" or "duration" must be provided')
        length = duration * sr

    if phi is None:
        phi = -np.pi * 0.5

    return np.cos(2 * np.pi * frequency * np.arange(length) / sr + phi)


@deprecate_positional_args
def chirp(*, fmin, fmax, sr=22050, length=None, duration=None, linear=False, phi=None):
    """Construct a "chirp" or "sine-sweep" signal.

    The chirp sweeps from frequency ``fmin`` to ``fmax`` (in Hz).

    Parameters
    ----------
    fmin : float > 0
        initial frequency

    fmax : float > 0
        final frequency

    sr : number > 0
        desired sampling rate of the output signal

    length : int > 0
        desired number of samples in the output signal.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.

    duration : float > 0
        desired duration in seconds.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.

    linear : boolean
        - If ``True``, use a linear sweep, i.e., frequency changes linearly with time
        - If ``False``, use a exponential sweep.

        Default is ``False``.

    phi : float or None
        phase offset, in radians.
        If unspecified, defaults to ``-np.pi * 0.5``.

    Returns
    -------
    chirp_signal : np.ndarray [shape=(length,), dtype=float64]
        Synthesized chirp signal

    Raises
    ------
    ParameterError
        - If either ``fmin`` or ``fmax`` are not provided.
        - If neither ``length`` nor ``duration`` are provided.

    See Also
    --------
    scipy.signal.chirp

    Examples
    --------
    Generate a exponential chirp from A2 to A8

    >>> exponential_chirp = librosa.chirp(fmin=110, fmax=110*64, duration=1)

    Or generate the same signal using ``length``

    >>> exponential_chirp = librosa.chirp(fmin=110, fmax=110*64, sr=22050, length=22050)

    Or generate a linear chirp instead

    >>> linear_chirp = librosa.chirp(fmin=110, fmax=110*64, duration=1, linear=True)

    Display spectrogram for both exponential and linear chirps.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> S_exponential = np.abs(librosa.stft(y=exponential_chirp))
    >>> librosa.display.specshow(librosa.amplitude_to_db(S_exponential, ref=np.max),
    ...                          x_axis='time', y_axis='linear', ax=ax[0])
    >>> ax[0].set(title='Exponential chirp', xlabel=None)
    >>> ax[0].label_outer()
    >>> S_linear = np.abs(librosa.stft(y=linear_chirp))
    >>> librosa.display.specshow(librosa.amplitude_to_db(S_linear, ref=np.max),
    ...                          x_axis='time', y_axis='linear', ax=ax[1])
    >>> ax[1].set(title='Linear chirp')
    """

    if fmin is None or fmax is None:
        raise ParameterError('both "fmin" and "fmax" must be provided')

    # Compute signal duration
    period = 1.0 / sr
    if length is None:
        if duration is None:
            raise ParameterError('either "length" or "duration" must be provided')
    else:
        duration = period * length

    if phi is None:
        phi = -np.pi * 0.5

    method = "linear" if linear else "logarithmic"
    return scipy.signal.chirp(
        np.arange(duration, step=period),
        fmin,
        duration,
        fmax,
        method=method,
        phi=phi / np.pi * 180,  # scipy.signal.chirp uses degrees for phase offset
    )


@deprecate_positional_args
def mu_compress(x, *, mu=255, quantize=True):
    """mu-law compression

    Given an input signal ``-1 <= x <= 1``, the mu-law compression
    is calculated by::

        sign(x) * ln(1 + mu * abs(x)) /  ln(1 + mu)

    Parameters
    ----------
    x : np.ndarray with values in [-1, +1]
        The input signal to compress

    mu : positive number
        The compression parameter.  Values of the form ``2**n - 1``
        (e.g., 15, 31, 63, etc.) are most common.

    quantize : bool
        If ``True``, quantize the compressed values into ``1 + mu``
        distinct integer values.

        If ``False``, mu-law compression is applied without quantization.

    Returns
    -------
    x_compressed : np.ndarray
        The compressed signal.

    Raises
    ------
    ParameterError
        If ``x`` has values outside the range [-1, +1]
        If ``mu <= 0``

    See Also
    --------
    mu_expand

    Examples
    --------
    Compression without quantization

    >>> x = np.linspace(-1, 1, num=16)
    >>> x
    array([-1.        , -0.86666667, -0.73333333, -0.6       , -0.46666667,
           -0.33333333, -0.2       , -0.06666667,  0.06666667,  0.2       ,
            0.33333333,  0.46666667,  0.6       ,  0.73333333,  0.86666667,
            1.        ])
    >>> y = librosa.mu_compress(x, quantize=False)
    >>> y
    array([-1.        , -0.97430198, -0.94432361, -0.90834832, -0.86336132,
           -0.80328309, -0.71255496, -0.52124063,  0.52124063,  0.71255496,
            0.80328309,  0.86336132,  0.90834832,  0.94432361,  0.97430198,
            1.        ])

    Compression with quantization

    >>> y = librosa.mu_compress(x, quantize=True)
    >>> y
    array([-128, -124, -120, -116, -110, -102,  -91,  -66,   66,   91,  102,
           110,  116,  120,  124,  127])

    Compression with quantization and a smaller range

    >>> y = librosa.mu_compress(x, mu=15, quantize=True)
    >>> y
    array([-8, -7, -7, -6, -6, -5, -4, -2,  2,  4,  5,  6,  6,  7,  7,  7])

    """

    if mu <= 0:
        raise ParameterError(
            "mu-law compression parameter mu={} "
            "must be strictly positive.".format(mu)
        )

    if np.any(x < -1) or np.any(x > 1):
        raise ParameterError(
            "mu-law input x={} must be in the " "range [-1, +1].".format(x)
        )

    x_comp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

    if quantize:
        return (
            np.digitize(
                x_comp, np.linspace(-1, 1, num=int(1 + mu), endpoint=True), right=True
            )
            - int(mu + 1) // 2
        )

    return x_comp


@deprecate_positional_args
def mu_expand(x, *, mu=255.0, quantize=True):
    """mu-law expansion

    This function is the inverse of ``mu_compress``. Given a mu-law compressed
    signal ``-1 <= x <= 1``, the mu-law expansion is calculated by::

        sign(x) * (1 / mu) * ((1 + mu)**abs(x) - 1)

    Parameters
    ----------
    x : np.ndarray
        The compressed signal.
        If ``quantize=True``, values must be in the range [-1, +1].
    mu : positive number
        The compression parameter.  Values of the form ``2**n - 1``
        (e.g., 15, 31, 63, etc.) are most common.
    quantize : boolean
        If ``True``, the input is assumed to be quantized to
        ``1 + mu`` distinct integer values.

    Returns
    -------
    x_expanded : np.ndarray with values in the range [-1, +1]
        The mu-law expanded signal.

    Raises
    ------
    ParameterError
        If ``x`` has values outside the range [-1, +1] and ``quantize=False``
        If ``mu <= 0``

    See Also
    --------
    mu_compress

    Examples
    --------
    Compress and expand without quantization

    >>> x = np.linspace(-1, 1, num=16)
    >>> x
    array([-1.        , -0.86666667, -0.73333333, -0.6       , -0.46666667,
           -0.33333333, -0.2       , -0.06666667,  0.06666667,  0.2       ,
            0.33333333,  0.46666667,  0.6       ,  0.73333333,  0.86666667,
            1.        ])
    >>> y = librosa.mu_compress(x, quantize=False)
    >>> y
    array([-1.        , -0.97430198, -0.94432361, -0.90834832, -0.86336132,
           -0.80328309, -0.71255496, -0.52124063,  0.52124063,  0.71255496,
            0.80328309,  0.86336132,  0.90834832,  0.94432361,  0.97430198,
            1.        ])
    >>> z = librosa.mu_expand(y, quantize=False)
    >>> z
    array([-1.        , -0.86666667, -0.73333333, -0.6       , -0.46666667,
           -0.33333333, -0.2       , -0.06666667,  0.06666667,  0.2       ,
            0.33333333,  0.46666667,  0.6       ,  0.73333333,  0.86666667,
            1.        ])

    Compress and expand with quantization.  Note that this necessarily
    incurs quantization error, particularly for values near +-1.

    >>> y = librosa.mu_compress(x, quantize=True)
    >>> y
    array([-128, -124, -120, -116, -110, -102,  -91,  -66,   66,   91,  102,
            110,  116,  120,  124,  127])
    >>> z = librosa.mu_expand(y, quantize=True)
    array([-1.        , -0.84027248, -0.70595818, -0.59301377, -0.4563785 ,
           -0.32155973, -0.19817918, -0.06450245,  0.06450245,  0.19817918,
            0.32155973,  0.4563785 ,  0.59301377,  0.70595818,  0.84027248,
            0.95743702])
    """
    if mu <= 0:
        raise ParameterError(
            "Inverse mu-law compression parameter "
            "mu={} must be strictly positive.".format(mu)
        )

    if quantize:
        x = x * 2.0 / (1 + mu)

    if np.any(x < -1) or np.any(x > 1):
        raise ParameterError(
            "Inverse mu-law input x={} must be " "in the range [-1, +1].".format(x)
        )

    return np.sign(x) / mu * (np.power(1 + mu, np.abs(x)) - 1)
