#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Display
=======

Data visualization
------------------
.. autosummary::
    :toctree: generated/

    specshow
    waveshow

Axis formatting
---------------
.. autosummary::
    :toctree: generated/

    TimeFormatter
    NoteFormatter
    SvaraFormatter
    LogHzFormatter
    ChromaFormatter
    ChromaSvaraFormatter
    TonnetzFormatter

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    cmap
    AdaptiveWaveplot

"""

import warnings

import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
from matplotlib.ticker import Formatter, ScalarFormatter
from matplotlib.ticker import LogLocator, FixedLocator, MaxNLocator
from matplotlib.ticker import SymmetricalLogLocator
import matplotlib
from packaging.version import parse as version_parse


from . import core
from . import util
from .util.exceptions import ParameterError
from .util.decorators import deprecate_positional_args

__all__ = [
    "specshow",
    "waveshow",
    "cmap",
    "TimeFormatter",
    "NoteFormatter",
    "LogHzFormatter",
    "ChromaFormatter",
    "TonnetzFormatter",
    "AdaptiveWaveplot",
]


class TimeFormatter(Formatter):
    """A tick formatter for time axes.

    Automatically switches between seconds, minutes:seconds,
    or hours:minutes:seconds.

    Parameters
    ----------
    lag : bool
        If ``True``, then the time axis is interpreted in lag coordinates.
        Anything past the midpoint will be converted to negative time.

    unit : str or None
        Abbreviation of the physical unit for axis labels and ticks.
        Either equal to `s` (seconds) or `ms` (milliseconds) or None (default).
        If set to None, the resulting TimeFormatter object adapts its string
        representation to the duration of the underlying time range:
        `hh:mm:ss` above 3600 seconds; `mm:ss` between 60 and 3600 seconds;
        and `ss` below 60 seconds.


    See also
    --------
    matplotlib.ticker.Formatter


    Examples
    --------

    For normal time

    >>> import matplotlib.pyplot as plt
    >>> times = np.arange(30)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    >>> ax.set(xlabel='Time')

    Manually set the physical time unit of the x-axis to milliseconds

    >>> times = np.arange(100)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(unit='ms'))
    >>> ax.set(xlabel='Time (ms)')

    For lag plots

    >>> times = np.arange(60)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(lag=True))
    >>> ax.set(xlabel='Lag')
    """

    def __init__(self, lag=False, unit=None):

        if unit not in ["s", "ms", None]:
            raise ParameterError("Unknown time unit: {}".format(unit))

        self.unit = unit
        self.lag = lag

    def __call__(self, x, pos=None):
        """Return the time format as pos"""

        _, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()

        # In lag-time axes, anything greater than dmax / 2 is negative time
        if self.lag and x >= dmax * 0.5:
            # In lag mode, don't tick past the limits of the data
            if x > dmax:
                return ""
            value = np.abs(x - dmax)
            # Do we need to tweak vmin/vmax here?
            sign = "-"
        else:
            value = x
            sign = ""

        if self.unit == "s":
            s = "{:.3g}".format(value)
        elif self.unit == "ms":
            s = "{:.3g}".format(value * 1000)
        else:
            if vmax - vmin > 3600:
                # Hours viz
                s = "{:d}:{:02d}:{:02d}".format(
                    int(value / 3600.0),
                    int(np.mod(value / 60.0, 60)),
                    int(np.mod(value, 60)),
                )
            elif vmax - vmin > 60:
                # Minutes viz
                s = "{:d}:{:02d}".format(int(value / 60.0), int(np.mod(value, 60)))
            elif vmax - vmin >= 1:
                # Seconds viz
                s = "{:.2g}".format(value)
            else:
                # Milliseconds viz
                s = "{:.3f}".format(value)

        return "{:s}{:s}".format(sign, s)


class NoteFormatter(Formatter):
    """Ticker formatter for Notes

    Parameters
    ----------
    octave : bool
        If ``True``, display the octave number along with the note name.

        Otherwise, only show the note name (and cent deviation)

    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    key : str
        Key for determining pitch spelling.

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See also
    --------
    LogHzFormatter
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax[1].set(ylabel='Note')
    """

    def __init__(self, octave=True, major=True, key="C:maj", unicode=True):

        self.octave = octave
        self.major = major
        self.key = key
        self.unicode = unicode

    def __call__(self, x, pos=None):

        if x <= 0:
            return ""

        # Only use cent precision if our vspan is less than an octave
        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ""

        cents = vmax <= 2 * max(1, vmin)

        return core.hz_to_note(
            x, octave=self.octave, cents=cents, key=self.key, unicode=self.unicode
        )


class SvaraFormatter(Formatter):
    """Ticker formatter for Svara

    Parameters
    ----------
    octave : bool
        If ``True``, display the octave number along with the note name.

        Otherwise, only show the note name (and cent deviation)

    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    Sa : number > 0
        Frequency (in Hz) of Sa

    mela : str or int
        For Carnatic svara, the index or name of the melakarta raga in question

        To use Hindustani svara, set ``mela=None``

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See also
    --------
    NoteFormatter
    matplotlib.ticker.Formatter
    librosa.hz_to_svara_c
    librosa.hz_to_svara_h


    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.SvaraFormatter(261))
    >>> ax[1].set(ylabel='Note')
    """

    def __init__(
        self, Sa, octave=True, major=True, abbr=False, mela=None, unicode=True
    ):

        if Sa is None:
            raise ParameterError(
                "Sa frequency is required for svara display formatting"
            )

        self.Sa = Sa
        self.octave = octave
        self.major = major
        self.abbr = abbr
        self.mela = mela
        self.unicode = unicode

    def __call__(self, x, pos=None):

        if x <= 0:
            return ""

        # Only use cent precision if our vspan is less than an octave
        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ""

        if self.mela is None:
            return core.hz_to_svara_h(
                x, Sa=self.Sa, octave=self.octave, abbr=self.abbr, unicode=self.unicode
            )
        else:
            return core.hz_to_svara_c(
                x,
                Sa=self.Sa,
                mela=self.mela,
                octave=self.octave,
                abbr=self.abbr,
                unicode=self.unicode,
            )


class LogHzFormatter(Formatter):
    """Ticker formatter for logarithmic frequency

    Parameters
    ----------
    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    See also
    --------
    NoteFormatter
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].yaxis.set_major_formatter(librosa.display.LogHzFormatter())
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax[1].set(ylabel='Note')
    """

    def __init__(self, major=True):

        self.major = major

    def __call__(self, x, pos=None):

        if x <= 0:
            return ""

        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ""

        return "{:g}".format(x)


class ChromaFormatter(Formatter):
    """A formatter for chroma axes

    See also
    --------
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.arange(12)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(values)
    >>> ax.yaxis.set_major_formatter(librosa.display.ChromaFormatter())
    >>> ax.set(ylabel='Pitch class')
    """

    def __init__(self, key="C:maj", unicode=True):
        self.key = key
        self.unicode = unicode

    def __call__(self, x, pos=None):
        """Format for chroma positions"""
        return core.midi_to_note(
            int(x), octave=False, cents=False, key=self.key, unicode=self.unicode
        )


class ChromaSvaraFormatter(Formatter):
    """A formatter for chroma axes with svara instead of notes.

    If mela is given, Carnatic svara names will be used.

    Otherwise, Hindustani svara names will be used.

    If `Sa` is not given, it will default to 0 (equivalent to `C`).

    See Also
    --------
    ChromaFormatter

    """

    def __init__(self, Sa=None, mela=None, abbr=True, unicode=True):
        if Sa is None:
            Sa = 0
        self.Sa = Sa
        self.mela = mela
        self.abbr = abbr
        self.unicode = unicode

    def __call__(self, x, pos=None):
        """Format for chroma positions"""
        if self.mela is not None:
            return core.midi_to_svara_c(
                int(x),
                Sa=self.Sa,
                mela=self.mela,
                octave=False,
                abbr=self.abbr,
                unicode=self.unicode,
            )
        else:
            return core.midi_to_svara_h(
                int(x), Sa=self.Sa, octave=False, abbr=self.abbr, unicode=self.unicode
            )


class TonnetzFormatter(Formatter):
    """A formatter for tonnetz axes

    See also
    --------
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.arange(6)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(values)
    >>> ax.yaxis.set_major_formatter(librosa.display.TonnetzFormatter())
    >>> ax.set(ylabel='Tonnetz')
    """

    def __call__(self, x, pos=None):
        """Format for tonnetz positions"""
        return [r"5$_x$", r"5$_y$", r"m3$_x$", r"m3$_y$", r"M3$_x$", r"M3$_y$"][int(x)]


class AdaptiveWaveplot:
    """A helper class for managing adaptive wave visualizations.

    This object is used to dynamically switch between sample-based and envelope-based
    visualizations of waveforms.
    When the display is zoomed in such that no more than `max_samples` would be
    visible, the sample-based display is used.
    When displaying the raw samples would require more than `max_samples`, an
    envelope-based plot is used instead.

    You should never need to instantiate this object directly, as it is constructed
    automatically by `waveshow`.

    Parameters
    ----------
    times : np.ndarray
        An array containing the time index (in seconds) for each sample.

    y : np.ndarray
        An array containing the (monophonic) wave samples.

    steps : matplotlib.lines.Lines2D
        The matplotlib artist used for the sample-based visualization.
        This is constructed by `matplotlib.pyplot.step`.

    envelope : matplotlib.collections.PolyCollection
        The matplotlib artist used for the envelope-based visualization.
        This is constructed by `matplotlib.pyplot.fill_between`.

    sr : number > 0
        The sampling rate of the audio

    max_samples : int > 0
        The maximum number of samples to use for sample-based display.

    See Also
    --------
    waveshow
    """

    def __init__(self, times, y, steps, envelope, sr=22050, max_samples=11025):
        self.times = times
        self.samples = y
        self.steps = steps
        self.envelope = envelope
        self.sr = sr
        self.max_samples = max_samples

    def update(self, ax):
        """Update the matplotlib display according to the current viewport limits.

        This is a callback function, and should not be used directly.

        Parameters
        ----------
        ax : matplotlib axes object
            The axes object to update
        """
        lims = ax.viewLim

        # Does our width cover fewer than max_samples?
        # If so, then use the sample-based plot
        if lims.width * self.sr <= self.max_samples:
            self.envelope.set_visible(False)
            self.steps.set_visible(True)

            # Now check that our viewport
            xdata = self.steps.get_xdata()
            if lims.x0 <= xdata[0] or lims.x1 >= xdata[-1]:
                # Viewport expands beyond current data in steps; update
                # we want to cover a window of self.max_samples centered on the current viewport
                midpoint_time = (lims.x1 + lims.x0) / 2
                idx_start = np.searchsorted(
                    self.times, midpoint_time - 0.5 * self.max_samples / self.sr
                )
                self.steps.set_data(
                    self.times[idx_start : idx_start + self.max_samples],
                    self.samples[idx_start : idx_start + self.max_samples],
                )
        else:
            # Otherwise, use the envelope plot
            self.envelope.set_visible(True)
            self.steps.set_visible(False)

        ax.figure.canvas.draw_idle()


@deprecate_positional_args
def cmap(
    data, *, robust=True, cmap_seq="magma", cmap_bool="gray_r", cmap_div="coolwarm"
):
    """Get a default colormap from the given data.

    If the data is boolean, use a black and white colormap.

    If the data has both positive and negative values,
    use a diverging colormap.

    Otherwise, use a sequential colormap.

    Parameters
    ----------
    data : np.ndarray
        Input data
    robust : bool
        If True, discard the top and bottom 2% of data when calculating
        range.
    cmap_seq : str
        The sequential colormap name
    cmap_bool : str
        The boolean colormap name
    cmap_div : str
        The diverging colormap name

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The colormap to use for ``data``

    See Also
    --------
    matplotlib.pyplot.colormaps
    """

    data = np.atleast_1d(data)

    if data.dtype == "bool":
        return get_cmap(cmap_bool, lut=2)

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    min_val, max_val = np.percentile(data, [min_p, max_p])

    if min_val >= 0 or max_val <= 0:
        return get_cmap(cmap_seq)

    return get_cmap(cmap_div)


def __envelope(x, hop):
    """Compute the max-envelope of non-overlapping frames of x at length hop

    x is assumed to be multi-channel, of shape (n_channels, n_samples).
    """
    x_frame = np.abs(util.frame(x, frame_length=hop, hop_length=hop))
    return x_frame.max(axis=1)


@deprecate_positional_args
def specshow(
    data,
    *,
    x_coords=None,
    y_coords=None,
    x_axis=None,
    y_axis=None,
    sr=22050,
    hop_length=512,
    n_fft=None,
    win_length=None,
    fmin=None,
    fmax=None,
    tuning=0.0,
    bins_per_octave=12,
    key="C:maj",
    Sa=None,
    mela=None,
    thaat=None,
    auto_aspect=True,
    htk=False,
    unicode=True,
    ax=None,
    **kwargs,
):
    """Display a spectrogram/chromagram/cqt/etc.

    For a detailed overview of this function, see :ref:`sphx_glr_auto_examples_plot_display.py`

    Parameters
    ----------
    data : np.ndarray [shape=(d, n)]
        Matrix to display (e.g., spectrogram)

    sr : number > 0 [scalar]
        Sample rate used to determine time scale in x-axis.

    hop_length : int > 0 [scalar]
        Hop length, also used to determine time scale in x-axis

    n_fft : int > 0 or None
        Number of samples per frame in STFT/spectrogram displays.
        By default, this will be inferred from the shape of ``data``
        as ``2 * (d - 1)``.
        If ``data`` was generated using an odd frame length, the correct
        value can be specified here.

    win_length : int > 0 or None
        The number of samples per window.
        By default, this will be inferred to match ``n_fft``.
        This is primarily useful for specifying odd window lengths in
        Fourier tempogram displays.

    x_axis, y_axis : None or str
        Range for the x- and y-axes.

        Valid types are:

        - None, 'none', or 'off' : no axis decoration is displayed.

        Frequency types:

        - 'linear', 'fft', 'hz' : frequency range is determined by
          the FFT window and sampling rate.
        - 'log' : the spectrum is displayed on a log scale.
        - 'fft_note': the spectrum is displayed on a log scale with pitches marked.
        - 'fft_svara': the spectrum is displayed on a log scale with svara marked.
        - 'mel' : frequencies are determined by the mel scale.
        - 'cqt_hz' : frequencies are determined by the CQT scale.
        - 'cqt_note' : pitches are determined by the CQT scale.
        - 'cqt_svara' : like `cqt_note` but using Hindustani or Carnatic svara

        All frequency types are plotted in units of Hz.

        Any spectrogram parameters (hop_length, sr, bins_per_octave, etc.)
        used to generate the input data should also be provided when
        calling `specshow`.

        Categorical types:

        - 'chroma' : pitches are determined by the chroma filters.
          Pitch classes are arranged at integer locations (0-11) according to
          a given key.

        - `chroma_h`, `chroma_c`: pitches are determined by chroma filters,
          and labeled as svara in the Hindustani (`chroma_h`) or Carnatic (`chroma_c`)
          according to a given thaat (Hindustani) or melakarta raga (Carnatic).

        - 'tonnetz' : axes are labeled by Tonnetz dimensions (0-5)
        - 'frames' : markers are shown as frame counts.

        Time types:

        - 'time' : markers are shown as milliseconds, seconds, minutes, or hours.
                Values are plotted in units of seconds.
        - 's' : markers are shown as seconds.
        - 'ms' : markers are shown as milliseconds.
        - 'lag' : like time, but past the halfway point counts as negative values.
        - 'lag_s' : same as lag, but in seconds.
        - 'lag_ms' : same as lag, but in milliseconds.

        Rhythm:

        - 'tempo' : markers are shown as beats-per-minute (BPM)
            using a logarithmic scale.  This is useful for
            visualizing the outputs of `feature.tempogram`.

        - 'fourier_tempo' : same as `'tempo'`, but used when
            tempograms are calculated in the Frequency domain
            using `feature.fourier_tempogram`.

    x_coords, y_coords : np.ndarray [shape=data.shape[0 or 1]]
        Optional positioning coordinates of the input data.
        These can be use to explicitly set the location of each
        element ``data[i, j]``, e.g., for displaying beat-synchronous
        features in natural time coordinates.

        If not provided, they are inferred from ``x_axis`` and ``y_axis``.

    fmin : float > 0 [scalar] or None
        Frequency of the lowest spectrogram bin.  Used for Mel and CQT
        scales.

        If ``y_axis`` is `cqt_hz` or `cqt_note` and ``fmin`` is not given,
        it is set by default to ``note_to_hz('C1')``.

    fmax : float > 0 [scalar] or None
        Used for setting the Mel frequency scales

    tuning : float
        Tuning deviation from A440, in fractions of a bin.

        This is used for CQT frequency scales, so that ``fmin`` is adjusted
        to ``fmin * 2**(tuning / bins_per_octave)``.

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave.  Used for CQT frequency scale.

    key : str
        The reference key to use when using note axes (`cqt_note`, `chroma`).

    Sa : float or int
        If using Hindustani or Carnatic svara axis decorations, specify Sa.

        For `cqt_svara`, ``Sa`` should be specified as a frequency in Hz.

        For `chroma_c` or `chroma_h`, ``Sa`` should correspond to the position
        of Sa within the chromagram.
        If not provided, Sa will default to 0 (equivalent to `C`)

    mela : str or int, optional
        If using `chroma_c` or `cqt_svara` display mode, specify the melakarta raga.

    thaat : str, optional
        If using `chroma_h` display mode, specify the parent thaat.

    auto_aspect : bool
        Axes will have 'equal' aspect if the horizontal and vertical dimensions
        cover the same extent and their types match.

        To override, set to `False`.

    htk : bool
        If plotting on a mel frequency axis, specify which version of the mel
        scale to use.

            - `False`: use Slaney formula (default)
            - `True`: use HTK formula

        See `core.mel_frequencies` for more information.

    unicode : bool
        If using note or svara decorations, setting `unicode=True`
        will use unicode glyphs for accidentals and octave encoding.

        Setting `unicode=False` will use ASCII glyphs.  This can be helpful
        if your font does not support musical notation symbols.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    **kwargs : additional keyword arguments
        Arguments passed through to `matplotlib.pyplot.pcolormesh`.

        By default, the following options are set:

            - ``rasterized=True``
            - ``shading='auto'``
            - ``edgecolors='None'``

    Returns
    -------
    colormesh : `matplotlib.collections.QuadMesh`
        The color mesh object produced by `matplotlib.pyplot.pcolormesh`

    See Also
    --------
    cmap : Automatic colormap detection
    matplotlib.pyplot.pcolormesh

    Examples
    --------
    Visualize an STFT power spectrum using default parameters

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=15)
    >>> fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    >>> img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
    ...                                sr=sr, ax=ax[0])
    >>> ax[0].set(title='Linear-frequency power spectrogram')
    >>> ax[0].label_outer()

    Or on a logarithmic scale, and using a larger hop

    >>> hop_length = 1024
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
    ...                             ref=np.max)
    >>> librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
    ...                          x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-frequency power spectrogram')
    >>> ax[1].label_outer()
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    """

    if np.issubdtype(data.dtype, np.complexfloating):
        warnings.warn(
            "Trying to display complex-valued input. " "Showing magnitude instead.",
            stacklevel=2,
        )
        data = np.abs(data)

    kwargs.setdefault("cmap", cmap(data))
    kwargs.setdefault("rasterized", True)
    kwargs.setdefault("edgecolors", "None")
    kwargs.setdefault("shading", "auto")

    all_params = dict(
        kwargs=kwargs,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        tuning=tuning,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=win_length,
        key=key,
        htk=htk,
        unicode=unicode,
    )

    # Get the x and y coordinates
    y_coords = __mesh_coords(y_axis, y_coords, data.shape[0], **all_params)
    x_coords = __mesh_coords(x_axis, x_coords, data.shape[1], **all_params)

    axes = __check_axes(ax)

    out = axes.pcolormesh(x_coords, y_coords, data, **kwargs)

    __set_current_image(ax, out)

    # Set up axis scaling
    __scale_axes(axes, x_axis, "x")
    __scale_axes(axes, y_axis, "y")

    # Construct tickers and locators
    __decorate_axis(
        axes.xaxis, x_axis, key=key, Sa=Sa, mela=mela, thaat=thaat, unicode=unicode
    )
    __decorate_axis(
        axes.yaxis, y_axis, key=key, Sa=Sa, mela=mela, thaat=thaat, unicode=unicode
    )

    # If the plot is a self-similarity/covariance etc. plot, square it
    if __same_axes(x_axis, y_axis, axes.get_xlim(), axes.get_ylim()) and auto_aspect:
        axes.set_aspect("equal")

    return out


def __set_current_image(ax, img):
    """Helper to set the current image in pyplot mode.

    If the provided ``ax`` is not `None`, then we assume that the user is using the object API.
    In this case, the pyplot current image is not set.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        plt.sci(img)


def __mesh_coords(ax_type, coords, n, **kwargs):
    """Compute axis coordinates"""

    if coords is not None:
        if len(coords) not in (n, n + 1):
            raise ParameterError(
                f"Coordinate shape mismatch: {len(coords)}!={n} or {n}+1"
            )
        return coords

    coord_map = {
        "linear": __coord_fft_hz,
        "fft": __coord_fft_hz,
        "fft_note": __coord_fft_hz,
        "fft_svara": __coord_fft_hz,
        "hz": __coord_fft_hz,
        "log": __coord_fft_hz,
        "mel": __coord_mel_hz,
        "cqt": __coord_cqt_hz,
        "cqt_hz": __coord_cqt_hz,
        "cqt_note": __coord_cqt_hz,
        "cqt_svara": __coord_cqt_hz,
        "chroma": __coord_chroma,
        "chroma_c": __coord_chroma,
        "chroma_h": __coord_chroma,
        "time": __coord_time,
        "s": __coord_time,
        "ms": __coord_time,
        "lag": __coord_time,
        "lag_s": __coord_time,
        "lag_ms": __coord_time,
        "tonnetz": __coord_n,
        "off": __coord_n,
        "tempo": __coord_tempo,
        "fourier_tempo": __coord_fourier_tempo,
        "frames": __coord_n,
        None: __coord_n,
    }

    if ax_type not in coord_map:
        raise ParameterError("Unknown axis type: {}".format(ax_type))
    return coord_map[ax_type](n, **kwargs)


def __check_axes(axes):
    """Check if "axes" is an instance of an axis object. If not, use `gca`."""
    if axes is None:
        import matplotlib.pyplot as plt

        axes = plt.gca()
    elif not isinstance(axes, Axes):
        raise ParameterError(
            "`axes` must be an instance of matplotlib.axes.Axes. "
            "Found type(axes)={}".format(type(axes))
        )
    return axes


def __scale_axes(axes, ax_type, which):
    """Set the axis scaling"""

    kwargs = dict()
    if which == "x":
        if version_parse(matplotlib.__version__) < version_parse("3.3.0"):
            thresh = "linthreshx"
            base = "basex"
            scale = "linscalex"
        else:
            thresh = "linthresh"
            base = "base"
            scale = "linscale"

        scaler = axes.set_xscale
        limit = axes.set_xlim
    else:
        if version_parse(matplotlib.__version__) < version_parse("3.3.0"):
            thresh = "linthreshy"
            base = "basey"
            scale = "linscaley"
        else:
            thresh = "linthresh"
            base = "base"
            scale = "linscale"

        scaler = axes.set_yscale
        limit = axes.set_ylim

    # Map ticker scales
    if ax_type == "mel":
        mode = "symlog"
        kwargs[thresh] = 1000.0
        kwargs[base] = 2

    elif ax_type in ["cqt", "cqt_hz", "cqt_note", "cqt_svara"]:
        mode = "log"
        kwargs[base] = 2

    elif ax_type in ["log", "fft_note", "fft_svara"]:
        mode = "symlog"
        kwargs[base] = 2
        kwargs[thresh] = core.note_to_hz("C2")
        kwargs[scale] = 0.5

    elif ax_type in ["tempo", "fourier_tempo"]:
        mode = "log"
        kwargs[base] = 2
        limit(16, 480)
    else:
        return

    scaler(mode, **kwargs)


def __decorate_axis(
    axis, ax_type, key="C:maj", Sa=None, mela=None, thaat=None, unicode=True
):
    """Configure axis tickers, locators, and labels"""

    if ax_type == "tonnetz":
        axis.set_major_formatter(TonnetzFormatter())
        axis.set_major_locator(FixedLocator(np.arange(6)))
        axis.set_label_text("Tonnetz")

    elif ax_type == "chroma":
        axis.set_major_formatter(ChromaFormatter(key=key, unicode=unicode))
        degrees = core.key_to_degrees(key)
        axis.set_major_locator(
            FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel())
        )
        axis.set_label_text("Pitch class")

    elif ax_type == "chroma_h":
        if Sa is None:
            Sa = 0
        axis.set_major_formatter(ChromaSvaraFormatter(Sa=Sa, unicode=unicode))
        if thaat is None:
            # If no thaat is given, show all svara
            degrees = np.arange(12)
        else:
            degrees = core.thaat_to_degrees(thaat)
        # Rotate degrees relative to Sa
        degrees = np.mod(degrees + Sa, 12)
        axis.set_major_locator(
            FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel())
        )
        axis.set_label_text("Svara")

    elif ax_type == "chroma_c":
        if Sa is None:
            Sa = 0
        axis.set_major_formatter(
            ChromaSvaraFormatter(Sa=Sa, mela=mela, unicode=unicode)
        )
        degrees = core.mela_to_degrees(mela)
        # Rotate degrees relative to Sa
        degrees = np.mod(degrees + Sa, 12)
        axis.set_major_locator(
            FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel())
        )
        axis.set_label_text("Svara")

    elif ax_type in ["tempo", "fourier_tempo"]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_label_text("BPM")

    elif ax_type == "time":
        axis.set_major_formatter(TimeFormatter(unit=None, lag=False))
        axis.set_major_locator(MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text("Time")

    elif ax_type == "s":
        axis.set_major_formatter(TimeFormatter(unit="s", lag=False))
        axis.set_major_locator(MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text("Time (s)")

    elif ax_type == "ms":
        axis.set_major_formatter(TimeFormatter(unit="ms", lag=False))
        axis.set_major_locator(MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text("Time (ms)")

    elif ax_type == "lag":
        axis.set_major_formatter(TimeFormatter(unit=None, lag=True))
        axis.set_major_locator(MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text("Lag")

    elif ax_type == "lag_s":
        axis.set_major_formatter(TimeFormatter(unit="s", lag=True))
        axis.set_major_locator(MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text("Lag (s)")

    elif ax_type == "lag_ms":
        axis.set_major_formatter(TimeFormatter(unit="ms", lag=True))
        axis.set_major_locator(MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text("Lag (ms)")

    elif ax_type == "cqt_note":
        axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
        # Where is C1 relative to 2**k hz?
        log_C1 = np.log2(core.note_to_hz("C1"))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(LogLocator(base=2.0, subs=(C_offset,)))
        axis.set_minor_formatter(NoteFormatter(key=key, major=False, unicode=unicode))
        axis.set_minor_locator(
            LogLocator(base=2.0, subs=C_offset * 2.0 ** (np.arange(1, 12) / 12.0))
        )
        axis.set_label_text("Note")

    elif ax_type == "cqt_svara":
        axis.set_major_formatter(SvaraFormatter(Sa=Sa, mela=mela, unicode=unicode))
        # Find the offset of Sa relative to 2**k Hz
        sa_offset = 2.0 ** (np.log2(Sa) - np.floor(np.log2(Sa)))

        axis.set_major_locator(LogLocator(base=2.0, subs=(sa_offset,)))
        axis.set_minor_formatter(
            SvaraFormatter(Sa=Sa, mela=mela, major=False, unicode=unicode)
        )
        axis.set_minor_locator(
            LogLocator(base=2.0, subs=sa_offset * 2.0 ** (np.arange(1, 12) / 12.0))
        )
        axis.set_label_text("Svara")

    elif ax_type in ["cqt_hz"]:
        axis.set_major_formatter(LogHzFormatter())
        log_C1 = np.log2(core.note_to_hz("C1"))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(LogLocator(base=2.0, subs=(C_offset,)))
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_minor_formatter(LogHzFormatter(major=False))
        axis.set_minor_locator(
            LogLocator(base=2.0, subs=C_offset * 2.0 ** (np.arange(1, 12) / 12.0))
        )
        axis.set_label_text("Hz")

    elif ax_type == "fft_note":
        axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
        # Where is C1 relative to 2**k hz?
        log_C1 = np.log2(core.note_to_hz("C1"))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(SymmetricalLogLocator(axis.get_transform()))
        axis.set_minor_formatter(NoteFormatter(key=key, major=False, unicode=unicode))
        axis.set_minor_locator(
            LogLocator(base=2.0, subs=2.0 ** (np.arange(1, 12) / 12.0))
        )
        axis.set_label_text("Note")

    elif ax_type == "fft_svara":
        axis.set_major_formatter(SvaraFormatter(Sa=Sa, mela=mela, unicode=unicode))
        # Find the offset of Sa relative to 2**k Hz
        log_Sa = np.log2(Sa)
        sa_offset = 2.0 ** (log_Sa - np.floor(log_Sa))

        axis.set_major_locator(
            SymmetricalLogLocator(axis.get_transform(), base=2.0, subs=[sa_offset])
        )
        axis.set_minor_formatter(
            SvaraFormatter(Sa=Sa, mela=mela, major=False, unicode=unicode)
        )
        axis.set_minor_locator(
            LogLocator(base=2.0, subs=sa_offset * 2.0 ** (np.arange(1, 12) / 12.0))
        )
        axis.set_label_text("Svara")

    elif ax_type in ["mel", "log"]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(SymmetricalLogLocator(axis.get_transform()))
        axis.set_label_text("Hz")

    elif ax_type in ["linear", "hz", "fft"]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_label_text("Hz")

    elif ax_type in ["frames"]:
        axis.set_label_text("Frames")

    elif ax_type in ["off", "none", None]:
        axis.set_label_text("")
        axis.set_ticks([])

    else:
        raise ParameterError("Unsupported axis type: {}".format(ax_type))


def __coord_fft_hz(n, sr=22050, n_fft=None, **_kwargs):
    """Get the frequencies for FFT bins"""
    if n_fft is None:
        n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = core.fft_frequencies(sr=sr, n_fft=n_fft)
    return basis


def __coord_mel_hz(n, fmin=0, fmax=None, sr=22050, htk=False, **_kwargs):
    """Get the frequencies for Mel bins"""

    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = 0.5 * sr

    basis = core.mel_frequencies(n, fmin=fmin, fmax=fmax, htk=htk)
    return basis


def __coord_cqt_hz(n, fmin=None, bins_per_octave=12, sr=22050, **_kwargs):
    """Get CQT bin frequencies"""
    if fmin is None:
        fmin = core.note_to_hz("C1")

    # Apply tuning correction
    fmin = fmin * 2.0 ** (_kwargs.get("tuning", 0.0) / bins_per_octave)

    # we drop by half a bin so that CQT bins are centered vertically
    freqs = core.cqt_frequencies(
        n,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
    )

    if np.any(freqs > 0.5 * sr):
        warnings.warn(
            "Frequency axis exceeds Nyquist. "
            "Did you remember to set all spectrogram parameters in specshow?",
            stacklevel=4,
        )

    return freqs


def __coord_chroma(n, bins_per_octave=12, **_kwargs):
    """Get chroma bin numbers"""
    return np.linspace(0, (12.0 * n) / bins_per_octave, num=n, endpoint=False)


def __coord_tempo(n, sr=22050, hop_length=512, **_kwargs):
    """Tempo coordinates"""
    basis = core.tempo_frequencies(n + 1, sr=sr, hop_length=hop_length)[1:]
    return basis


def __coord_fourier_tempo(n, sr=22050, hop_length=512, win_length=None, **_kwargs):
    """Fourier tempogram coordinates"""
    if win_length is None:
        win_length = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = core.fourier_tempo_frequencies(
        sr=sr, hop_length=hop_length, win_length=win_length
    )
    return basis


def __coord_n(n, **_kwargs):
    """Get bare positions"""
    return np.arange(n)


def __coord_time(n, sr=22050, hop_length=512, **_kwargs):
    """Get time coordinates from frames"""
    return core.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)


def __same_axes(x_axis, y_axis, xlim, ylim):
    """Check if two axes are the same, used to determine squared plots"""
    axes_same_and_not_none = (x_axis == y_axis) and (x_axis is not None)
    axes_same_lim = xlim == ylim
    return axes_same_and_not_none and axes_same_lim


@deprecate_positional_args
def waveshow(
    y,
    *,
    sr=22050,
    max_points=11025,
    x_axis="time",
    offset=0.0,
    marker="",
    where="post",
    label=None,
    ax=None,
    **kwargs,
):
    """Visualize a waveform in the time domain.

    This function constructs a plot which adaptively switches between a raw
    samples-based view of the signal (`matplotlib.pyplot.step`) and an
    amplitude-envelope view of the signal (`matplotlib.pyplot.fill_between`)
    depending on the time extent of the plot's viewport.

    More specifically, when the plot spans a time interval of less than ``max_points /
    sr`` (by default, 1/2 second), the samples-based view is used, and otherwise a
    downsampled amplitude envelope is used.
    This is done to limit the complexity of the visual elements to guarantee an
    efficient, visually interpretable plot.

    When using interactive rendering (e.g., in a Jupyter notebook or IPython
    console), the plot will automatically update as the view-port is changed, either
    through widget controls or programmatic updates.

    .. note:: When visualizing stereo waveforms, the amplitude envelope will be generated
        so that the upper limits derive from the left channel, and the lower limits derive
        from the right channel, which can produce a vertically asymmetric plot.

        When zoomed in to the sample view, only the first channel will be shown.
        If you want to visualize both channels at the sample level, it is recommended to
        plot each signal independently.

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or (2,n)]
        audio time series (mono or stereo)

    sr : number > 0 [scalar]
        sampling rate of ``y`` (samples per second)

    max_points : positive integer
        Maximum number of samples to draw.  When the plot covers a time extent
        smaller than ``max_points / sr`` (default: 1/2 second), samples are drawn.

        If drawing raw samples would exceed `max_points`, then a downsampled
        amplitude envelope extracted from non-overlapping windows of `y` is
        visualized instead.  The parameters of the amplitude envelope are defined so
        that the resulting plot cannot produce more than `max_points` frames.

    x_axis : str or None
        Display of the x-axis ticks and tick markers. Accepted values are:

        - 'time' : markers are shown as milliseconds, seconds, minutes, or hours.
                    Values are plotted in units of seconds.

        - 's' : markers are shown as seconds.

        - 'ms' : markers are shown as milliseconds.

        - 'lag' : like time, but past the halfway point counts as negative values.

        - 'lag_s' : same as lag, but in seconds.

        - 'lag_ms' : same as lag, but in milliseconds.

        - `None`, 'none', or 'off': ticks and tick markers are hidden.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    offset : float
        Horizontal offset (in seconds) to start the waveform plot

    marker : string
        Marker symbol to use for sample values. (default: no markers)

        See also: `matplotlib.markers`.

    where : string, {'pre', 'mid', 'post'}
        This setting determines how both waveform and envelope plots interpolate
        between observations.

        See `matplotlib.pyplot.step` for details.

        Default: 'post'

    label : string [optional]
        The label string applied to this plot.
        Note that the label

    **kwargs
        Additional keyword arguments to `matplotlib.pyplot.fill_between` and
        `matplotlib.pyplot.step`.

        Note that only those arguments which are common to both functions will be
        supported.

    Returns
    -------
    librosa.display.AdaptiveWaveplot
        An object of type `librosa.display.AdaptiveWaveplot`

    See Also
    --------
    AdaptiveWaveplot
    matplotlib.pyplot.step
    matplotlib.pyplot.fill_between
    matplotlib.markers

    Examples
    --------
    Plot a monophonic waveform with an envelope view

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.waveshow(y, sr=sr, ax=ax[0])
    >>> ax[0].set(title='Envelope view, mono')
    >>> ax[0].label_outer()

    Or a stereo waveform

    >>> y, sr = librosa.load(librosa.ex('choice', hq=True), mono=False, duration=10)
    >>> librosa.display.waveshow(y, sr=sr, ax=ax[1])
    >>> ax[1].set(title='Envelope view, stereo')
    >>> ax[1].label_outer()

    Or harmonic and percussive components with transparency

    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> y_harm, y_perc = librosa.effects.hpss(y)
    >>> librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax[2], label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax[2], label='Percussive')
    >>> ax[2].set(title='Multiple waveforms')
    >>> ax[2].legend()

    Zooming in on a plot to show raw sample values

    >>> fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
    >>> ax.set(xlim=[6.0, 6.01], title='Sample view', ylim=[-0.2, 0.2])
    >>> librosa.display.waveshow(y, sr=sr, ax=ax, marker='.', label='Full signal')
    >>> librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax2, label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax2, label='Percussive')
    >>> ax.label_outer()
    >>> ax.legend()
    >>> ax2.legend()

    """
    util.valid_audio(y, mono=False)

    # Pad an extra channel dimension, if necessary
    if y.ndim == 1:
        y = y[np.newaxis, :]

    if max_points <= 0:
        raise ParameterError(
            "max_points={} must be strictly positive".format(max_points)
        )

    # Create the adaptive drawing object
    axes = __check_axes(ax)

    if "color" not in kwargs:
        kwargs.setdefault("color", next(axes._get_lines.prop_cycler)["color"])

    # Reduce by envelope calculation
    # this choice of hop ensures that the envelope has at most max_points values
    hop_length = max(1, y.shape[-1] // max_points)
    y_env = __envelope(y, hop_length)

    # Split the envelope into top and bottom
    y_bottom, y_top = -y_env[-1], y_env[0]

    times = offset + core.times_like(y, sr=sr, hop_length=1)

    # Only plot up to max_points worth of data here
    (steps,) = axes.step(
        times[:max_points], y[0, :max_points], marker=marker, where=where, **kwargs
    )

    envelope = axes.fill_between(
        times[: len(y_top) * hop_length : hop_length],
        y_bottom,
        y_top,
        step=where,
        label=label,
        **kwargs,
    )
    adaptor = AdaptiveWaveplot(
        times, y[0], steps, envelope, sr=sr, max_samples=max_points
    )

    axes.callbacks.connect("xlim_changed", adaptor.update)

    # Force an initial update to ensure the state is consistent
    adaptor.update(axes)

    # Construct tickers and locators
    __decorate_axis(axes.xaxis, x_axis)

    return adaptor
