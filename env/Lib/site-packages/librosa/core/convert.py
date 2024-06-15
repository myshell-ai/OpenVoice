#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit conversion utilities"""

import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import deprecate_positional_args

__all__ = [
    "frames_to_samples",
    "frames_to_time",
    "samples_to_frames",
    "samples_to_time",
    "time_to_samples",
    "time_to_frames",
    "blocks_to_samples",
    "blocks_to_frames",
    "blocks_to_time",
    "note_to_hz",
    "note_to_midi",
    "midi_to_hz",
    "midi_to_note",
    "hz_to_note",
    "hz_to_midi",
    "hz_to_mel",
    "hz_to_octs",
    "mel_to_hz",
    "octs_to_hz",
    "A4_to_tuning",
    "tuning_to_A4",
    "fft_frequencies",
    "cqt_frequencies",
    "mel_frequencies",
    "tempo_frequencies",
    "fourier_tempo_frequencies",
    "A_weighting",
    "B_weighting",
    "C_weighting",
    "D_weighting",
    "Z_weighting",
    "frequency_weighting",
    "multi_frequency_weighting",
    "samples_like",
    "times_like",
    "midi_to_svara_h",
    "midi_to_svara_c",
    "note_to_svara_h",
    "note_to_svara_c",
    "hz_to_svara_h",
    "hz_to_svara_c",
]


@deprecate_positional_args
def frames_to_samples(frames, *, hop_length=512, n_fft=None):
    """Converts frame indices to audio sample indices.

    Parameters
    ----------
    frames : number or np.ndarray [shape=(n,)]
        frame index or vector of frame indices
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : number or np.ndarray
        time (in samples) of each given frame number::

            times[i] = frames[i] * hop_length

    See Also
    --------
    frames_to_time : convert frame indices to time values
    samples_to_frames : convert sample indices to frame indices

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> beat_samples = librosa.frames_to_samples(beats)
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)


@deprecate_positional_args
def samples_to_frames(samples, *, hop_length=512, n_fft=None):
    """Converts sample indices into STFT frames.

    Examples
    --------
    >>> # Get the frame numbers for every 256 samples
    >>> librosa.samples_to_frames(np.arange(0, 22050, 256))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
            7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13,
           14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
           21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27,
           28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34,
           35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41,
           42, 42, 43])

    Parameters
    ----------
    samples : int or np.ndarray [shape=(n,)]
        sample index or vector of sample indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``- n_fft // 2``
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : int or np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times::

            frames[i] = floor( samples[i] / hop_length )

    See Also
    --------
    samples_to_time : convert sample indices to time values
    frames_to_samples : convert frame indices to sample indices
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    samples = np.asanyarray(samples)
    return np.floor((samples - offset) // hop_length).astype(int)


@deprecate_positional_args
def frames_to_time(frames, *, sr=22050, hop_length=512, n_fft=None):
    """Converts frame counts to time (seconds).

    Parameters
    ----------
    frames : np.ndarray [shape=(n,)]
        frame index or vector of frame indices
    sr : number > 0 [scalar]
        audio sampling rate
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        time (in seconds) of each given frame number::

            times[i] = frames[i] * hop_length / sr

    See Also
    --------
    time_to_frames : convert time values to frame indices
    frames_to_samples : convert frame indices to sample indices

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> beat_times = librosa.frames_to_time(beats, sr=sr)
    """

    samples = frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)

    return samples_to_time(samples, sr=sr)


@deprecate_positional_args
def time_to_frames(times, *, sr=22050, hop_length=512, n_fft=None):
    """Converts time stamps into STFT frames.

    Parameters
    ----------
    times : np.ndarray [shape=(n,)]
        time (in seconds) or vector of time values

    sr : number > 0 [scalar]
        audio sampling rate

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``- n_fft // 2``
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times::

            frames[i] = floor( times[i] * sr / hop_length )

    See Also
    --------
    frames_to_time : convert frame indices to time values
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get the frame numbers for every 100ms

    >>> librosa.time_to_frames(np.arange(0, 1, 0.1),
    ...                         sr=22050, hop_length=512)
    array([ 0,  4,  8, 12, 17, 21, 25, 30, 34, 38])

    """

    samples = time_to_samples(times, sr=sr)

    return samples_to_frames(samples, hop_length=hop_length, n_fft=n_fft)


@deprecate_positional_args
def time_to_samples(times, *, sr=22050):
    """Convert timestamps (in seconds) to sample indices.

    Parameters
    ----------
    times : number or np.ndarray
        Time value or array of time values (in seconds)
    sr : number > 0
        Sampling rate

    Returns
    -------
    samples : int or np.ndarray [shape=times.shape, dtype=int]
        Sample indices corresponding to values in ``times``

    See Also
    --------
    time_to_frames : convert time values to frame indices
    samples_to_time : convert sample indices to time values

    Examples
    --------
    >>> librosa.time_to_samples(np.arange(0, 1, 0.1), sr=22050)
    array([    0,  2205,  4410,  6615,  8820, 11025, 13230, 15435,
           17640, 19845])

    """

    return (np.asanyarray(times) * sr).astype(int)


@deprecate_positional_args
def samples_to_time(samples, *, sr=22050):
    """Convert sample indices to time (in seconds).

    Parameters
    ----------
    samples : np.ndarray
        Sample index or array of sample indices
    sr : number > 0
        Sampling rate

    Returns
    -------
    times : np.ndarray [shape=samples.shape]
        Time values corresponding to ``samples`` (in seconds)

    See Also
    --------
    samples_to_frames : convert sample indices to frame indices
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get timestamps corresponding to every 512 samples

    >>> librosa.samples_to_time(np.arange(0, 22050, 512))
    array([ 0.   ,  0.023,  0.046,  0.07 ,  0.093,  0.116,  0.139,
            0.163,  0.186,  0.209,  0.232,  0.255,  0.279,  0.302,
            0.325,  0.348,  0.372,  0.395,  0.418,  0.441,  0.464,
            0.488,  0.511,  0.534,  0.557,  0.58 ,  0.604,  0.627,
            0.65 ,  0.673,  0.697,  0.72 ,  0.743,  0.766,  0.789,
            0.813,  0.836,  0.859,  0.882,  0.906,  0.929,  0.952,
            0.975,  0.998])
    """

    return np.asanyarray(samples) / float(sr)


@deprecate_positional_args
def blocks_to_frames(blocks, *, block_length):
    """Convert block indices to frame indices

    Parameters
    ----------
    blocks : np.ndarray
        Block index or array of block indices
    block_length : int > 0
        The number of frames per block

    Returns
    -------
    frames : np.ndarray [shape=samples.shape, dtype=int]
        The index or indices of frames corresponding to the beginning
        of each provided block.

    See Also
    --------
    blocks_to_samples
    blocks_to_time

    Examples
    --------
    Get frame indices for each block in a stream

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename, block_length=16,
    ...                         frame_length=2048, hop_length=512)
    >>> for n, y in enumerate(stream):
    ...     n_frame = librosa.blocks_to_frames(n, block_length=16)

    """
    return block_length * np.asanyarray(blocks)


@deprecate_positional_args
def blocks_to_samples(blocks, *, block_length, hop_length):
    """Convert block indices to sample indices

    Parameters
    ----------
    blocks : np.ndarray
        Block index or array of block indices
    block_length : int > 0
        The number of frames per block
    hop_length : int > 0
        The number of samples to advance between frames

    Returns
    -------
    samples : np.ndarray [shape=samples.shape, dtype=int]
        The index or indices of samples corresponding to the beginning
        of each provided block.

        Note that these correspond to the *first* sample index in
        each block, and are not frame-centered.

    See Also
    --------
    blocks_to_frames
    blocks_to_time

    Examples
    --------
    Get sample indices for each block in a stream

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename, block_length=16,
    ...                         frame_length=2048, hop_length=512)
    >>> for n, y in enumerate(stream):
    ...     n_sample = librosa.blocks_to_samples(n, block_length=16,
    ...                                          hop_length=512)

    """
    frames = blocks_to_frames(blocks, block_length=block_length)
    return frames_to_samples(frames, hop_length=hop_length)


@deprecate_positional_args
def blocks_to_time(blocks, *, block_length, hop_length, sr):
    """Convert block indices to time (in seconds)

    Parameters
    ----------
    blocks : np.ndarray
        Block index or array of block indices
    block_length : int > 0
        The number of frames per block
    hop_length : int > 0
        The number of samples to advance between frames
    sr : int > 0
        The sampling rate (samples per second)

    Returns
    -------
    times : np.ndarray [shape=samples.shape]
        The time index or indices (in seconds) corresponding to the
        beginning of each provided block.

        Note that these correspond to the time of the *first* sample
        in each block, and are not frame-centered.

    See Also
    --------
    blocks_to_frames
    blocks_to_samples

    Examples
    --------
    Get time indices for each block in a stream

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename, block_length=16,
    ...                         frame_length=2048, hop_length=512)
    >>> for n, y in enumerate(stream):
    ...     n_time = librosa.blocks_to_time(n, block_length=16,
    ...                                     hop_length=512, sr=sr)

    """
    samples = blocks_to_samples(
        blocks, block_length=block_length, hop_length=hop_length
    )
    return samples_to_time(samples, sr=sr)


def note_to_hz(note, **kwargs):
    """Convert one or more note names to frequency (Hz)

    Examples
    --------
    >>> # Get the frequency of a note
    >>> librosa.note_to_hz('C')
    array([ 16.352])
    >>> # Or multiple notes
    >>> librosa.note_to_hz(['A3', 'A4', 'A5'])
    array([ 220.,  440.,  880.])
    >>> # Or notes with tuning deviations
    >>> librosa.note_to_hz('C2-32', round_midi=False)
    array([ 64.209])

    Parameters
    ----------
    note : str or iterable of str
        One or more note names to convert
    **kwargs : additional keyword arguments
        Additional parameters to `note_to_midi`

    Returns
    -------
    frequencies : number or np.ndarray [shape=(len(note),)]
        Array of frequencies (in Hz) corresponding to ``note``

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    """
    return midi_to_hz(note_to_midi(note, **kwargs))


@deprecate_positional_args
def note_to_midi(note, *, round_midi=True):
    """Convert one or more spelled notes to MIDI number(s).

    Notes may be spelled out with optional accidentals or octave numbers.

    The leading note name is case-insensitive.

    Sharps are indicated with ``#``, flats may be indicated with ``!`` or ``b``.

    Parameters
    ----------
    note : str or iterable of str
        One or more note names.
    round_midi : bool
        - If ``True``, allow for fractional midi notes
        - Otherwise, round cent deviations to the nearest note

    Returns
    -------
    midi : float or np.array
        Midi note numbers corresponding to inputs.

    Raises
    ------
    ParameterError
        If the input is not in valid note format

    See Also
    --------
    midi_to_note
    note_to_hz

    Examples
    --------
    >>> librosa.note_to_midi('C')
    12
    >>> librosa.note_to_midi('C#3')
    49
    >>> librosa.note_to_midi('C♯3')  # Using Unicode sharp
    49
    >>> librosa.note_to_midi('C♭3')  # Using Unicode flat
    47
    >>> librosa.note_to_midi('f4')
    65
    >>> librosa.note_to_midi('Bb-1')
    10
    >>> librosa.note_to_midi('A!8')
    116
    >>> librosa.note_to_midi('G𝄪6')  # Double-sharp
    93
    >>> librosa.note_to_midi('B𝄫6')  # Double-flat
    93
    >>> librosa.note_to_midi('C♭𝄫5')  # Triple-flats also work
    69
    >>> # Lists of notes also work
    >>> librosa.note_to_midi(['C', 'E', 'G'])
    array([12, 16, 19])
    """

    if not isinstance(note, str):
        return np.array([note_to_midi(n, round_midi=round_midi) for n in note])

    pitch_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    acc_map = {
        "#": 1,
        "": 0,
        "b": -1,
        "!": -1,
        "♯": 1,
        "𝄪": 2,
        "♭": -1,
        "𝄫": -2,
        "♮": 0,
    }

    match = re.match(
        r"^(?P<note>[A-Ga-g])"
        r"(?P<accidental>[#♯𝄪b!♭𝄫♮]*)"
        r"(?P<octave>[+-]?\d+)?"
        r"(?P<cents>[+-]\d+)?$",
        note,
    )
    if not match:
        raise ParameterError("Improper note format: {:s}".format(note))

    pitch = match.group("note").upper()
    offset = np.sum([acc_map[o] for o in match.group("accidental")])
    octave = match.group("octave")
    cents = match.group("cents")

    if not octave:
        octave = 0
    else:
        octave = int(octave)

    if not cents:
        cents = 0
    else:
        cents = int(cents) * 1e-2

    note_value = 12 * (octave + 1) + pitch_map[pitch] + offset + cents

    if round_midi:
        note_value = int(np.round(note_value))

    return note_value


@deprecate_positional_args
def midi_to_note(midi, *, octave=True, cents=False, key="C:maj", unicode=True):
    """Convert one or more MIDI numbers to note strings.

    MIDI numbers will be rounded to the nearest integer.

    Notes will be of the format 'C0', 'C♯0', 'D0', ...

    Examples
    --------
    >>> librosa.midi_to_note(0)
    'C-1'

    >>> librosa.midi_to_note(37)
    'C♯2'

    >>> librosa.midi_to_note(37, unicode=False)
    'C#2'

    >>> librosa.midi_to_note(-2)
    'A♯-2'

    >>> librosa.midi_to_note(104.7)
    'A7'

    >>> librosa.midi_to_note(104.7, cents=True)
    'A7-30'

    >>> librosa.midi_to_note(list(range(12, 24)))
    ['C0', 'C♯0', 'D0', 'D♯0', 'E0', 'F0', 'F♯0', 'G0', 'G♯0', 'A0', 'A♯0', 'B0']

    Use a key signature to resolve enharmonic equivalences

    >>> librosa.midi_to_note(range(12, 24), key='F:min')
    ['C0', 'D♭0', 'D0', 'E♭0', 'E0', 'F0', 'G♭0', 'G0', 'A♭0', 'A0', 'B♭0', 'B0']

    Parameters
    ----------
    midi : int or iterable of int
        Midi numbers to convert.

    octave : bool
        If True, include the octave number

    cents : bool
        If true, cent markers will be appended for fractional notes.
        Eg, ``midi_to_note(69.3, cents=True) == 'A4+03'``

    key : str
        A key signature to use when resolving enharmonic equivalences.

    unicode : bool
        If ``True`` (default), accidentals will use Unicode notation: ♭ or ♯

        If ``False``, accidentals will use ASCII-compatible notation: b or #

    Returns
    -------
    notes : str or iterable of str
        Strings describing each midi note.

    Raises
    ------
    ParameterError
        if ``cents`` is True and ``octave`` is False

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    key_to_notes
    """

    if cents and not octave:
        raise ParameterError("Cannot encode cents without octave information.")

    if not np.isscalar(midi):
        return [
            midi_to_note(x, octave=octave, cents=cents, key=key, unicode=unicode)
            for x in midi
        ]

    note_map = notation.key_to_notes(key=key, unicode=unicode)

    note_num = int(np.round(midi))
    note_cents = int(100 * np.around(midi - note_num, 2))

    note = note_map[note_num % 12]

    if octave:
        note = "{:s}{:0d}".format(note, int(note_num / 12) - 1)
    if cents:
        note = "{:s}{:+02d}".format(note, note_cents)

    return note


def midi_to_hz(notes):
    """Get the frequency (Hz) of MIDI note(s)

    Examples
    --------
    >>> librosa.midi_to_hz(36)
    65.406

    >>> librosa.midi_to_hz(np.arange(36, 48))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,
             87.307,   92.499,   97.999,  103.826,  110.   ,
            116.541,  123.471])

    Parameters
    ----------
    notes : int or np.ndarray [shape=(n,), dtype=int]
        midi number(s) of the note(s)

    Returns
    -------
    frequency : number or np.ndarray [shape=(n,), dtype=float]
        frequency (frequencies) of ``notes`` in Hz

    See Also
    --------
    hz_to_midi
    note_to_hz
    """

    return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))


def hz_to_midi(frequencies):
    """Get MIDI note number(s) for given frequencies

    Examples
    --------
    >>> librosa.hz_to_midi(60)
    34.506
    >>> librosa.hz_to_midi([110, 220, 440])
    array([ 45.,  57.,  69.])

    Parameters
    ----------
    frequencies : float or np.ndarray [shape=(n,), dtype=float]
        frequencies to convert

    Returns
    -------
    note_nums : number or np.ndarray [shape=(n,), dtype=float]
        MIDI notes to ``frequencies``

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    """

    return 12 * (np.log2(np.asanyarray(frequencies)) - np.log2(440.0)) + 69


def hz_to_note(frequencies, **kwargs):
    """Convert one or more frequencies (in Hz) to the nearest note names.

    Parameters
    ----------
    frequencies : float or iterable of float
        Input frequencies, specified in Hz
    **kwargs : additional keyword arguments
        Arguments passed through to `midi_to_note`

    Returns
    -------
    notes : list of str
        ``notes[i]`` is the closest note name to ``frequency[i]``
        (or ``frequency`` if the input is scalar)

    See Also
    --------
    hz_to_midi
    midi_to_note
    note_to_hz

    Examples
    --------
    Get a single note name for a frequency

    >>> librosa.hz_to_note(440.0)
    ['A5']

    Get multiple notes with cent deviation

    >>> librosa.hz_to_note([32, 64], cents=True)
    ['C1-38', 'C2-38']

    Get multiple notes, but suppress octave labels

    >>> librosa.hz_to_note(440.0 * (2.0 ** np.linspace(0, 1, 12)),
    ...                    octave=False)
    ['A', 'A#', 'B', 'C', 'C#', 'D', 'E', 'F', 'F#', 'G', 'G#', 'A']

    """
    return midi_to_note(hz_to_midi(frequencies), **kwargs)


@deprecate_positional_args
def hz_to_mel(frequencies, *, htk=False):
    """Convert Hz to Mels

    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels : number or np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


@deprecate_positional_args
def mel_to_hz(mels, *, htk=False):
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.

    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


@deprecate_positional_args
def hz_to_octs(frequencies, *, tuning=0.0, bins_per_octave=12):
    """Convert frequencies (Hz) to (fractional) octave numbers.

    Examples
    --------
    >>> librosa.hz_to_octs(440.0)
    4.
    >>> librosa.hz_to_octs([32, 64, 128, 256])
    array([ 0.219,  1.219,  2.219,  3.219])

    Parameters
    ----------
    frequencies : number >0 or np.ndarray [shape=(n,)] or float
        scalar or vector of frequencies
    tuning : float
        Tuning deviation from A440 in (fractional) bins per octave.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    octaves : number or np.ndarray [shape=(n,)]
        octave number for each frequency

    See Also
    --------
    octs_to_hz
    """

    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)

    return np.log2(np.asanyarray(frequencies) / (float(A440) / 16))


@deprecate_positional_args
def octs_to_hz(octs, *, tuning=0.0, bins_per_octave=12):
    """Convert octaves numbers to frequencies.

    Octaves are counted relative to A.

    Examples
    --------
    >>> librosa.octs_to_hz(1)
    55.
    >>> librosa.octs_to_hz([-2, -1, 0, 1, 2])
    array([   6.875,   13.75 ,   27.5  ,   55.   ,  110.   ])

    Parameters
    ----------
    octs : np.ndarray [shape=(n,)] or float
        octave number for each frequency
    tuning : float
        Tuning deviation from A440 in (fractional) bins per octave.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    frequencies : number or np.ndarray [shape=(n,)]
        scalar or vector of frequencies

    See Also
    --------
    hz_to_octs
    """
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)

    return (float(A440) / 16) * (2.0 ** np.asanyarray(octs))


@deprecate_positional_args
def A4_to_tuning(A4, *, bins_per_octave=12):
    """Convert a reference pitch frequency (e.g., ``A4=435``) to a tuning
    estimation, in fractions of a bin per octave.

    This is useful for determining the tuning deviation relative to
    A440 of a given frequency, assuming equal temperament. By default,
    12 bins per octave are used.

    This method is the inverse of `tuning_to_A4`.

    Examples
    --------
    The base case of this method in which A440 yields 0 tuning offset
    from itself.

    >>> librosa.A4_to_tuning(440.0)
    0.

    Convert a non-A440 frequency to a tuning offset relative
    to A440 using the default of 12 bins per octave.

    >>> librosa.A4_to_tuning(432.0)
    -0.318

    Convert two reference pitch frequencies to corresponding
    tuning estimations at once, but using 24 bins per octave.

    >>> librosa.A4_to_tuning([440.0, 444.0], bins_per_octave=24)
    array([   0.,   0.313   ])

    Parameters
    ----------
    A4 : float or np.ndarray [shape=(n,), dtype=float]
        Reference frequency(s) corresponding to A4.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    tuning : float or np.ndarray [shape=(n,), dtype=float]
        Tuning deviation from A440 in (fractional) bins per octave.

    See Also
    --------
    tuning_to_A4
    """
    return bins_per_octave * (np.log2(np.asanyarray(A4)) - np.log2(440.0))


@deprecate_positional_args
def tuning_to_A4(tuning, *, bins_per_octave=12):
    """Convert a tuning deviation (from 0) in fractions of a bin per
    octave (e.g., ``tuning=-0.1``) to a reference pitch frequency
    relative to A440.

    This is useful if you are working in a non-A440 tuning system
    to determine the reference pitch frequency given a tuning
    offset and assuming equal temperament. By default, 12 bins per
    octave are used.

    This method is the inverse of  `A4_to_tuning`.

    Examples
    --------
    The base case of this method in which a tuning deviation of 0
    gets us to our A440 reference pitch.

    >>> librosa.tuning_to_A4(0.0)
    440.

    Convert a nonzero tuning offset to its reference pitch frequency.

    >>> librosa.tuning_to_A4(-0.318)
    431.992

    Convert 3 tuning deviations at once to respective reference
    pitch frequencies, using 36 bins per octave.

    >>> librosa.tuning_to_A4([0.1, 0.2, -0.1], bins_per_octave=36)
    array([   440.848,    441.698   439.154])

    Parameters
    ----------
    tuning : float or np.ndarray [shape=(n,), dtype=float]
        Tuning deviation from A440 in fractional bins per octave.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    A4 : float or np.ndarray [shape=(n,), dtype=float]
        Reference frequency corresponding to A4.

    See Also
    --------
    A4_to_tuning
    """
    return 440.0 * 2.0 ** (np.asanyarray(tuning) / bins_per_octave)


@deprecate_positional_args
def fft_frequencies(*, sr=22050, n_fft=2048):
    """Alternative implementation of `np.fft.fftfreq`

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size

    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies ``(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)``

    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])

    """

    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


@deprecate_positional_args
def cqt_frequencies(n_bins, *, fmin, bins_per_octave=12, tuning=0.0):
    """Compute the center frequencies of Constant-Q bins.

    Examples
    --------
    >>> # Get the CQT frequencies for 24 notes, starting at C2
    >>> librosa.cqt_frequencies(24, fmin=librosa.note_to_hz('C2'))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,   87.307,
             92.499,   97.999,  103.826,  110.   ,  116.541,  123.471,
            130.813,  138.591,  146.832,  155.563,  164.814,  174.614,
            184.997,  195.998,  207.652,  220.   ,  233.082,  246.942])

    Parameters
    ----------
    n_bins : int > 0 [scalar]
        Number of constant-Q bins
    fmin : float > 0 [scalar]
        Minimum frequency
    bins_per_octave : int > 0 [scalar]
        Number of bins per octave
    tuning : float
        Deviation from A440 tuning in fractional bins

    Returns
    -------
    frequencies : np.ndarray [shape=(n_bins,)]
        Center frequency for each CQT bin
    """

    correction = 2.0 ** (float(tuning) / bins_per_octave)
    frequencies = 2.0 ** (np.arange(0, n_bins, dtype=float) / bins_per_octave)

    return correction * fmin * frequencies


@deprecate_positional_args
def mel_frequencies(n_mels=128, *, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale.

    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.

    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoaoustical experiments, several implementations coexist
    in the audio signal processing literature [#]_. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [#]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [#]_ (HTK) according to the following formula::

        mel = 2595.0 * np.log10(1.0 + f / 700.0).

    The choice of implementation is determined by the ``htk`` keyword argument: setting
    ``htk=False`` leads to the Auditory toolbox implementation, whereas setting it ``htk=True``
    leads to the HTK implementation.

    .. [#] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.

    .. [#] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

    .. [#] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.

    See Also
    --------
    hz_to_mel
    mel_to_hz
    librosa.feature.melspectrogram
    librosa.feature.mfcc

    Parameters
    ----------
    n_mels : int > 0 [scalar]
        Number of mel bins.
    fmin : float >= 0 [scalar]
        Minimum frequency (Hz).
    fmax : float >= 0 [scalar]
        Maximum frequency (Hz).
    htk : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


@deprecate_positional_args
def tempo_frequencies(n_bins, *, hop_length=512, sr=22050):
    """Compute the frequencies (in beats per minute) corresponding
    to an onset auto-correlation or tempogram matrix.

    Parameters
    ----------
    n_bins : int > 0
        The number of lag bins
    hop_length : int > 0
        The number of samples between each bin
    sr : number > 0
        The audio sampling rate

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_bins,)]
        vector of bin frequencies measured in BPM.

        .. note:: ``bin_frequencies[0] = +np.inf`` corresponds to 0-lag

    Examples
    --------
    Get the tempo frequencies corresponding to a 384-bin (8-second) tempogram

    >>> librosa.tempo_frequencies(384)
    array([      inf,  2583.984,  1291.992, ...,     6.782,
               6.764,     6.747])
    """

    bin_frequencies = np.zeros(int(n_bins), dtype=np.float64)

    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))

    return bin_frequencies


@deprecate_positional_args
def fourier_tempo_frequencies(*, sr=22050, win_length=384, hop_length=512):
    """Compute the frequencies (in beats per minute) corresponding
    to a Fourier tempogram matrix.

    Parameters
    ----------
    sr : number > 0
        The audio sampling rate
    win_length : int > 0
        The number of frames per analysis window
    hop_length : int > 0
        The number of samples between each bin

    Returns
    -------
    bin_frequencies : ndarray [shape=(win_length // 2 + 1 ,)]
        vector of bin frequencies measured in BPM.

    Examples
    --------
    Get the tempo frequencies corresponding to a 384-bin (8-second) tempogram

    >>> librosa.fourier_tempo_frequencies(win_length=384)
    array([ 0.   ,  0.117,  0.234, ..., 22.266, 22.383, 22.5  ])

    """

    # sr / hop_length gets the frame rate
    # multiplying by 60 turns frames / sec into frames / minute
    return fft_frequencies(sr=sr * 60 / float(hop_length), n_fft=win_length)


# A-weighting should be capitalized: suppress the naming warning
@deprecate_positional_args
def A_weighting(frequencies, *, min_db=-80.0):  # pylint: disable=invalid-name
    """Compute the A-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    A_weighting : scalar or np.ndarray [shape=(n,)]
        ``A_weighting[i]`` is the A-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    B_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the A-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.A_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)',
    ...        ylabel='Weighting (log10)',
    ...        title='A-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0

    const = np.array([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
    weights = 2.0 + 20.0 * (
        np.log10(const[0])
        + 2 * np.log10(f_sq)
        - np.log10(f_sq + const[0])
        - np.log10(f_sq + const[1])
        - 0.5 * np.log10(f_sq + const[2])
        - 0.5 * np.log10(f_sq + const[3])
    )

    return weights if min_db is None else np.maximum(min_db, weights)


@deprecate_positional_args
def B_weighting(frequencies, *, min_db=-80.0):  # pylint: disable=invalid-name
    """Compute the B-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    B_weighting : scalar or np.ndarray [shape=(n,)]
        ``B_weighting[i]`` is the B-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the B-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.B_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)',
    ...        ylabel='Weighting (log10)',
    ...        title='B-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0

    const = np.array([12194.217, 20.598997, 158.48932]) ** 2.0
    weights = 0.17 + 20.0 * (
        np.log10(const[0])
        + 1.5 * np.log10(f_sq)
        - np.log10(f_sq + const[0])
        - np.log10(f_sq + const[1])
        - 0.5 * np.log10(f_sq + const[2])
    )

    return weights if min_db is None else np.maximum(min_db, weights)


@deprecate_positional_args
def C_weighting(frequencies, *, min_db=-80.0):  # pylint: disable=invalid-name
    """Compute the C-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    C_weighting : scalar or np.ndarray [shape=(n,)]
        ``C_weighting[i]`` is the C-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    D_weighting

    Examples
    --------
    Get the C-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.C_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='C-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0

    const = np.array([12194.217, 20.598997]) ** 2.0
    weights = 0.062 + 20.0 * (
        np.log10(const[0])
        + np.log10(f_sq)
        - np.log10(f_sq + const[0])
        - np.log10(f_sq + const[1])
    )

    return weights if min_db is None else np.maximum(min_db, weights)


@deprecate_positional_args
def D_weighting(frequencies, *, min_db=-80.0):  # pylint: disable=invalid-name
    """Compute the D-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    D_weighting : scalar or np.ndarray [shape=(n,)]
        ``D_weighting[i]`` is the D-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    C_weighting

    Examples
    --------
    Get the D-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.D_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='D-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0

    const = np.array([8.3046305e-3, 1018.7, 1039.6, 3136.5, 3424, 282.7, 1160]) ** 2.0
    weights = 20.0 * (
        0.5 * np.log10(f_sq)
        - np.log10(const[0])
        + 0.5
        * (
            +np.log10((const[1] - f_sq) ** 2 + const[2] * f_sq)
            - np.log10((const[3] - f_sq) ** 2 + const[4] * f_sq)
            - np.log10(const[5] + f_sq)
            - np.log10(const[6] + f_sq)
        )
    )

    return weights if min_db is None else np.maximum(min_db, weights)


@deprecate_positional_args
def Z_weighting(frequencies, *, min_db=None):  # pylint: disable=invalid-name
    weights = np.zeros(len(frequencies))
    return weights if min_db is None else np.maximum(min_db, weights)


WEIGHTING_FUNCTIONS = {
    "A": A_weighting,
    "B": B_weighting,
    "C": C_weighting,
    "D": D_weighting,
    "Z": Z_weighting,
    None: Z_weighting,
}


@deprecate_positional_args
def frequency_weighting(frequencies, *, kind="A", **kwargs):
    """Compute the weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    kind : str in
        The weighting kind. e.g. `'A'`, `'B'`, `'C'`, `'D'`, `'Z'`
    **kwargs
        Additional keyword arguments to A_weighting, B_weighting, etc.

    Returns
    -------
    weighting : scalar or np.ndarray [shape=(n,)]
        ``weighting[i]`` is the weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the A-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.frequency_weighting(freqs, kind='A')
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='A-Weighting of CQT frequencies')
    """
    if isinstance(kind, str):
        kind = kind.upper()
    return WEIGHTING_FUNCTIONS[kind](frequencies, **kwargs)


@deprecate_positional_args
def multi_frequency_weighting(frequencies, *, kinds="ZAC", **kwargs):
    """Compute multiple weightings of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    kinds : list or tuple or str
        An iterable of weighting kinds. e.g. `('Z', 'B')`, `'ZAD'`, `'C'`
    **kwargs : keywords to pass to the weighting function.

    Returns
    -------
    weighting : scalar or np.ndarray [shape=(len(kinds), n)]
        ``weighting[i, j]`` is the weighting of ``frequencies[j]``
        using the curve determined by ``kinds[i]``.

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    A_weighting
    B_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the A, B, C, D, and Z weightings for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weightings = 'ABCDZ'
    >>> weights = librosa.multi_frequency_weighting(freqs, kinds=weightings)
    >>> fig, ax = plt.subplots()
    >>> for label, w in zip(weightings, weights):
    ...     ax.plot(freqs, w, label=label)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='Weightings of CQT frequencies')
    >>> ax.legend()
    """
    return np.stack(
        [frequency_weighting(frequencies, kind=k, **kwargs) for k in kinds], axis=0
    )


@deprecate_positional_args
def times_like(X, *, sr=22050, hop_length=512, n_fft=None, axis=-1):
    """Return an array of time values to match the time axis from a feature matrix.

    Parameters
    ----------
    X : np.ndarray or scalar
        - If ndarray, X is a feature matrix, e.g. STFT, chromagram, or mel spectrogram.
        - If scalar, X represents the number of frames.
    sr : number > 0 [scalar]
        audio sampling rate
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.
    axis : int [scalar]
        The axis representing the time axis of X.
        By default, the last axis (-1) is taken.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        ndarray of times (in seconds) corresponding to each frame of X.

    See Also
    --------
    samples_like :
        Return an array of sample indices to match the time axis from a feature matrix.

    Examples
    --------
    Provide a feature matrix input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> times = librosa.times_like(D)
    >>> times
    array([0.   , 0.023, ..., 5.294, 5.317])

    Provide a scalar input:

    >>> n_frames = 2647
    >>> times = librosa.times_like(n_frames)
    >>> times
    array([  0.00000000e+00,   2.32199546e-02,   4.64399093e-02, ...,
             6.13935601e+01,   6.14167800e+01,   6.14400000e+01])
    """
    samples = samples_like(X, hop_length=hop_length, n_fft=n_fft, axis=axis)
    return samples_to_time(samples, sr=sr)


@deprecate_positional_args
def samples_like(X, *, hop_length=512, n_fft=None, axis=-1):
    """Return an array of sample indices to match the time axis from a feature matrix.

    Parameters
    ----------
    X : np.ndarray or scalar
        - If ndarray, X is a feature matrix, e.g. STFT, chromagram, or mel spectrogram.
        - If scalar, X represents the number of frames.
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.
    axis : int [scalar]
        The axis representing the time axis of ``X``.
        By default, the last axis (-1) is taken.

    Returns
    -------
    samples : np.ndarray [shape=(n,)]
        ndarray of sample indices corresponding to each frame of ``X``.

    See Also
    --------
    times_like :
        Return an array of time values to match the time axis from a feature matrix.

    Examples
    --------
    Provide a feature matrix input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> X = librosa.stft(y)
    >>> samples = librosa.samples_like(X)
    >>> samples
    array([     0,    512, ..., 116736, 117248])

    Provide a scalar input:

    >>> n_frames = 2647
    >>> samples = librosa.samples_like(n_frames)
    >>> samples
    array([      0,     512,    1024, ..., 1353728, 1354240, 1354752])
    """
    if np.isscalar(X):
        frames = np.arange(X)
    else:
        frames = np.arange(X.shape[axis])
    return frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)


@deprecate_positional_args
def midi_to_svara_h(midi, *, Sa, abbr=True, octave=True, unicode=True):
    """Convert MIDI numbers to Hindustani svara

    Parameters
    ----------
    midi : numeric or np.ndarray
        The MIDI number or numbers to convert

    Sa : number > 0
        MIDI number of the reference Sa.

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'r', 'R', 'g', 'G', ...)

        If `False`, return long-form names ('Sa', 're', 'Re', 'ga', 'Ga', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or list of str
        The svara corresponding to the given MIDI number(s)

    See Also
    --------
    hz_to_svara_h
    note_to_svara_h
    midi_to_svara_c
    midi_to_note

    Examples
    --------
    The first three svara with Sa at midi number 60:

    >>> librosa.midi_svara_h([60, 61, 62], Sa=60)
    ['S', 'r', 'R']

    With Sa=67, midi 60-62 are in the octave below:

    >>> librosa.midi_to_svara_h([60, 61, 62], Sa=67)
    ['ṃ', 'Ṃ', 'P̣']

    Or without unicode decoration:

    >>> librosa.midi_to_svara_h([60, 61, 62], Sa=67, unicode=False)
    ['m,', 'M,', 'P,']

    Or going up an octave, with Sa=60, and using unabbreviated notes

    >>> librosa.midi_to_svara_h([72, 73, 74], Sa=60, abbr=False)
    ['Ṡa', 'ṙe', 'Ṙe']
    """

    SVARA_MAP = [
        "Sa",
        "re",
        "Re",
        "ga",
        "Ga",
        "ma",
        "Ma",
        "Pa",
        "dha",
        "Dha",
        "ni",
        "Ni",
    ]

    SVARA_MAP_SHORT = list(s[0] for s in SVARA_MAP)

    if not np.isscalar(midi):
        return [
            midi_to_svara_h(m, Sa=Sa, abbr=abbr, octave=octave, unicode=unicode)
            for m in midi
        ]

    svara_num = int(np.round(midi - Sa))

    if abbr:
        svara = SVARA_MAP_SHORT[svara_num % 12]
    else:
        svara = SVARA_MAP[svara_num % 12]

    if octave:
        if 24 > svara_num >= 12:
            if unicode:
                svara = svara[0] + "\u0307" + svara[1:]
            else:
                svara += "'"
        elif -12 <= svara_num < 0:
            if unicode:
                svara = svara[0] + "\u0323" + svara[1:]
            else:
                svara += ","

    return svara


@deprecate_positional_args
def hz_to_svara_h(frequencies, *, Sa, abbr=True, octave=True, unicode=True):
    """Convert frequencies (in Hz) to Hindustani svara

    Note that this conversion assumes 12-tone equal temperament.

    Parameters
    ----------
    frequencies : positive number or np.ndarray
        The frequencies (in Hz) to convert

    Sa : positive number
        Frequency (in Hz) of the reference Sa.

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'r', 'R', 'g', 'G', ...)

        If `False`, return long-form names ('Sa', 're', 'Re', 'ga', 'Ga', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or list of str
        The svara corresponding to the given frequency/frequencies

    See Also
    --------
    midi_to_svara_h
    note_to_svara_h
    hz_to_svara_c
    hz_to_note

    Examples
    --------
    Convert Sa in three octaves:

    >>> librosa.hz_to_svara_h([261/2, 261, 261*2], Sa=261)
    ['Ṣ', 'S', 'Ṡ']

    Convert one octave worth of frequencies with full names:

    >>> freqs = librosa.cqt_frequencies(n_bins=12, fmin=261)
    >>> librosa.hz_to_svara_h(freqs, Sa=freqs[0], abbr=False)
    ['Sa', 're', 'Re', 'ga', 'Ga', 'ma', 'Ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni']
    """

    midis = hz_to_midi(frequencies)
    return midi_to_svara_h(
        midis, Sa=hz_to_midi(Sa), abbr=abbr, octave=octave, unicode=unicode
    )


@deprecate_positional_args
def note_to_svara_h(notes, *, Sa, abbr=True, octave=True, unicode=True):
    """Convert western notes to Hindustani svara

    Note that this conversion assumes 12-tone equal temperament.

    Parameters
    ----------
    notes : str or list of str
        Notes to convert (e.g., `'C#'` or `['C4', 'Db4', 'D4']`

    Sa : str
        Note corresponding to Sa (e.g., `'C'` or `'C5'`).

        If no octave information is provided, it will default to octave 0
        (``C0`` ~= 16 Hz)

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'r', 'R', 'g', 'G', ...)

        If `False`, return long-form names ('Sa', 're', 'Re', 'ga', 'Ga', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or list of str
        The svara corresponding to the given notes

    See Also
    --------
    midi_to_svara_h
    hz_to_svara_h
    note_to_svara_c
    note_to_midi
    note_to_hz

    Examples
    --------
    >>> librosa.note_to_svara_h(['C4', 'G4', 'C5', 'G5'], Sa='C5')
    ['Ṣ', 'P̣', 'S', 'P']
    """

    midis = note_to_midi(notes, round_midi=False)

    return midi_to_svara_h(
        midis, Sa=note_to_midi(Sa), abbr=abbr, octave=octave, unicode=unicode
    )


@deprecate_positional_args
def midi_to_svara_c(midi, *, Sa, mela, abbr=True, octave=True, unicode=True):
    """Convert MIDI numbers to Carnatic svara within a given melakarta raga

    Parameters
    ----------
    midi : numeric
        The MIDI numbers to convert

    Sa : number > 0
        MIDI number of the reference Sa.

        Default: 60 (261.6 Hz, `C4`)

    mela : int or str
        The name or index of the melakarta raga

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'R1', 'R2', 'G1', 'G2', ...)

        If `False`, return long-form names ('Sa', 'Ri1', 'Ri2', 'Ga1', 'Ga2', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information and subscript
        numbers.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

    Returns
    -------
    svara : str or list of str
        The svara corresponding to the given MIDI number(s)

    See Also
    --------
    hz_to_svara_c
    note_to_svara_c
    mela_to_degrees
    mela_to_svara
    list_mela
    """
    if not np.isscalar(midi):
        return [
            midi_to_svara_c(
                m, Sa=Sa, mela=mela, abbr=abbr, octave=octave, unicode=unicode
            )
            for m in midi
        ]

    svara_num = int(np.round(midi - Sa))

    svara_map = notation.mela_to_svara(mela, abbr=abbr, unicode=unicode)

    svara = svara_map[svara_num % 12]

    if octave:
        if 24 > svara_num >= 12:
            if unicode:
                svara = svara[0] + "\u0307" + svara[1:]
            else:
                svara += "'"
        elif -12 <= svara_num < 0:
            if unicode:
                svara = svara[0] + "\u0323" + svara[1:]
            else:
                svara += ","

    return svara


@deprecate_positional_args
def hz_to_svara_c(frequencies, *, Sa, mela, abbr=True, octave=True, unicode=True):
    """Convert frequencies (in Hz) to Carnatic svara

    Note that this conversion assumes 12-tone equal temperament.

    Parameters
    ----------
    frequencies : positive number or np.ndarray
        The frequencies (in Hz) to convert

    Sa : positive number
        Frequency (in Hz) of the reference Sa.

    mela : int [1, 72] or string
        The melakarta raga to use.

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'R1', 'R2', 'G1', 'G2', ...)

        If `False`, return long-form names ('Sa', 'Ri1', 'Ri2', 'Ga1', 'Ga2', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or list of str
        The svara corresponding to the given frequency/frequencies

    See Also
    --------
    note_to_svara_c
    midi_to_svara_c
    hz_to_svara_h
    hz_to_note
    list_mela

    Examples
    --------
    Convert Sa in three octaves:

    >>> librosa.hz_to_svara_c([261/2, 261, 261*2], Sa=261, mela='kanakangi')
    ['Ṣ', 'S', 'Ṡ']

    Convert one octave worth of frequencies using melakarta #36:

    >>> freqs = librosa.cqt_frequencies(n_bins=12, fmin=261)
    >>> librosa.hz_to_svara_c(freqs, Sa=freqs[0], mela=36)
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'D₃', 'N₃']
    """

    midis = hz_to_midi(frequencies)
    return midi_to_svara_c(
        midis, Sa=hz_to_midi(Sa), mela=mela, abbr=abbr, octave=octave, unicode=unicode
    )


@deprecate_positional_args
def note_to_svara_c(notes, *, Sa, mela, abbr=True, octave=True, unicode=True):
    """Convert western notes to Carnatic svara

    Note that this conversion assumes 12-tone equal temperament.

    Parameters
    ----------
    notes : str or list of str
        Notes to convert (e.g., `'C#'` or `['C4', 'Db4', 'D4']`

    Sa : str
        Note corresponding to Sa (e.g., `'C'` or `'C5'`).

        If no octave information is provided, it will default to octave 0
        (``C0`` ~= 16 Hz)

    mela : str or int [1, 72]
        Melakarta raga name or index

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'R1', 'R2', 'G1', 'G2', ...)

        If `False`, return long-form names ('Sa', 'Ri1', 'Ri2', 'Ga1', 'Ga2', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or list of str
        The svara corresponding to the given notes

    See Also
    --------
    midi_to_svara_c
    hz_to_svara_c
    note_to_svara_h
    note_to_midi
    note_to_hz
    list_mela

    Examples
    --------
    >>> librosa.note_to_svara_h(['C4', 'G4', 'C5', 'D5', 'G5'], Sa='C5', mela=1)
    ['Ṣ', 'P̣', 'S', 'G₁', 'P']
    """
    midis = note_to_midi(notes, round_midi=False)

    return midi_to_svara_c(
        midis, Sa=note_to_midi(Sa), mela=mela, abbr=abbr, octave=octave, unicode=unicode
    )
