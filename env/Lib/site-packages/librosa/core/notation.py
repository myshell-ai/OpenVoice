#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Music notation utilities"""

import re
import numpy as np
from .._cache import cache
from ..util.exceptions import ParameterError
from ..util.decorators import deprecate_positional_args

__all__ = [
    "key_to_degrees",
    "key_to_notes",
    "mela_to_degrees",
    "mela_to_svara",
    "thaat_to_degrees",
    "list_mela",
    "list_thaat",
]

THAAT_MAP = dict(
    bilaval=[0, 2, 4, 5, 7, 9, 11],
    khamaj=[0, 2, 4, 5, 7, 9, 10],
    kafi=[0, 2, 3, 5, 7, 9, 10],
    asavari=[0, 2, 3, 5, 7, 8, 10],
    bhairavi=[0, 1, 3, 5, 7, 8, 10],
    kalyan=[0, 2, 4, 6, 7, 9, 11],
    marva=[0, 1, 4, 6, 7, 9, 11],
    poorvi=[0, 1, 4, 6, 7, 8, 11],
    todi=[0, 1, 3, 6, 7, 8, 11],
    bhairav=[0, 1, 4, 5, 7, 8, 11],
)

# Enumeration will start from 1
MELAKARTA_MAP = {
    k: i
    for i, k in enumerate(
        [
            "kanakangi",
            "ratnangi",
            "ganamurthi",
            "vanaspathi",
            "manavathi",
            "tanarupi",
            "senavathi",
            "hanumathodi",
            "dhenuka",
            "natakapriya",
            "kokilapriya",
            "rupavathi",
            "gayakapriya",
            "vakulabharanam",
            "mayamalavagaula",
            "chakravakom",
            "suryakantham",
            "hatakambari",
            "jhankaradhwani",
            "natabhairavi",
            "keeravani",
            "kharaharapriya",
            "gaurimanohari",
            "varunapriya",
            "mararanjini",
            "charukesi",
            "sarasangi",
            "harikambhoji",
            "dheerasankarabharanam",
            "naganandini",
            "yagapriya",
            "ragavardhini",
            "gangeyabhushani",
            "vagadheeswari",
            "sulini",
            "chalanatta",
            "salagam",
            "jalarnavam",
            "jhalavarali",
            "navaneetham",
            "pavani",
            "raghupriya",
            "gavambodhi",
            "bhavapriya",
            "subhapanthuvarali",
            "shadvidhamargini",
            "suvarnangi",
            "divyamani",
            "dhavalambari",
            "namanarayani",
            "kamavardhini",
            "ramapriya",
            "gamanasrama",
            "viswambhari",
            "syamalangi",
            "shanmukhapriya",
            "simhendramadhyamam",
            "hemavathi",
            "dharmavathi",
            "neethimathi",
            "kanthamani",
            "rishabhapriya",
            "latangi",
            "vachaspathi",
            "mechakalyani",
            "chitrambari",
            "sucharitra",
            "jyotisvarupini",
            "dhatuvardhini",
            "nasikabhushani",
            "kosalam",
            "rasikapriya",
        ],
        1,
    )
}


def thaat_to_degrees(thaat):
    """Construct the svara indices (degrees) for a given thaat

    Parameters
    ----------
    thaat : str
        The name of the thaat

    Returns
    -------
    indices : np.ndarray
        A list of the seven svara indices (starting from 0=Sa)
        contained in the specified thaat

    See Also
    --------
    key_to_degrees
    mela_to_degrees
    list_thaat

    Examples
    --------
    >>> librosa.thaat_to_degrees('bilaval')
    array([ 0,  2,  4,  5,  7,  9, 11])

    >>> librosa.thaat_to_degrees('todi')
    array([ 0,  1,  3,  6,  7,  8, 11])
    """
    return np.asarray(THAAT_MAP[thaat.lower()])


def mela_to_degrees(mela):
    """Construct the svara indices (degrees) for a given melakarta raga

    Parameters
    ----------
    mela : str or int
        Either the name or integer index ([1, 2, ..., 72]) of the melakarta raga

    Returns
    -------
    degrees : np.ndarray
        A list of the seven svara indices (starting from 0=Sa)
        contained in the specified raga

    See Also
    --------
    thaat_to_degrees
    key_to_degrees
    list_mela

    Examples
    --------
    Melakarta #1 (kanakangi):

    >>> librosa.mela_to_degrees(1)
    array([0, 1, 2, 5, 7, 8, 9])

    Or using a name directly:

    >>> librosa.mela_to_degrees('kanakangi')
    array([0, 1, 2, 5, 7, 8, 9])
    """

    if isinstance(mela, str):
        index = MELAKARTA_MAP[mela.lower()] - 1
    elif 0 < mela <= 72:
        index = mela - 1
    else:
        raise ParameterError("mela={} must be in range [1, 72]".format(mela))

    # always have Sa [0]
    degrees = [0]

    # Fill in Ri and Ga
    lower = index % 36
    if 0 <= lower < 6:
        # Ri1, Ga1
        degrees.extend([1, 2])
    elif 6 <= lower < 12:
        # Ri1, Ga2
        degrees.extend([1, 3])
    elif 12 <= lower < 18:
        # Ri1, Ga3
        degrees.extend([1, 4])
    elif 18 <= lower < 24:
        # Ri2, Ga2
        degrees.extend([2, 3])
    elif 24 <= lower < 30:
        # Ri2, Ga3
        degrees.extend([2, 4])
    else:
        # Ri3, Ga3
        degrees.extend([3, 4])

    # Determine Ma
    if index < 36:
        # Ma1
        degrees.append(5)
    else:
        # Ma2
        degrees.append(6)

    # always have Pa [7]
    degrees.append(7)

    # Determine Dha and Ni
    upper = index % 6
    if upper == 0:
        # Dha1, Ni1
        degrees.extend([8, 9])
    elif upper == 1:
        # Dha1, Ni2
        degrees.extend([8, 10])
    elif upper == 2:
        # Dha1, Ni3
        degrees.extend([8, 11])
    elif upper == 3:
        # Dha2, Ni2
        degrees.extend([9, 10])
    elif upper == 4:
        # Dha2, Ni3
        degrees.extend([9, 11])
    else:
        # Dha3, Ni3
        degrees.extend([10, 11])

    return np.array(degrees)


@deprecate_positional_args
@cache(level=10)
def mela_to_svara(mela, *, abbr=True, unicode=True):
    """Spell the Carnatic svara names for a given melakarta raga

    This function exists to resolve enharmonic equivalences between
    pitch classes:

        - Ri2 / Ga1
        - Ri3 / Ga2
        - Dha2 / Ni1
        - Dha3 / Ni2

    For svara outside the raga, names are chosen to preserve orderings
    so that all Ri precede all Ga, and all Dha precede all Ni.

    Parameters
    ----------
    mela : str or int
        the name or numerical index of the melakarta raga

    abbr : bool
        If `True`, use single-letter svara names: S, R, G, ...

        If `False`, use full names: Sa, Ri, Ga, ...

    unicode : bool
        If `True`, use unicode symbols for numberings, e.g., Ri\u2081

        If `False`, use low-order ASCII, e.g., Ri1.

    Returns
    -------
    svara : list of strings

        The svara names for each of the 12 pitch classes.

    See Also
    --------
    key_to_notes
    mela_to_degrees
    list_mela

    Examples
    --------
    Melakarta #1 (Kanakangi) uses R1, G1, D1, N1

    >>> librosa.mela_to_svara(1)
    ['S', 'R₁', 'G₁', 'G₂', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'N₁', 'N₂', 'N₃']

    #19 (Jhankaradhwani) uses R2 and G2 so the third svara are Ri:

    >>> librosa.mela_to_svara(19)
    ['S', 'R₁', 'R₂', 'G₂', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'N₁', 'N₂', 'N₃']

    #31 (Yagapriya) uses R3 and G3, so third and fourth svara are Ri:

    >>> librosa.mela_to_svara(31)
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'N₁', 'N₂', 'N₃']

    #34 (Vagadheeswari) uses D2 and N2, so Ni1 becomes Dha2:

    >>> librosa.mela_to_svara(34)
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'N₂', 'N₃']

    #36 (Chalanatta) uses D3 and N3, so Ni2 becomes Dha3:

    >>> librosa.mela_to_svara(36)
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'D₃', 'N₃']

    # You can also query by raga name instead of index:

    >>> librosa.mela_to_svara('chalanatta')
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'D₃', 'N₃']
    """

    # The following will be constant for all ragas
    svara_map = [
        "Sa",
        "Ri\u2081",
        None,  # Ri2/Ga1
        None,  # Ri3/Ga2
        "Ga\u2083",
        "Ma\u2081",
        "Ma\u2082",
        "Pa",
        "Dha\u2081",
        None,  # Dha2/Ni1
        None,  # Dha3/Ni2
        "Ni\u2083",
    ]

    if isinstance(mela, str):
        mela_idx = MELAKARTA_MAP[mela.lower()] - 1
    elif 0 < mela <= 72:
        mela_idx = mela - 1
    else:
        raise ParameterError("mela={} must be in range [1, 72]".format(mela))

    # Determine Ri2/Ga1
    lower = mela_idx % 36
    if lower < 6:
        # First six will have Ri1/Ga1
        svara_map[2] = "Ga\u2081"
    else:
        # All others have either Ga2/Ga3
        # So we'll call this Ri2
        svara_map[2] = "Ri\u2082"

    # Determine Ri3/Ga2
    if lower < 30:
        # First thirty should get Ga2
        svara_map[3] = "Ga\u2082"
    else:
        # Only the last six have Ri3
        svara_map[3] = "Ri\u2083"

    upper = mela_idx % 6

    # Determine Dha2/Ni1
    if upper == 0:
        # these are the only ones with Ni1
        svara_map[9] = "Ni\u2081"
    else:
        # Everyone else has Dha2
        svara_map[9] = "Dha\u2082"

    # Determine Dha3/Ni2
    if upper == 5:
        # This one has Dha3
        svara_map[10] = "Dha\u2083"
    else:
        # Everyone else has Ni2
        svara_map[10] = "Ni\u2082"

    if abbr:
        svara_map = [
            s.translate(str.maketrans({"a": "", "h": "", "i": ""})) for s in svara_map
        ]

    if not unicode:
        svara_map = [
            s.translate(str.maketrans({"\u2081": "1", "\u2082": "2", "\u2083": "3"}))
            for s in svara_map
        ]

    return list(svara_map)


def list_mela():
    """List melakarta ragas by name and index.

    Melakarta raga names are transcribed from [#]_, with the exception of #45
    (subhapanthuvarali).

    .. [#] Bhagyalekshmy, S. (1990).
        Ragas in Carnatic music.
        South Asia Books.

    Returns
    -------
    mela_map : dict
        A dictionary mapping melakarta raga names to indices (1, 2, ..., 72)

    Examples
    --------
    >>> librosa.list_mela()
    {'kanakangi': 1,
     'ratnangi': 2,
     'ganamurthi': 3,
     'vanaspathi': 4,
     ...}

    See Also
    --------
    mela_to_degrees
    mela_to_svara
    list_thaat
    """
    return MELAKARTA_MAP.copy()


def list_thaat():
    """List supported thaats by name.

    Returns
    -------
    thaats : list
        A list of supported thaats

    Examples
    --------
    >>> librosa.list_thaat()
    ['bilaval',
     'khamaj',
     'kafi',
     'asavari',
     'bhairavi',
     'kalyan',
     'marva',
     'poorvi',
     'todi',
     'bhairav']

    See Also
    --------
    list_mela
    thaat_to_degrees
    """
    return list(THAAT_MAP.keys())


@deprecate_positional_args
@cache(level=10)
def key_to_notes(key, *, unicode=True):
    """Lists all 12 note names in the chromatic scale, as spelled according to
    a given key (major or minor).

    This function exists to resolve enharmonic equivalences between different
    spellings for the same pitch (e.g. C♯ vs D♭), and is primarily useful when producing
    human-readable outputs (e.g. plotting) for pitch content.

    Note names are decided by the following rules:

    1. If the tonic of the key has an accidental (sharp or flat), that accidental will be
       used consistently for all notes.

    2. If the tonic does not have an accidental, accidentals will be inferred to minimize
       the total number used for diatonic scale degrees.

    3. If there is a tie (e.g., in the case of C:maj vs A:min), sharps will be preferred.

    Parameters
    ----------
    key : string
        Must be in the form TONIC:key.  Tonic must be upper case (``CDEFGAB``),
        key must be lower-case (``maj`` or ``min``).

        Single accidentals (``b!♭`` for flat, or ``#♯`` for sharp) are supported.

        Examples: ``C:maj, Db:min, A♭:min``.

    unicode : bool
        If ``True`` (default), use Unicode symbols (♯𝄪♭𝄫)for accidentals.

        If ``False``, Unicode symbols will be mapped to low-order ASCII representations::

            ♯ -> #, 𝄪 -> ##, ♭ -> b, 𝄫 -> bb

    Returns
    -------
    notes : list
        ``notes[k]`` is the name for semitone ``k`` (starting from C)
        under the given key.  All chromatic notes (0 through 11) are
        included.

    See Also
    --------
    midi_to_note

    Examples
    --------
    `C:maj` will use all sharps

    >>> librosa.key_to_notes('C:maj')
    ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

    `A:min` has the same notes

    >>> librosa.key_to_notes('A:min')
    ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

    `A♯:min` will use sharps, but spell note 0 (`C`) as `B♯`

    >>> librosa.key_to_notes('A#:min')
    ['B♯', 'C♯', 'D', 'D♯', 'E', 'E♯', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

    `G♯:maj` will use a double-sharp to spell note 7 (`G`) as `F𝄪`:

    >>> librosa.key_to_notes('G#:maj')
    ['B♯', 'C♯', 'D', 'D♯', 'E', 'E♯', 'F♯', 'F𝄪', 'G♯', 'A', 'A♯', 'B']

    `F♭:min` will use double-flats

    >>> librosa.key_to_notes('Fb:min')
    ['D𝄫', 'D♭', 'E𝄫', 'E♭', 'F♭', 'F', 'G♭', 'A𝄫', 'A♭', 'B𝄫', 'B♭', 'C♭']
    """

    # Parse the key signature
    match = re.match(
        r"^(?P<tonic>[A-Ga-g])"
        r"(?P<accidental>[#♯b!♭]?)"
        r":(?P<scale>(maj|min)(or)?)$",
        key,
    )
    if not match:
        raise ParameterError("Improper key format: {:s}".format(key))

    pitch_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    acc_map = {"#": 1, "": 0, "b": -1, "!": -1, "♯": 1, "♭": -1}

    tonic = match.group("tonic").upper()
    accidental = match.group("accidental")
    offset = acc_map[accidental]

    scale = match.group("scale")[:3].lower()

    # Determine major or minor
    major = scale == "maj"

    # calculate how many clockwise steps we are on CoF (== # sharps)
    if major:
        tonic_number = ((pitch_map[tonic] + offset) * 7) % 12
    else:
        tonic_number = ((pitch_map[tonic] + offset) * 7 + 9) % 12

    # Decide if using flats or sharps
    # Logic here is as follows:
    #   1. respect the given notation for the tonic.
    #      Sharp tonics will always use sharps, likewise flats.
    #   2. If no accidental in the tonic, try to minimize accidentals.
    #   3. If there's a tie for accidentals, use sharp for major and flat for minor.

    if offset < 0:
        # use flats explicitly
        use_sharps = False

    elif offset > 0:
        # use sharps explicitly
        use_sharps = True

    elif 0 <= tonic_number < 6:
        use_sharps = True

    elif tonic_number > 6:
        use_sharps = False

    # Basic note sequences for simple keys
    notes_sharp = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
    notes_flat = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]

    # These apply when we have >= 6 sharps
    sharp_corrections = [
        (5, "E♯"),
        (0, "B♯"),
        (7, "F𝄪"),
        (2, "C𝄪"),
        (9, "G𝄪"),
        (4, "D𝄪"),
        (11, "A𝄪"),
    ]

    # These apply when we have >= 6 flats
    flat_corrections = [
        (11, "C♭"),
        (4, "F♭"),
        (9, "B𝄫"),
        (2, "E𝄫"),
        (7, "A𝄫"),
        (0, "D𝄫"),
    ]  # last would be (5, 'G𝄫')

    # Apply a mod-12 correction to distinguish B#:maj from C:maj
    n_sharps = tonic_number
    if tonic_number == 0 and tonic == "B":
        n_sharps = 12

    if use_sharps:
        # This will only execute if n_sharps >= 6
        for n in range(0, n_sharps - 6 + 1):
            index, name = sharp_corrections[n]
            notes_sharp[index] = name

        notes = notes_sharp
    else:
        n_flats = (12 - tonic_number) % 12

        # This will only execute if tonic_number <= 6
        for n in range(0, n_flats - 6 + 1):
            index, name = flat_corrections[n]
            notes_flat[index] = name

        notes = notes_flat

    # Finally, apply any unicode down-translation if necessary
    if not unicode:
        translations = str.maketrans({"♯": "#", "𝄪": "##", "♭": "b", "𝄫": "bb"})
        notes = list(n.translate(translations) for n in notes)

    return notes


def key_to_degrees(key):
    """Construct the diatonic scale degrees for a given key.

    Parameters
    ----------
    key : str
        Must be in the form TONIC:key.  Tonic must be upper case (``CDEFGAB``),
        key must be lower-case (``maj`` or ``min``).

        Single accidentals (``b!♭`` for flat, or ``#♯`` for sharp) are supported.

        Examples: ``C:maj, Db:min, A♭:min``.

    Returns
    -------
    degrees : np.ndarray
        An array containing the semitone numbers (0=C, 1=C#, ... 11=B)
        for each of the seven scale degrees in the given key, starting
        from the tonic.

    See Also
    --------
    key_to_notes

    Examples
    --------
    >>> librosa.key_to_degrees('C:maj')
    array([ 0,  2,  4,  5,  7,  9, 11])

    >>> librosa.key_to_degrees('C#:maj')
    array([ 1,  3,  5,  6,  8, 10,  0])

    >>> librosa.key_to_degrees('A:min')
    array([ 9, 11,  0,  2,  4,  5,  7])

    """
    notes = dict(
        maj=np.array([0, 2, 4, 5, 7, 9, 11]), min=np.array([0, 2, 3, 5, 7, 8, 10])
    )

    match = re.match(
        r"^(?P<tonic>[A-Ga-g])"
        r"(?P<accidental>[#♯b!♭]?)"
        r":(?P<scale>(maj|min)(or)?)$",
        key,
    )
    if not match:
        raise ParameterError("Improper key format: {:s}".format(key))

    pitch_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    acc_map = {"#": 1, "": 0, "b": -1, "!": -1, "♯": 1, "♭": -1}
    tonic = match.group("tonic").upper()
    accidental = match.group("accidental")
    offset = acc_map[accidental]

    scale = match.group("scale")[:3].lower()

    return (notes[scale] + pitch_map[tonic] + offset) % 12
