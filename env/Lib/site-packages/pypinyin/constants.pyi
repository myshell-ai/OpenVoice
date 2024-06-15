from enum import IntEnum, unique
from typing import Dict, List, Any, Text

PHRASES_DICT = ...  # type: Dict[Text, List[List[Text]]]

PINYIN_DICT = ...  # type: Dict[int, Text]

RE_TONE2 = ...  # type: Any

RE_HANS = ...  # type: Any


@unique
class Style(IntEnum):

    NORMAL = ...

    TONE = ...

    TONE2 = ...

    TONE3 = ...

    INITIALS = ...

    FIRST_LETTER = ...

    FINALS = ...

    FINALS_TONE = ...

    FINALS_TONE2 = ...

    FINALS_TONE3 = ...

    BOPOMOFO = ...

    BOPOMOFO_FIRST = ...

    CYRILLIC = ...

    CYRILLIC_FIRST = ...

    WADEGILES = ...


NORMAL = ...  # type: Style
STYLE_NORMAL = ...  # type: Style
TONE = ...  # type: Style
STYLE_TONE = ...  # type: Style
TONE2  = ...  # type: Style
STYLE_TONE2 = ...  # type: Style
TONE3 = ...  # type: Style
STYLE_TONE3 = ...  # type: Style
INITIALS = ...  # type: Style
STYLE_INITIALS = ...  # type: Style
FIRST_LETTER  = ...  # type: Style
STYLE_FIRST_LETTER = ...  # type: Style
FINALS = ...  # type: Style
STYLE_FINALS = ...  # type: Style
FINALS_TONE  = ...  # type: Style
STYLE_FINALS_TONE = ...  # type: Style
FINALS_TONE2 = ...  # type: Style
STYLE_FINALS_TONE2 = ...  # type: Style
FINALS_TONE3 = ...  # type: Style
STYLE_FINALS_TONE3 = ...  # type: Style
BOPOMOFO = ...  # type: Style
STYLE_BOPOMOFO = ...  # type: Style
BOPOMOFO_FIRST = ...  # type: Style
STYLE_BOPOMOFO_FIRST = ...  # type: Style
CYRILLIC = ...  # type: Style
STYLE_CYRILLIC = ...  # type: Style
CYRILLIC_FIRST = ...  # type: Style
STYLE_CYRILLIC_FIRST = ...  # type: Style
