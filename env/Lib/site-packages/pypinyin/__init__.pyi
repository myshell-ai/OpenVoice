from typing import Any

from . import compat
from . import constants
from . import core

__title__ = ...  # type: Any
__version__ = ...  # type: Any
__author__ = ...  # type: Any
__license__ = ...  # type: Any
__copyright__ = ...  # type: Any
__all__ = ...  # type: Any

PY2 = compat.PY2

Style = constants.Style
STYLE_NORMAL = constants.STYLE_NORMAL
NORMAL = constants.NORMAL
STYLE_TONE = constants.STYLE_TONE
TONE = constants.TONE
STYLE_TONE2 = constants.STYLE_TONE2
TONE2 = constants.TONE2
STYLE_TONE3 = constants.STYLE_TONE3
TONE3 = constants.TONE3
STYLE_INITIALS = constants.STYLE_INITIALS
INITIALS = constants.INITIALS
STYLE_FIRST_LETTER = constants.STYLE_FIRST_LETTER
FIRST_LETTER = constants.FIRST_LETTER
STYLE_FINALS = constants.STYLE_FINALS
FINALS = constants.FINALS
STYLE_FINALS_TONE = constants.STYLE_FINALS_TONE
FINALS_TONE = constants.FINALS_TONE
STYLE_FINALS_TONE2 = constants.STYLE_FINALS_TONE2
FINALS_TONE2 = constants.FINALS_TONE2
STYLE_FINALS_TONE3 = constants.STYLE_FINALS_TONE3
FINALS_TONE3 = constants.FINALS_TONE3
STYLE_BOPOMOFO = constants.STYLE_BOPOMOFO
BOPOMOFO = constants.BOPOMOFO
STYLE_BOPOMOFO_FIRST = constants.STYLE_BOPOMOFO_FIRST
BOPOMOFO_FIRST = constants.BOPOMOFO_FIRST
STYLE_CYRILLIC = constants.STYLE_CYRILLIC
CYRILLIC = constants.CYRILLIC
STYLE_CYRILLIC_FIRST = constants.STYLE_CYRILLIC_FIRST
CYRILLIC_FIRST = constants.CYRILLIC_FIRST

pinyin = core.pinyin
lazy_pinyin = core.lazy_pinyin
slug = core.slug
load_single_dict = core.load_single_dict
load_phrases_dict = core.load_phrases_dict
