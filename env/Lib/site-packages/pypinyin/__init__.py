#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""汉字拼音转换工具."""

from __future__ import unicode_literals

from pypinyin.compat import PY2
from pypinyin.constants import (  # noqa
    Style,
    STYLE_NORMAL, NORMAL,
    STYLE_TONE, TONE,
    STYLE_TONE2, TONE2,
    STYLE_TONE3, TONE3,
    STYLE_INITIALS, INITIALS,
    STYLE_FIRST_LETTER, FIRST_LETTER,
    STYLE_FINALS, FINALS,
    STYLE_FINALS_TONE, FINALS_TONE,
    STYLE_FINALS_TONE2, FINALS_TONE2,
    STYLE_FINALS_TONE3, FINALS_TONE3,
    STYLE_BOPOMOFO, BOPOMOFO,
    STYLE_BOPOMOFO_FIRST, BOPOMOFO_FIRST,
    STYLE_CYRILLIC, CYRILLIC,
    STYLE_CYRILLIC_FIRST, CYRILLIC_FIRST
)
from pypinyin.core import (     # noqa
    pinyin, lazy_pinyin, slug, load_single_dict, load_phrases_dict
)

__title__ = 'pypinyin'
__version__ = '0.50.0'
__author__ = 'mozillazg, 闲耘'
__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2016 mozillazg, 闲耘'
__all__ = [
    'pinyin', 'lazy_pinyin', 'slug',
    'load_single_dict', 'load_phrases_dict',
    'Style',
    'STYLE_NORMAL', 'NORMAL',
    'STYLE_TONE', 'TONE',
    'STYLE_TONE2', 'TONE2',
    'STYLE_TONE3', 'TONE3',
    'STYLE_INITIALS', 'INITIALS',
    'STYLE_FINALS', 'FINALS',
    'STYLE_FINALS_TONE', 'FINALS_TONE',
    'STYLE_FINALS_TONE2', 'FINALS_TONE2',
    'STYLE_FINALS_TONE3', 'FINALS_TONE3',
    'STYLE_FIRST_LETTER', 'FIRST_LETTER',
    'STYLE_BOPOMOFO', 'BOPOMOFO',
    'STYLE_BOPOMOFO_FIRST', 'BOPOMOFO_FIRST',
    'STYLE_CYRILLIC', 'CYRILLIC',
    'STYLE_CYRILLIC_FIRST', 'CYRILLIC_FIRST'
]
if PY2:
    # fix "TypeError: Item in ``from list'' not a string" on Python 2
    __all__ = [x.encode('utf-8') for x in __all__]
