# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import re

from pypinyin import phonetic_symbol

# 声母表
_INITIALS = [
    'b',
    'p',
    'm',
    'f',
    'd',
    't',
    'n',
    'l',
    'g',
    'k',
    'h',
    'j',
    'q',
    'x',
    'zh',
    'ch',
    'sh',
    'r',
    'z',
    'c',
    's',
]
# 声母表, 把 y, w 也当作声母
_INITIALS_NOT_STRICT = _INITIALS + ['y', 'w']

# 韵母表
_FINALS = [
    'i',
    'u',
    'ü',
    'a',
    'ia',
    'ua',
    'o',
    'uo',
    'e',
    'ie',
    'üe',
    'ai',
    'uai',
    'ei',
    'uei',
    'ao',
    'iao',
    'ou',
    'iou',
    'an',
    'ian',
    'uan',
    'üan',
    'en',
    'in',
    'uen',
    'ün',
    'ang',
    'iang',
    'uang',
    'eng',
    'ing',
    'ueng',
    'ong',
    'iong',

    'er',
    'ê',
]

# 带声调字符与数字表示声调的对应关系
PHONETIC_SYMBOL_DICT = phonetic_symbol.phonetic_symbol.copy()
PHONETIC_SYMBOL_DICT_KEY_LENGTH_NOT_ONE = dict(
    (k, v)
    for k, v in PHONETIC_SYMBOL_DICT.items()
    if len(k) > 1
)
# 匹配带声调字符的正则表达式
RE_PHONETIC_SYMBOL = re.compile(
    r'[{0}]'.format(
        re.escape(''.join(x for x in PHONETIC_SYMBOL_DICT if len(x) == 1))
    )
)

# 匹配使用数字标识声调的字符的正则表达式
RE_TONE2 = re.compile(r'([aeoiuvnmêü])([1-5])$')
# 匹配 TONE2 中标识韵母声调的正则表达式
RE_TONE3 = re.compile(r'^([a-zêü]+)([1-5])([a-zêü]*)$')

# 匹配单个数字
RE_NUMBER = re.compile(r'\d')
