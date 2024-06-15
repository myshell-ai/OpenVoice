# -*- coding: utf-8 -*-
"""CYRILLIC 相关的几个拼音风格实现:

Style.CYRILLIC
Style.CYRILLIC_FIRST
"""
from __future__ import unicode_literals
import re

from pypinyin.constants import Style
from pypinyin.style import register
from pypinyin.style._constants import RE_TONE3
from pypinyin.style._utils import replace_symbol_to_number

# 俄语转换表
CYRILLIC_REPLACE = (
    (re.compile(r'ong'), 'ung'),
    (re.compile(r'([zcs])i'), '\\1U'),
    (re.compile(r'([xqj])u'), '\\1v'),
    (re.compile(r'^wu(.?)$'), 'u\\1'),
    (re.compile(r'(.+)r(.?)$'), '\\1R\\2'),
    (re.compile(r'^zh'), 'Cr'),
    (re.compile(r'^ch'), 'C'),
    (re.compile(r'^j'), 'qZ'),
    (re.compile(r'^z'), 'qZ'),
    (re.compile(r'^x'), 's'),
    (re.compile(r'^sh'), 'S'),
    (re.compile(r'([^CSdst])uo'), '\\1o'),
    (re.compile(r'^y(.*)$'), 'I\\1'),
    (re.compile(r'Iai'), 'AI'),
    (re.compile(r'Ia'), 'A'),
    (re.compile(r'Ie'), 'E'),
    (re.compile(r'Ii'), 'i'),
    (re.compile(r'Iou'), 'V'),
    (re.compile(r'Iu'), 'v'),
    (re.compile(r'(.v)(\d?)$'), '\\1I\\2'),
    (re.compile(r'Io'), 'O'),
    (re.compile(r'iu'), 'v'),
    (re.compile(r'ie'), 'E'),
    (re.compile(r'hui'), 'huei'),
    (re.compile(r'ui'), 'uI'),
    (re.compile(r'ai'), 'aI'),
    (re.compile(r'ei'), 'eI'),
    (re.compile(r'ia'), 'A'),
    (re.compile(r'(.*[^h])n([^g]?)$'), '\\1nM\\2'),
    (re.compile(r'(.*[^h])ng(.?)$'), '\\1n\\2'),
    (re.compile(r'^v(\d?$)'), 'vI'),
)
CYRILLIC_TABLE = dict(zip(
    u'abwgdEOrZiIklmnopRstufhqcCSHTMUevAV',
    u'абвгдеёжзийклмнопрстуфхццчшщъьыэюяю'
))


class CyrillicfoConverter(object):
    def to_cyrillic(self, pinyin, **kwargs):
        pinyin = self._pre_convert(pinyin)
        # 查表替换成注音
        for find_re, replace in CYRILLIC_REPLACE:
            pinyin = find_re.sub(replace, pinyin)
        pinyin = ''.join(CYRILLIC_TABLE.get(x, x) for x in pinyin)
        return pinyin

    def to_cyrillic_first(self, pinyin, **kwargs):
        pinyin = self.to_cyrillic(pinyin, **kwargs)
        return pinyin[0]

    def _pre_convert(self, pinyin):
        # 用数字表示声调
        pinyin = replace_symbol_to_number(pinyin)
        # 将声调数字移动到最后
        return RE_TONE3.sub(r'\1\3\2', pinyin)


converter = CyrillicfoConverter()
register(Style.CYRILLIC, func=converter.to_cyrillic)
register(Style.CYRILLIC_FIRST, func=converter.to_cyrillic_first)
