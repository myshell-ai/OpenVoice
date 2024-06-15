# -*- coding: utf-8 -*-
"""韵母相关拼音风格:

Style.FINALS
Style.FINALS_TONE
Style.FINALS_TONE2
Style.FINALS_TONE3
"""
from __future__ import unicode_literals

from pypinyin.constants import Style
from pypinyin.style import register
from pypinyin.style._tone_convert import (
    to_finals, to_finals_tone, to_finals_tone2, to_finals_tone3
)


class FinalsConverter(object):
    def to_finals(self, pinyin, **kwargs):
        """无声调韵母"""
        return to_finals(pinyin, strict=kwargs.get('strict', True))

    def to_finals_tone(self, pinyin, **kwargs):
        """声调在韵母头上"""
        return to_finals_tone(pinyin, strict=kwargs.get('strict', True))

    def to_finals_tone2(self, pinyin, **kwargs):
        """数字声调"""
        return to_finals_tone2(pinyin, strict=kwargs.get('strict', True))

    def to_finals_tone3(self, pinyin, **kwargs):
        """数字声调"""
        return to_finals_tone3(pinyin, strict=kwargs.get('strict', True))


converter = FinalsConverter()
register(Style.FINALS, func=converter.to_finals)
register(Style.FINALS_TONE, func=converter.to_finals_tone)
register(Style.FINALS_TONE2, func=converter.to_finals_tone2)
register(Style.FINALS_TONE3, func=converter.to_finals_tone3)
