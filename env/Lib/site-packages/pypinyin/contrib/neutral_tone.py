# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re

from pypinyin import Style
from pypinyin.contrib._tone_rule import right_mark_index


_re_number = re.compile(r'\d')


class NeutralToneWith5Mixin(object):
    """声调使用数字表示的相关拼音风格下的结果使用 5 标识轻声。

    使用方法::

        from pypinyin import lazy_pinyin, Style
        from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
        from pypinyin.converter import DefaultConverter
        from pypinyin.core import Pinyin

        # 原来的结果中不会标识轻声
        print(lazy_pinyin('好了', style=Style.TONE2))
        # 输出: ['ha3o', 'le']


        class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
            pass

        my_pinyin = Pinyin(MyConverter())
        pinyin = my_pinyin.pinyin
        lazy_pinyin = my_pinyin.lazy_pinyin

        #  新的结果中使用 ``5`` 标识轻声
        print(lazy_pinyin('好了', style=Style.TONE2))
        # 输出: ['ha3o', 'le5']

        print(pinyin('好了', style=Style.TONE2))
        # 输出：[['ha3o'], ['le5']]


    """

    NUMBER_TONE = (Style.TONE2, Style.TONE3, Style.FINALS_TONE2,
                   Style.FINALS_TONE3)
    NUMBER_AT_END = (Style.TONE3, Style.FINALS_TONE3)

    def post_convert_style(self, han, orig_pinyin, converted_pinyin,
                           style, strict, **kwargs):
        pre_data = super(NeutralToneWith5Mixin, self).post_convert_style(
            han, orig_pinyin, converted_pinyin, style, strict, **kwargs)

        if style not in self.NUMBER_TONE:
            return pre_data

        if pre_data is not None:
            converted_pinyin = pre_data
        if not converted_pinyin:    # 空字符串
            return converted_pinyin
        # 有声调，跳过
        if _re_number.search(converted_pinyin):
            return converted_pinyin

        if style in self.NUMBER_AT_END:
            return '{}5'.format(converted_pinyin)

        # 找到应该在哪个字母上标声调
        mark_index = right_mark_index(converted_pinyin)
        before = converted_pinyin[:mark_index + 1]
        after = converted_pinyin[mark_index + 1:]

        return '{}5{}'.format(before, after)
