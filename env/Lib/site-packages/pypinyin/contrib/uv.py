# -*- coding: utf-8 -*-
from __future__ import unicode_literals


class V2UMixin(object):
    """无声调相关拼音风格下的结果使用 ``ü`` 代替原来的 ``v``

    使用方法::

        from pypinyin import lazy_pinyin, Style
        from pypinyin.contrib.uv import V2UMixin
        from pypinyin.converter import DefaultConverter
        from pypinyin.core import Pinyin

        # 原来的结果中会使用 ``v`` 表示 ``ü``
        print(lazy_pinyin('战略'))
        # 输出：['zhan', 'lve']


        class MyConverter(V2UMixin, DefaultConverter):
            pass

        my_pinyin = Pinyin(MyConverter())
        pinyin = my_pinyin.pinyin
        lazy_pinyin = my_pinyin.lazy_pinyin

        #  新的结果中使用 ``ü`` 代替原来的 ``v``
        print(lazy_pinyin('战略'))
        # 输出: ['zhan', 'lüe']

        print(pinyin('战略', style=Style.NORMAL))
        # 输出：[['zhan'], ['lüe']]


    """

    def post_convert_style(self, han, orig_pinyin, converted_pinyin,
                           style, strict, **kwargs):
        pre_data = super(V2UMixin, self).post_convert_style(
            han, orig_pinyin, converted_pinyin, style, strict, **kwargs)

        if pre_data is not None:
            converted_pinyin = pre_data

        return converted_pinyin.replace('v', 'ü')
