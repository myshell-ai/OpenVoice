# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from itertools import chain

from pypinyin.compat import text_type, bytes_type
from pypinyin.constants import RE_HANS, PHRASES_DICT
from pypinyin.seg import mmseg


def seg(hans):
    hans = simple_seg(hans)
    ret = []
    for x in hans:
        if not RE_HANS.match(x):   # 没有拼音的字符，不再参与二次分词
            ret.append(x)
        elif PHRASES_DICT:
            ret.extend(list(mmseg.seg.cut(x)))
        else:   # 禁用了词语库，不分词
            ret.append(x)
    return ret


def simple_seg(hans):
    """将传入的字符串按是否是汉字来分割"""
    assert not isinstance(hans, bytes_type), \
        'must be unicode string or [unicode, ...] list'

    if isinstance(hans, text_type):
        return _seg(hans)
    else:
        hans = list(hans)
        if len(hans) == 1:
            return simple_seg(hans[0])
        return list(chain(*[simple_seg(x) for x in hans]))


def _seg(chars):
    """按是否是汉字进行分词"""
    s = ''  # 保存一个词
    ret = []  # 分词结果
    flag = 0  # 上一个字符是什么? 0: 汉字, 1: 不是汉字

    for n, c in enumerate(chars):
        if RE_HANS.match(c):  # 汉字, 确定 flag 的初始值
            if n == 0:  # 第一个字符
                flag = 0

            if flag == 0:
                s += c
            else:  # 上一个字符不是汉字, 分词
                ret.append(s)
                flag = 0
                s = c

        else:  # 不是汉字
            if n == 0:  # 第一个字符, 确定 flag 的初始值
                flag = 1

            if flag == 1:
                s += c
            else:  # 上一个字符是汉字, 分词
                ret.append(s)
                flag = 1
                s = c

    ret.append(s)  # 最后的词
    return ret
