# -*- coding: utf-8 -*-
from __future__ import unicode_literals

"""
标调位置

    有 ɑ 不放过，

　　没 ɑ 找 o、e；

　　ɑ、o、e、i、u、ü

　　标调就按这顺序；

　　i、u 若是连在一起，

　　谁在后面就标谁。

http://www.hwjyw.com/resource/content/2010/06/04/8183.shtml
https://www.zhihu.com/question/23655297
https://github.com/mozillazg/python-pinyin/issues/160
http://www.pinyin.info/rules/where.html
"""

# TODO: 增加测试用例：使用 pinyin_dict 中的数据收集所有带声调拼音测试这个规则的正确性
#       1. 收集所有带声调拼音
#       2. 转换为数字声调拼音(tone3)，然后再转换为声调拼音
#       3. 比对转换后的声调拼音跟原始拼音，确保结果一致


def right_mark_index(pinyin_no_tone):
    # 'iou', 'uei', 'uen': 还原原始韵母后，还需要按照还原前的声调进行标记
    if 'iou' in pinyin_no_tone:
        return pinyin_no_tone.index('u')
    if 'uei' in pinyin_no_tone:
        return pinyin_no_tone.index('i')
    if 'uen' in pinyin_no_tone:
        return pinyin_no_tone.index('u')

    # 有 ɑ 不放过, 没 ɑ 找 o、e
    for c in ['a', 'o', 'e']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c) + len(c) - 1

    # i、u 若是连在一起，谁在后面就标谁
    for c in ['iu', 'ui']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c) + len(c) - 1

    # ɑ、o、e、i、u、ü
    for c in ['i', 'u', 'v', 'ü']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c) + len(c) - 1

    # n, m, ê
    for c in ['n', 'm', 'ê']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c) + len(c) - 1
