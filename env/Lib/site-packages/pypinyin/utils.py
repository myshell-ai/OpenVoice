#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# 用于向后兼容，TODO: 废弃
from pypinyin.seg.simpleseg import simple_seg  # noqa
from pypinyin.style._tone_convert import tone2_to_tone


# 用于向后兼容，TODO: 废弃
def _replace_tone2_style_dict_to_default(string):
    return tone2_to_tone(string)


def _remove_dup_items(lst, remove_empty=False):
    new_lst = []
    for item in lst:
        if remove_empty and not item:
            continue
        if item not in new_lst:
            new_lst.append(item)
    return new_lst


def _remove_dup_and_empty(lst_list):
    new_lst_list = []
    for lst in lst_list:
        lst = _remove_dup_items(lst, remove_empty=True)
        if lst:
            new_lst_list.append(lst)
        else:
            new_lst_list.append([''])

    return new_lst_list
