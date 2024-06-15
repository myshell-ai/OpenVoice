# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import re

from pypinyin import phonetic_symbol
from pypinyin.style._tone_rule import right_mark_index
from pypinyin.style._constants import RE_TONE3, RE_TONE2
from pypinyin.style.tone import converter
from pypinyin.style._utils import (
    get_initials, replace_symbol_to_no_symbol,
    get_finals, replace_symbol_to_number
)

_re_number = re.compile(r'\d')


def to_normal(pinyin, v_to_u=False):
    """将 :py:attr:`~pypinyin.Style.TONE`、
    :py:attr:`~pypinyin.Style.TONE2` 或
    :py:attr:`~pypinyin.Style.TONE3` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音

    :param pinyin: :py:attr:`~pypinyin.Style.TONE`、
                   :py:attr:`~pypinyin.Style.TONE2` 或
                   :py:attr:`~pypinyin.Style.TONE3` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :return: :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import to_normal
      >>> to_normal('zhōng')
      'zhong'
      >>> to_normal('zho1ng')
      'zhong'
      >>> to_normal('zhong1')
      'zhong'
      >>> to_normal('lüè')
      'lve'
      >>> to_normal('lüè', v_to_u=True)
      'lüe'
    """
    s = tone_to_tone2(pinyin, v_to_u=True)
    s = tone2_to_normal(s)
    return _fix_v_u(pinyin, s, v_to_u=v_to_u)


def to_tone(pinyin):
    """将 :py:attr:`~pypinyin.Style.TONE2` 或
    :py:attr:`~pypinyin.Style.TONE3` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.TONE` 风格的拼音

    :param pinyin: :py:attr:`~pypinyin.Style.TONE2` 或
                   :py:attr:`~pypinyin.Style.TONE3` 风格的拼音
    :return: :py:attr:`~pypinyin.Style.TONE` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import to_tone
      >>> to_tone('zho1ng')
      'zhōng'
      >>> to_tone('zhong1')
      'zhōng'
    """
    pinyin = pinyin.replace('v', 'ü')
    if not _re_number.search(pinyin):
        return pinyin

    s = tone_to_tone2(pinyin)
    s = tone2_to_tone(s)
    return s


def to_tone2(pinyin, v_to_u=False, neutral_tone_with_five=False, **kwargs):
    """将 :py:attr:`~pypinyin.Style.TONE` 或
    :py:attr:`~pypinyin.Style.TONE3` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.TONE2` 风格的拼音

    :param pinyin: :py:attr:`~pypinyin.Style.TONE` 或
                   :py:attr:`~pypinyin.Style.TONE3` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :param neutral_tone_with_five: 是否使用 ``5`` 标识轻声
    :param kwargs: 用于兼容老版本的 ``neutral_tone_with_5`` 参数，当传入
                   ``neutral_tone_with_5`` 参数时，
                   将覆盖 ``neutral_tone_with_five`` 的值。
    :return: :py:attr:`~pypinyin.Style.TONE2` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import to_tone2
      >>> to_tone2('zhōng')
      'zho1ng'
      >>> to_tone2('zhong1')
      'zho1ng'
      >>> to_tone2('shang')
      'shang'
      >>> to_tone2('shang', neutral_tone_with_five=True)
      'sha5ng'
      >>> to_tone2('lüè')
      'lve4'
      >>> to_tone2('lüè', v_to_u=True)
      'lüe4'
    """
    if kwargs.get('neutral_tone_with_5', None) is not None:
        neutral_tone_with_five = kwargs['neutral_tone_with_5']
    pinyin = pinyin.replace('5', '')
    s = tone_to_tone3(
        pinyin, v_to_u=True, neutral_tone_with_five=neutral_tone_with_five)
    s = tone3_to_tone2(s)
    return _fix_v_u(pinyin, s, v_to_u)


def to_tone3(pinyin, v_to_u=False, neutral_tone_with_five=False, **kwargs):
    """将 :py:attr:`~pypinyin.Style.TONE` 或
    :py:attr:`~pypinyin.Style.TONE2` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.TONE3` 风格的拼音

    :param pinyin: :py:attr:`~pypinyin.Style.TONE` 或
                   :py:attr:`~pypinyin.Style.TONE2` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :param neutral_tone_with_five: 是否使用 ``5`` 标识轻声
    :param kwargs: 用于兼容老版本的 ``neutral_tone_with_5`` 参数，当传入
                   ``neutral_tone_with_5`` 参数时，
                   将覆盖 ``neutral_tone_with_five`` 的值。
    :return: :py:attr:`~pypinyin.Style.TONE2` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import to_tone3
      >>> to_tone3('zhōng')
      'zhong1'
      >>> to_tone3('zho1ng')
      'zhong1'
      >>> to_tone3('shang')
      'shang'
      >>> to_tone3('shang', neutral_tone_with_five=True)
      'shang5'
      >>> to_tone3('lüè')
      'lve4'
      >>> to_tone3('lüè', v_to_u=True)
      'lüe4'
    """
    if kwargs.get('neutral_tone_with_5', None) is not None:
        neutral_tone_with_five = kwargs['neutral_tone_with_5']
    pinyin = pinyin.replace('5', '')
    s = tone_to_tone2(
        pinyin, v_to_u=True, neutral_tone_with_five=neutral_tone_with_five)
    s = tone2_to_tone3(s)
    return _fix_v_u(pinyin, s, v_to_u)


def to_initials(pinyin, strict=True):
    """将 :py:attr:`~pypinyin.Style.TONE`、
    :py:attr:`~pypinyin.Style.TONE2` 、
    :py:attr:`~pypinyin.Style.TONE3` 或
    :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.INITIALS` 风格的拼音

    :param pinyin: :py:attr:`~pypinyin.Style.TONE`、
                   :py:attr:`~pypinyin.Style.TONE2` 、
                   :py:attr:`~pypinyin.Style.TONE3` 或
                   :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音
    :param strict: 返回结果是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :return: :py:attr:`~pypinyin.Style.INITIALS` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import to_initials
      >>> to_initials('zhōng')
      'zh'

    """
    return get_initials(pinyin, strict=strict)


def to_finals(pinyin, strict=True, v_to_u=False):
    """将 :py:attr:`~pypinyin.Style.TONE`、
    :py:attr:`~pypinyin.Style.TONE2` 、
    :py:attr:`~pypinyin.Style.TONE3` 或
    :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.FINALS` 风格的拼音

    :param pinyin: :py:attr:`~pypinyin.Style.TONE`、
                   :py:attr:`~pypinyin.Style.TONE2` 、
                   :py:attr:`~pypinyin.Style.TONE3` 或
                   :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音
    :param strict: 返回结果是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :return: :py:attr:`~pypinyin.Style.FINALS` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import to_finals
      >>> to_finals('zhōng')
      'ong'

    """
    new_pinyin = replace_symbol_to_no_symbol(pinyin).replace('v', 'ü')
    finals = get_finals(new_pinyin, strict=strict)
    finals = _fix_v_u(finals, finals, v_to_u)
    return finals


def to_finals_tone(pinyin, strict=True):
    """将 :py:attr:`~pypinyin.Style.TONE`、
    :py:attr:`~pypinyin.Style.TONE2` 或
    :py:attr:`~pypinyin.Style.TONE3` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.FINALS_TONE` 风格的拼音

    :param pinyin: :py:attr:`~pypinyin.Style.TONE`、
                   :py:attr:`~pypinyin.Style.TONE2` 或
                   :py:attr:`~pypinyin.Style.TONE3` 风格的拼音
    :param strict: 返回结果是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :return: :py:attr:`~pypinyin.Style.FINALS_TONE` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import to_finals_tone
      >>> to_finals_tone('zhōng')
      'ōng'

    """
    finals = to_finals_tone2(pinyin, strict=strict)

    finals = tone2_to_tone(finals)

    return finals


def to_finals_tone2(pinyin, strict=True, v_to_u=False,
                    neutral_tone_with_five=False):
    """将 :py:attr:`~pypinyin.Style.TONE`、
    :py:attr:`~pypinyin.Style.TONE2` 或
    :py:attr:`~pypinyin.Style.TONE3` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.FINALS_TONE2` 风格的拼音

    :param pinyin: :py:attr:`~pypinyin.Style.TONE`、
                   :py:attr:`~pypinyin.Style.TONE2` 或
                   :py:attr:`~pypinyin.Style.TONE3` 风格的拼音
    :param strict: 返回结果是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :param neutral_tone_with_five: 是否使用 ``5`` 标识轻声
    :return: :py:attr:`~pypinyin.Style.FINALS_TONE2` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import to_finals_tone2
      >>> to_finals_tone2('zhōng')
      'o1ng'

    """
    pinyin = pinyin.replace('5', '')
    finals = to_finals_tone3(pinyin, strict=strict, v_to_u=v_to_u,
                             neutral_tone_with_five=neutral_tone_with_five)

    finals = tone3_to_tone2(finals, v_to_u=v_to_u)

    return finals


def to_finals_tone3(pinyin, strict=True, v_to_u=False,
                    neutral_tone_with_five=False):
    """将 :py:attr:`~pypinyin.Style.TONE`、
    :py:attr:`~pypinyin.Style.TONE2` 或
    :py:attr:`~pypinyin.Style.TONE3` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.FINALS_TONE3` 风格的拼音

    :param pinyin: :py:attr:`~pypinyin.Style.TONE`、
                   :py:attr:`~pypinyin.Style.TONE2` 或
                   :py:attr:`~pypinyin.Style.TONE3` 风格的拼音
    :param strict: 返回结果是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :param neutral_tone_with_five: 是否使用 ``5`` 标识轻声
    :return: :py:attr:`~pypinyin.Style.FINALS_TONE3` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import to_finals_tone3
      >>> to_finals_tone3('zhōng')
      'ong1'

    """
    pinyin = pinyin.replace('5', '')
    finals = to_finals(pinyin, strict=strict, v_to_u=v_to_u)
    if not finals:
        return finals

    numbers = _re_number.findall(replace_symbol_to_number(pinyin))
    if not numbers:
        if neutral_tone_with_five:
            numbers = ['5']
        else:
            return finals

    number = numbers[0]
    finals = finals + number

    return finals


def tone_to_normal(tone, v_to_u=False):
    """将 :py:attr:`~pypinyin.Style.TONE` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音

    :param tone: :py:attr:`~pypinyin.Style.TONE` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :return: :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import tone_to_normal
      >>> tone_to_normal('zhōng')
      'zhong'
      >>> tone_to_normal('lüè')
      'lve'
      >>> tone_to_normal('lüè', v_to_u=True)
      'lüe'
    """
    s = tone_to_tone2(tone, v_to_u=v_to_u)
    s = _re_number.sub('', s)
    return _v_to_u(s, v_to_u)


def tone_to_tone2(tone, v_to_u=False, neutral_tone_with_five=False, **kwargs):
    """将 :py:attr:`~pypinyin.Style.TONE` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.TONE2` 风格的拼音

    :param tone: :py:attr:`~pypinyin.Style.TONE` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :param neutral_tone_with_five: 是否使用 ``5`` 标识轻声
    :param kwargs: 用于兼容老版本的 ``neutral_tone_with_5`` 参数，当传入
                   ``neutral_tone_with_5`` 参数时，
                   将覆盖 ``neutral_tone_with_five`` 的值。
    :return: :py:attr:`~pypinyin.Style.TONE2` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import tone_to_tone2
      >>> tone_to_tone2('zhōng')
      'zho1ng'
      >>> tone_to_tone2('shang')
      'shang'
      >>> tone_to_tone2('shang', neutral_tone_with_5=True)
      'sha5ng'
      >>> tone_to_tone2('lüè')
      'lve4'
      >>> tone_to_tone2('lüè', v_to_u=True)
      'lüe4'
    """
    if kwargs.get('neutral_tone_with_5', None) is not None:
        neutral_tone_with_five = kwargs['neutral_tone_with_5']
    tone3 = tone_to_tone3(
        tone, v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five)
    s = tone3_to_tone2(tone3)
    return _v_to_u(s, v_to_u)


def tone_to_tone3(tone, v_to_u=False, neutral_tone_with_five=False, **kwargs):
    """将 :py:attr:`~pypinyin.Style.TONE` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.TONE3` 风格的拼音

    :param tone: :py:attr:`~pypinyin.Style.TONE` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :param neutral_tone_with_five: 是否使用 ``5`` 标识轻声
    :param kwargs: 用于兼容老版本的 ``neutral_tone_with_5`` 参数，当传入
                   ``neutral_tone_with_5`` 参数时，
                   将覆盖 ``neutral_tone_with_five`` 的值。
    :return: :py:attr:`~pypinyin.Style.TONE3` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import tone_to_tone3
      >>> tone_to_tone3('zhōng')
      'zhong1'
      >>> tone_to_tone3('shang')
      'shang'
      >>> tone_to_tone3('shang', neutral_tone_with_five=True)
      'shang5'
      >>> tone_to_tone3('lüè')
      'lve4'
      >>> tone_to_tone3('lüè', v_to_u=True)
      'lüe4'
    """
    if kwargs.get('neutral_tone_with_5', None) is not None:
        neutral_tone_with_five = kwargs['neutral_tone_with_5']
    tone3 = converter.to_tone3(tone)
    s = _improve_tone3(tone3, neutral_tone_with_five=neutral_tone_with_five)
    return _v_to_u(s, v_to_u)


def tone2_to_normal(tone2, v_to_u=False):
    """将 :py:attr:`~pypinyin.Style.TONE2` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音

    :param tone2: :py:attr:`~pypinyin.Style.TONE2` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :return: Style.NORMAL 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import tone2_to_normal
      >>> tone2_to_normal('zho1ng')
      'zhong'
      >>> tone2_to_normal('lüe4')
      'lve'
      >>> tone2_to_normal('lüe4', v_to_u=True)
      'lüe'
    """
    s = _re_number.sub('', tone2)
    s = _v_to_u(s, v_to_u)
    return _fix_v_u(tone2, s, v_to_u=v_to_u)


def tone2_to_tone(tone2):
    """将 :py:attr:`~pypinyin.Style.TONE2` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.TONE` 风格的拼音

    :param tone2: :py:attr:`~pypinyin.Style.TONE2` 风格的拼音
    :return: Style.TONE 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import tone2_to_tone
      >>> tone2_to_tone('zho1ng')
      'zhōng'
    """
    regex = re.compile(RE_TONE2.pattern.replace('$', ''))
    d = phonetic_symbol.phonetic_symbol_reverse
    string = tone2.replace('ü', 'v').replace('5', '').replace('0', '')

    def _replace(m):
        s = m.group(0)
        return d.get(s) or s

    return regex.sub(_replace, string).replace('v', 'ü')


def tone2_to_tone3(tone2, v_to_u=False):
    """将 :py:attr:`~pypinyin.Style.TONE2` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.TONE3` 风格的拼音

    :param tone2: :py:attr:`~pypinyin.Style.TONE2` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :return: :py:attr:`~pypinyin.Style.TONE3` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import tone2_to_tone3
      >>> tone2_to_tone3('zho1ng')
      'zhong1'
      >>> tone2_to_tone3('lüe4')
      'lve4'
      >>> tone2_to_tone3('lüe4', v_to_u=True)
      'lüe4'
    """
    tone3 = RE_TONE3.sub(r'\1\3\2', tone2)
    return _fix_v_u(tone2, tone3, v_to_u=v_to_u)


def tone3_to_normal(tone3, v_to_u=False):
    """将 :py:attr:`~pypinyin.Style.TONE3` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音

    :param tone3: :py:attr:`~pypinyin.Style.TONE3` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :return: :py:attr:`~pypinyin.Style.NORMAL` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import tone3_to_normal
      >>> tone3_to_normal('zhong1')
      'zhong'
      >>> tone3_to_normal('lüe4')
      'lve'
      >>> tone3_to_normal('lüe4', v_to_u=True)
      'lüe'
    """
    s = _re_number.sub('', tone3)
    s = _v_to_u(s, v_to_u)
    return _fix_v_u(tone3, s, v_to_u=v_to_u)


def tone3_to_tone(tone3):
    """将 :py:attr:`~pypinyin.Style.TONE3` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.TONE` 风格的拼音

    :param tone3: :py:attr:`~pypinyin.Style.TONE3` 风格的拼音
    :return: :py:attr:`~pypinyin.Style.TONE` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import tone3_to_tone
      >>> tone3_to_tone('zhong1')
      'zhōng'
    """
    tone2 = tone3_to_tone2(tone3, v_to_u=True)
    return tone2_to_tone(tone2)


def tone3_to_tone2(tone3, v_to_u=False):
    """将 :py:attr:`~pypinyin.Style.TONE3` 风格的拼音转换为
    :py:attr:`~pypinyin.Style.TONE2` 风格的拼音

    :param tone3: :py:attr:`~pypinyin.Style.TONE3` 风格的拼音
    :param v_to_u: 是否使用 ``ü`` 代替原来的 ``v``，
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :return: :py:attr:`~pypinyin.Style.TONE2` 风格的拼音

    Usage::

      >>> from pypinyin.contrib.tone_convert import tone3_to_tone2
      >>> tone3_to_tone2('zhong1')
      'zho1ng'
      >>> tone3_to_tone2('lüe4')
      'lve4'
      >>> tone3_to_tone2('lüe4', v_to_u=True)
      'lüe4'
    """
    no_number_tone3 = tone3_to_normal(tone3)
    mark_index = right_mark_index(no_number_tone3)
    if mark_index is None:
        mark_index = len(no_number_tone3) - 1
    before = no_number_tone3[:mark_index + 1]
    after = no_number_tone3[mark_index + 1:]

    number = _get_number_from_pinyin(tone3)
    if number is None:
        return tone3

    s = '{}{}{}'.format(before, number, after)
    return _fix_v_u(tone3, s, v_to_u=v_to_u)


def _improve_tone3(tone3, neutral_tone_with_five=False):
    number = _get_number_from_pinyin(tone3)
    if number is None and neutral_tone_with_five and tone3 != '':
        tone3 = '{}5'.format(tone3)
    return tone3


def _get_number_from_pinyin(pinyin):
    numbers = _re_number.findall(pinyin)
    if numbers:
        number = numbers[0]
    else:
        number = None
    return number


def _v_to_u(pinyin, replace=False):
    if not replace:
        return pinyin
    return pinyin.replace('v', 'ü')


def _fix_v_u(origin_py, new_py, v_to_u):
    if not v_to_u:
        return new_py.replace('ü', 'v')

    return _v_to_u(new_py, replace=True)
