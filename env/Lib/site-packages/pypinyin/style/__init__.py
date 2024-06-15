# -*- coding: utf-8 -*-
from functools import wraps

# 存储各拼音风格对应的实现
_registry = {}


def convert(pinyin, style, strict, default=None, **kwargs):
    """根据拼音风格把原始拼音转换为不同的格式

    :param pinyin: 原始有声调的单个拼音
    :type pinyin: unicode
    :param style: 拼音风格
    :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                   是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :type strict: bool
    :param default: 拼音风格对应的实现不存在时返回的默认值
    :param kwargs: 兼容后续可能会新增的关键字参数。当前包含如下关键字参数:
                   ``han``: 当前拼音对应的原始汉字。

    :return: 按照拼音风格进行处理过后的拼音字符串
    :rtype: unicode
    """
    if style in _registry:
        return _registry[style](pinyin, strict=strict, **kwargs)
    return default


def register(style, func=None):
    """注册一个拼音风格实现。
    自定义的函数应当使用 ``**kwargs`` 来兼容后续可能会新增的关键字参数，
    当前默认会传递如下参数：

    * ``pinyin``: 原始有声调的单个拼音
    * ``strict``: 是否开启 strict 模式
    * ``han``: 当前拼音对应的原始汉字

    ::

        @register('echo')
        def echo(pinyin, **kwargs):
            return pinyin

        # or
        register('echo', echo)
    """
    if func is not None:
        _registry[style] = func
        return

    def decorator(func):
        _registry[style] = func

        @wraps(func)
        def wrapper(pinyin, **kwargs):
            return func(pinyin, **kwargs)

        return wrapper
    return decorator


def auto_discover():
    """自动注册内置的拼音风格实现"""
    from pypinyin.style import (  # noqa
        initials,
        tone,
        finals,
        bopomofo,
        cyrillic,
        wadegiles,
        others,
    )
