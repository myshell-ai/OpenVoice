#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from argparse import ArgumentParser
import logging
import sys

import pypinyin
from pypinyin.compat import PY2

style_map = {
    'NORMAL': pypinyin.Style.NORMAL,
    'zhao': pypinyin.Style.NORMAL,
    'TONE': pypinyin.Style.TONE,
    'zh4ao': pypinyin.Style.TONE,
    'TONE2': pypinyin.Style.TONE2,
    'zha4o': pypinyin.Style.TONE2,
    'TONE3': pypinyin.Style.TONE3,
    'zhao4': pypinyin.Style.TONE3,
    'INITIALS': pypinyin.Style.INITIALS,
    'zh': pypinyin.Style.INITIALS,
    'FIRST_LETTER': pypinyin.Style.FIRST_LETTER,
    'z': pypinyin.Style.FIRST_LETTER,
    'FINALS': pypinyin.Style.FINALS,
    'ao': pypinyin.Style.FINALS,
    'FINALS_TONE': pypinyin.Style.FINALS_TONE,
    '4ao': pypinyin.Style.FINALS_TONE,
    'FINALS_TONE2': pypinyin.Style.FINALS_TONE2,
    'a4o': pypinyin.Style.FINALS_TONE2,
    'FINALS_TONE3': pypinyin.Style.FINALS_TONE3,
    'ao4': pypinyin.Style.FINALS_TONE3,
    'BOPOMOFO': pypinyin.Style.BOPOMOFO,
    'BOPOMOFO_FIRST': pypinyin.Style.BOPOMOFO_FIRST,
    'CYRILLIC': pypinyin.Style.CYRILLIC,
    'CYRILLIC_FIRST': pypinyin.Style.CYRILLIC_FIRST,
}
func_map = {
    'pinyin': pypinyin.pinyin,
    'slug': pypinyin.slug,
}
default_style = 'zh4ao'


class NullWriter(object):
    """数据流黑洞，类似 linux/unix 下 /dev/null 的效果。"""
    def write(self, string):
        pass


def get_parser():
    parser = ArgumentParser(description='convert chinese to pinyin.')
    parser.add_argument('-V', '--version', action='version',
                        version='{0} {1}'.format(
                            pypinyin.__title__, pypinyin.__version__
                        ))
    # 要执行的函数名称
    parser.add_argument('-f', '--func',
                        help='function name (default: "pinyin")',
                        choices=['pinyin', 'slug'],
                        default='pinyin')
    # 拼音风格
    parser.add_argument(
        '-s', '--style',
        help='pinyin style (default: "{0}")'.format(default_style),
        choices=style_map.keys(), default=default_style
    )
    parser.add_argument('-p', '--separator',
                        help='slug separator (default: "-")',
                        default='-')
    parser.add_argument('-e', '--errors',
                        help=('how to handle none-pinyin string'
                              ' (default: "default")'),
                        choices=['default', 'ignore', 'replace'],
                        default='default')
    # 输出多音字
    parser.add_argument('-m', '--heteronym', help='enable heteronym',
                        action='store_true')
    # 要查询的汉字
    parser.add_argument('hans', help='chinese string')
    return parser


def main():
    # 禁用除 CRITICAL 外的日志消息
    logging.disable(logging.CRITICAL)

    # read hans from stdin
    if not sys.stdin.isatty():
        pipe_data = sys.stdin.read().strip()
    else:
        pipe_data = ''
    args = sys.argv[1:]
    if pipe_data:
        args.append(pipe_data)

    # 获取命令行选项和参数
    parser = get_parser()
    options = parser.parse_args(args)
    if PY2:
        hans = options.hans.decode(sys.stdin.encoding or 'utf-8')
    else:
        hans = options.hans
    func = getattr(pypinyin, options.func)
    style = style_map[options.style]
    heteronym = options.heteronym
    separator = options.separator
    errors = options.errors

    func_kwargs = {
        'pinyin': {'heteronym': heteronym, 'errors': errors},
        'slug': {'heteronym': heteronym, 'separator': separator,
                 'errors': errors},
    }
    if PY2:
        kwargs = func_kwargs[func.func_name]
    else:
        kwargs = func_kwargs[func.__name__]

    # 重设标准输出流和标准错误流
    # 不输出任何字符，防止污染命令行命令的输出结果
    # 其实主要是为了干掉 jieba 内的 print 语句 ;)
    sys.stdout = sys.stderr = NullWriter()
    result = func(hans, style=style, **kwargs)
    # 恢复默认
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    if not result:
        print('')
    elif result and isinstance(result, (list, tuple)):
        if isinstance(result[0], (list, tuple)):
            print(' '.join([','.join(s) for s in result]))
        else:
            print(result)
    else:
        print(result)


if __name__ == '__main__':
    main()
