#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import sys

SUPPORT_UCS4 = len('\U00020000') == 1

PY2 = sys.version_info < (3, 0)

subversion = getattr(sys, 'subversion', [''])
# 这些 Python 实现虽然是 Python 2 但字符串的行为跟 Python 3 是一样的
if subversion[0] in (
    'IronPython',
):
    PY2 = False  # pragma: no cover

if not PY2:
    text_type = str
    bytes_type = bytes
else:
    text_type = unicode  # noqa
    bytes_type = str

try:
    callable_check = callable  # noqa
except NameError:
    def callable_check(obj):
        return hasattr(obj, '__call__')
