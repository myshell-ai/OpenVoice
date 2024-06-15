from __future__ import absolute_import, print_function, unicode_literals

import sys
from collections.abc import Callable

is_ironpython = "IronPython" in sys.version


def is_callable(x):
    return isinstance(x, Callable)


def execfile(fname, glob, loc=None):
    loc = loc if (loc is not None) else glob

    exec(
        compile(
            open(
                fname,
                'r',
                encoding='utf-8',
            ).read(),
            fname,
            'exec',
        ),
        glob,
        loc,
    )
