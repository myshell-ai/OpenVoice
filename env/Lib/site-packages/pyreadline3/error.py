# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006-2020  Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************
from __future__ import absolute_import, print_function, unicode_literals


class ReadlineError(Exception):
    pass


class GetSetError(ReadlineError):
    pass
