# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************
from __future__ import absolute_import, print_function, unicode_literals

import logging
import logging.handlers
import socket
import struct

from pyreadline3.unicode_helper import ensure_unicode

try:
    import msvcrt
except ImportError:
    msvcrt = None
    print("problem")


port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
host = 'localhost'


def check_key():
    if msvcrt is None:
        return False
    else:
        if msvcrt.kbhit():
            q = ensure_unicode(msvcrt.getch())
            return q
    return ""


singleline = False


def main():
    print("Starting TCP logserver on port:", port)
    print("Press q to quit logserver", port)
    print("Press c to clear screen", port)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    s.bind(("", port))
    s.settimeout(1)
    while True:
        try:
            data, addr = s.recvfrom(100000)
            print(data, end="")
        except socket.timeout:
            key = check_key().lower()
            if "q" == key:
                print("Quitting logserver")
                break
            elif "c" == key:
                print("\n" * 100)


if __name__ == "__main__":
    main()
