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

from pyreadline3.unicode_helper import ensure_str

host = "localhost"
port = logging.handlers.DEFAULT_TCP_LOGGING_PORT


pyreadline_logger = logging.getLogger('PYREADLINE')
pyreadline_logger.setLevel(logging.DEBUG)
pyreadline_logger.propagate = False
formatter = logging.Formatter(str('%(message)s'))
file_handler = None


class NULLHandler(logging.Handler):
    def emit(self, s):
        pass


class SocketStream(object):
    def __init__(self, host, port):
        self.logsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def write(self, s):
        self.logsocket.sendto(ensure_str(s), (host, port))

    def flush(self):
        pass

    def close(self):
        pass


socket_handler = None
pyreadline_logger.addHandler(NULLHandler())


def start_socket_log():
    global socket_handler
    socket_handler = logging.StreamHandler(SocketStream(host, port))
    socket_handler.setFormatter(formatter)
    pyreadline_logger.addHandler(socket_handler)


def stop_socket_log():
    global socket_handler
    if socket_handler:
        pyreadline_logger.removeHandler(socket_handler)
        socket_handler = None


def start_file_log(filename):
    global file_handler
    file_handler = logging.FileHandler(filename, "w")
    pyreadline_logger.addHandler(file_handler)


def stop_file_log():
    global file_handler
    if file_handler:
        pyreadline_logger.removeHandler(file_handler)
        file_handler.close()
        file_handler = None


def stop_logging():
    log("STOPING LOG")
    stop_file_log()
    stop_socket_log()


def log(s):
    s = ensure_str(s)
    pyreadline_logger.debug(s)
