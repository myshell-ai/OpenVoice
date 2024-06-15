# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Jack Trainor.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************
###################################
#
# Based on recipe posted to ctypes-users
# see archive
# http://aspn.activestate.com/ASPN/Mail/Message/ctypes-users/1771866
#
#

##########################################################################
#
# The Python win32clipboard lib functions work well enough ... except that they
# can only cut and paste items from within one application, not across
# applications or processes.
#
# I've written a number of Python text filters I like to run on the contents of
# the clipboard so I need to call the Windows clipboard API with global memory
# for my filters to work properly.
#
# Here's some sample code solving this problem using ctypes.
#
# This is my first work with ctypes.  It's powerful stuff, but passing arguments
# in and out of functions is tricky.  More sample code would have been helpful,
# hence this contribution.
#
##########################################################################
from __future__ import absolute_import, print_function, unicode_literals

import ctypes
import ctypes.wintypes as wintypes
from ctypes import *

from pyreadline3.keysyms.winconstants import CF_UNICODETEXT, GHND
from pyreadline3.unicode_helper import ensure_str, ensure_unicode

OpenClipboard = windll.user32.OpenClipboard
OpenClipboard.argtypes = [wintypes.HWND]
OpenClipboard.restype = wintypes.BOOL

EmptyClipboard = windll.user32.EmptyClipboard

GetClipboardData = windll.user32.GetClipboardData
GetClipboardData.argtypes = [wintypes.UINT]
GetClipboardData.restype = wintypes.HANDLE

GetClipboardFormatName = windll.user32.GetClipboardFormatNameA
GetClipboardFormatName.argtypes = [wintypes.UINT, c_char_p, c_int]

SetClipboardData = windll.user32.SetClipboardData
SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
SetClipboardData.restype = wintypes.HANDLE

EnumClipboardFormats = windll.user32.EnumClipboardFormats
EnumClipboardFormats.argtypes = [c_int]

CloseClipboard = windll.user32.CloseClipboard
CloseClipboard.argtypes = []


GlobalAlloc = windll.kernel32.GlobalAlloc
GlobalAlloc.argtypes = [wintypes.UINT, c_size_t]
GlobalAlloc.restype = wintypes.HGLOBAL
GlobalLock = windll.kernel32.GlobalLock
GlobalLock.argtypes = [wintypes.HGLOBAL]
GlobalLock.restype = c_void_p
GlobalUnlock = windll.kernel32.GlobalUnlock
GlobalUnlock.argtypes = [c_int]

_strncpy = ctypes.windll.kernel32.lstrcpynW
_strncpy.restype = c_wchar_p
_strncpy.argtypes = [c_wchar_p, c_wchar_p, c_size_t]


def enum():
    OpenClipboard(0)
    q = EnumClipboardFormats(0)
    while q:
        q = EnumClipboardFormats(q)
    CloseClipboard()


def getformatname(format):
    buffer = c_buffer(" " * 100)
    bufferSize = sizeof(buffer)
    OpenClipboard(0)
    GetClipboardFormatName(format, buffer, bufferSize)
    CloseClipboard()
    return buffer.value


def GetClipboardText():
    text = ""
    if OpenClipboard(0):
        hClipMem = GetClipboardData(CF_UNICODETEXT)
        if hClipMem:
            text = wstring_at(GlobalLock(hClipMem))
            GlobalUnlock(hClipMem)
        CloseClipboard()
    return text


def SetClipboardText(text):
    buffer = create_unicode_buffer(ensure_unicode(text))
    bufferSize = sizeof(buffer)
    hGlobalMem = GlobalAlloc(GHND, c_size_t(bufferSize))
    GlobalLock.restype = c_void_p
    lpGlobalMem = GlobalLock(hGlobalMem)
    _strncpy(cast(lpGlobalMem, c_wchar_p),
             cast(addressof(buffer), c_wchar_p),
             c_size_t(bufferSize))
    GlobalUnlock(c_int(hGlobalMem))
    if OpenClipboard(0):
        EmptyClipboard()
        SetClipboardData(CF_UNICODETEXT, hGlobalMem)
        CloseClipboard()


if __name__ == '__main__':
    txt = GetClipboardText()                            # display last text clipped
    print(txt)
