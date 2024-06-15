# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006  Michael Graz. <mgraz@plan10.com>
#       Copyright (C) 2006  Michael Graz. <mgraz@plan10.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************
from __future__ import absolute_import, print_function, unicode_literals

import sys
import unittest

import pyreadline3.logger as logger
from pyreadline3 import keysyms
from pyreadline3.lineeditor import lineobj
from pyreadline3.logger import log
from pyreadline3.modes.emacs import *
from pyreadline3.test.common import *

sys.path.insert(0, '../..')

logger.sock_silent = True
logger.show_event = ["debug"]

# ----------------------------------------------------------------------


class EmacsModeTest (EmacsMode):
    tested_commands = {}

    def __init__(self):
        EmacsMode.__init__(self, MockReadline())
        self.mock_console = MockConsole()
        self.init_editing_mode(None)
        self.lst_completions = []
        self.completer = self.mock_completer
        self.completer_delims = ' '
        self.tabstop = 4
        self.mark_directories = False
        self.show_all_if_ambiguous = False

    def get_mock_console(self):
        return self.mock_console
    console = property(get_mock_console)

    def _set_line(self, text):
        self.l_buffer.set_line(text)

    def get_line(self):
        return self.l_buffer.get_line_text()
    line = property(get_line)

    def get_line_cursor(self):
        return self.l_buffer.point
    line_cursor = property(get_line_cursor)

    def input(self, keytext):
        if keytext[0:1] == '"' and keytext[-1:] == '"':
            lst_key = ['"%s"' % c for c in keytext[1:-1]]
        else:
            lst_key = [keytext]
        for key in lst_key:
            keyinfo, event = keytext_to_keyinfo_and_event(key)
            dispatch_func = self.key_dispatch.get(
                keyinfo.tuple(), self.self_insert)
            self.tested_commands[dispatch_func.__name__] = dispatch_func
            log("keydisp: %s %s" % (key, dispatch_func.__name__))
            dispatch_func(event)
            self.previous_func = dispatch_func

    def accept_line(self, e):
        if EmacsMode.accept_line(self, e):
            # simulate return
            # self.add_history (self.line)
            self.l_buffer.reset_line()

    def mock_completer(self, text, state):
        return self.lst_completions[state]

# ----------------------------------------------------------------------


class TestsKeyinfo (unittest.TestCase):

    def test_keyinfo(self):
        keyinfo, event = keytext_to_keyinfo_and_event('"d"')
        self.assertEqual('d', event.char)
        keyinfo, event = keytext_to_keyinfo_and_event('"D"')
        self.assertEqual('D', event.char)
        keyinfo, event = keytext_to_keyinfo_and_event('"$"')
        self.assertEqual('$', event.char)
        keyinfo, event = keytext_to_keyinfo_and_event('Escape')
        self.assertEqual('\x1b', event.char)


class TestsMovement (unittest.TestCase):
    def test_cursor(self):
        r = EmacsModeTest()
        self.assertEqual(r.line, '')
        r.input('"First Second Third"')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 18)
        r.input('Control-a')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 0)
        r.input('Control-e')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 18)
        r.input('Home')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 0)
        r.input('Right')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 1)
        r.input('Ctrl-f')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 2)
        r.input('Ctrl-Right')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 5)
        r.input('Ctrl-Right')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 12)
        r.input('Ctrl-Right')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 18)
        r.input('Ctrl-Right')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 18)
        r.input('Ctrl-Left')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 13)
        r.input('Ctrl-Left')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 6)
        r.input('Ctrl-Left')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 0)
        r.input('Ctrl-Left')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 0)


class TestsDelete (unittest.TestCase):
    def test_delete(self):
        r = EmacsModeTest()
        self.assertEqual(r.line, '')
        r.input('"First Second Third"')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 18)
        r.input('Delete')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 18)
        r.input('Left')
        r.input('Left')
        r.input('Delete')
        self.assertEqual(r.line, 'First Second Thid')
        self.assertEqual(r.line_cursor, 16)
        r.input('Delete')
        self.assertEqual(r.line, 'First Second Thi')
        self.assertEqual(r.line_cursor, 16)
        r.input('Backspace')
        self.assertEqual(r.line, 'First Second Th')
        self.assertEqual(r.line_cursor, 15)
        r.input('Home')
        r.input('Right')
        r.input('Right')
        self.assertEqual(r.line, 'First Second Th')
        self.assertEqual(r.line_cursor, 2)
        r.input('Backspace')
        self.assertEqual(r.line, 'Frst Second Th')
        self.assertEqual(r.line_cursor, 1)
        r.input('Backspace')
        self.assertEqual(r.line, 'rst Second Th')
        self.assertEqual(r.line_cursor, 0)
        r.input('Backspace')
        self.assertEqual(r.line, 'rst Second Th')
        self.assertEqual(r.line_cursor, 0)
        r.input('Escape')
        self.assertEqual(r.line, '')
        self.assertEqual(r.line_cursor, 0)

    def test_delete_word(self):
        r = EmacsModeTest()
        self.assertEqual(r.line, '')
        r.input('"First Second Third"')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 18)
        r.input('Control-Backspace')
        self.assertEqual(r.line, 'First Second ')
        self.assertEqual(r.line_cursor, 13)
        r.input('Backspace')
        r.input('Left')
        r.input('Left')
        self.assertEqual(r.line, 'First Second')
        self.assertEqual(r.line_cursor, 10)
        r.input('Control-Backspace')
        self.assertEqual(r.line, 'First nd')
        self.assertEqual(r.line_cursor, 6)
        r.input('Escape')
        self.assertEqual(r.line, '')
        self.assertEqual(r.line_cursor, 0)
        r.input('"First Second Third"')
        r.input('Home')
        r.input('Right')
        r.input('Right')
        r.input('Control-Delete')
        self.assertEqual(r.line, 'FiSecond Third')
        self.assertEqual(r.line_cursor, 2)
        r.input('Control-Delete')
        self.assertEqual(r.line, 'FiThird')
        self.assertEqual(r.line_cursor, 2)
        r.input('Control-Delete')
        self.assertEqual(r.line, 'Fi')
        self.assertEqual(r.line_cursor, 2)
        r.input('Control-Delete')
        self.assertEqual(r.line, 'Fi')
        self.assertEqual(r.line_cursor, 2)
        r.input('Escape')
        self.assertEqual(r.line, '')
        self.assertEqual(r.line_cursor, 0)


class TestsSelectionMovement (unittest.TestCase):
    def test_cursor(self):
        r = EmacsModeTest()
        self.assertEqual(r.line, '')
        r.input('"First Second Third"')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 18)
        self.assertEqual(r.l_buffer.selection_mark, -1)
        r.input('Home')
        r.input('Shift-Right')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 1)
        self.assertEqual(r.l_buffer.selection_mark, 0)
        r.input('Shift-Control-Right')
        self.assertEqual(r.line, 'First Second Third')
        self.assertEqual(r.line_cursor, 5)
        self.assertEqual(r.l_buffer.selection_mark, 0)
        r.input('"a"')
        self.assertEqual(r.line, 'a Second Third')
        self.assertEqual(r.line_cursor, 1)
        self.assertEqual(r.l_buffer.selection_mark, -1)
        r.input('Shift-End')
        self.assertEqual(r.line, 'a Second Third')
        self.assertEqual(r.line_cursor, 14)
        self.assertEqual(r.l_buffer.selection_mark, 1)
        r.input('Delete')
        self.assertEqual(r.line, 'a')
        self.assertEqual(r.line_cursor, 1)
        self.assertEqual(r.l_buffer.selection_mark, -1)


class TestsHistory (unittest.TestCase):
    def test_history_1(self):
        r = EmacsModeTest()
        r.add_history('aa')
        r.add_history('bbb')
        self.assertEqual(r.line, '')
        r.input('Up')
        self.assertEqual(r.line, 'bbb')
        self.assertEqual(r.line_cursor, 3)
        r.input('Up')
        self.assertEqual(r.line, 'aa')
        self.assertEqual(r.line_cursor, 2)
        r.input('Up')
        self.assertEqual(r.line, 'aa')
        self.assertEqual(r.line_cursor, 2)
        r.input('Down')
        self.assertEqual(r.line, 'bbb')
        self.assertEqual(r.line_cursor, 3)
        r.input('Down')
        self.assertEqual(r.line, '')
        self.assertEqual(r.line_cursor, 0)

    def test_history_2(self):
        r = EmacsModeTest()
        r.add_history('aaaa')
        r.add_history('aaba')
        r.add_history('aaca')
        r.add_history('akca')
        r.add_history('bbb')
        r.add_history('ako')
        self.assert_line(r, '', 0)
        r.input('"a"')
        r.input('Up')
        self.assert_line(r, 'ako', 1)
        r.input('Up')
        self.assert_line(r, 'akca', 1)
        r.input('Up')
        self.assert_line(r, 'aaca', 1)
        r.input('Up')
        self.assert_line(r, 'aaba', 1)
        r.input('Up')
        self.assert_line(r, 'aaaa', 1)
        r.input('Right')
        self.assert_line(r, 'aaaa', 2)
        r.input('Down')
        self.assert_line(r, 'aaba', 2)
        r.input('Down')
        self.assert_line(r, 'aaca', 2)
        r.input('Down')
        self.assert_line(r, 'aaca', 2)
        r.input('Left')
        r.input('Left')
        r.input('Down')
        r.input('Down')
        self.assert_line(r, 'bbb', 3)
        r.input('Left')
        self.assert_line(r, 'bbb', 2)
        r.input('Down')
        self.assert_line(r, 'bbb', 2)
        r.input('Up')
        self.assert_line(r, 'bbb', 2)

    def test_history_3(self):
        r = EmacsModeTest()
        r.add_history('aaaa')
        r.add_history('aaba')
        r.add_history('aaca')
        r.add_history('akca')
        r.add_history('bbb')
        r.add_history('ako')
        self.assert_line(r, '', 0)
        r.input('')
        r.input('Up')
        self.assert_line(r, 'ako', 3)
        r.input('Down')
        self.assert_line(r, '', 0)
        r.input('Up')
        self.assert_line(r, 'ako', 3)

    def test_history_3(self):
        r = EmacsModeTest()
        r.add_history('aaaa')
        r.add_history('aaba')
        r.add_history('aaca')
        r.add_history('akca')
        r.add_history('bbb')
        r.add_history('ako')
        self.assert_line(r, '', 0)
        r.input('k')
        r.input('Up')
        self.assert_line(r, 'k', 1)

    def test_complete(self):
        import rlcompleter
        logger.sock_silent = False

        log("-" * 50)
        r = EmacsModeTest()
        completerobj = rlcompleter.Completer()

        def _nop(val, word):
            return word
        completerobj._callable_postfix = _nop
        r.completer = completerobj.complete
        r._bind_key("tab", r.complete)
        r.input('"exi(ksdjksjd)"')
        r.input('Control-a')
        r.input('Right')
        r.input('Right')
        r.input('Right')
        r.input('Tab')
        self.assert_line(r, "exit(ksdjksjd)", 4)

        r.input('Escape')
        r.input('"exi"')
        r.input('Control-a')
        r.input('Right')
        r.input('Right')
        r.input('Right')
        r.input('Tab')
        self.assert_line(r, "exit", 4)

    def assert_line(self, r, line, cursor):
        self.assertEqual(r.line, line)
        self.assertEqual(r.line_cursor, cursor)

# ----------------------------------------------------------------------
# utility functions

# ----------------------------------------------------------------------


if __name__ == '__main__':
    Tester()
    tested = sorted(EmacsModeTest.tested_commands.keys())
#    print(" Tested functions ".center(60,"-"))
#    print( "\n".join(tested))
#    print()

    all_funcs = dict([(x.__name__, x)
                      for x in list(EmacsModeTest().key_dispatch.values())])
    all_funcs = list(all_funcs.keys())
    not_tested = sorted([x for x in all_funcs if x not in tested])
    print(" Not tested functions ".center(60, "-"))
    print("\n".join(not_tested))
