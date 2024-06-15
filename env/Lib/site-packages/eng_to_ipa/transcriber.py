# 2/19 - not yet implemented

import re
from os.path import join, abspath, dirname
import eng_to_ipa.stress
import sqlite3
from collections import defaultdict


class Transcriber:

    def __init__(self, mode="sql", stress="both"):
        self._mode = mode
        self.stress = stress
        self.c = None  # potential SQL cursor

    @property
    def _mode(self):
        return self.mode

    @_mode.setter
    def _mode(self, value):
        if value.lower() == "sql":
            conn = sqlite3.connect(join(abspath(dirname(__file__)), "./resources/CMU_dict.db"))
            self.c = conn.cursor()
        self.mode = ""
