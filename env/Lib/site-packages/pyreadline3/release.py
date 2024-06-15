# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

"""Release data for the pyreadline3 project.

$Id$"""

# *****************************************************************************
#       Copyright (C) 2006-2020  Jorgen Stenarson. <jorgen.stenarson@kroywen.se>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

# Name of the package for release purposes.  This is the name which labels
# the tarballs and RPMs made by distutils, so it's best to lowercase it.
name = 'pyreadline3'

# For versions with substrings (like 0.6.16.svn), use an extra . to separate
# the new substring.  We have to avoid using either dashes or underscores,
# because bdist_rpm does not accept dashes (an RPM) convention, and
# bdist_deb does not accept underscores (a Debian convention).

branch = ''

version = '3.4.1'

description = "A python implementation of GNU readline."

long_description = \
    """
The `pyreadline3` package is based on the stale package `pyreadline` located
at https://github.com/pyreadline/pyreadline.
The original `pyreadline` package is a python implementation of GNU `readline`
functionality.
It is based on the `ctypes` based UNC `readline` package by Gary Bishop.
It is not complete.
It has been tested for use with Windows 10.

Version 3.4+ or pyreadline3 runs on Python 3.5+.

Features

- keyboard text selection and copy/paste
- Shift-arrowkeys for text selection
- Control-c can be used for copy activate with allow_ctrl_c(True) in config file
- Double tapping ctrl-c will raise a KeyboardInterrupt, use
  ctrl_c_tap_time_interval(x)
- where x is your preferred tap time window, default 0.3 s.
- paste pastes first line of content on clipboard.
- ipython_paste, pastes tab-separated data as list of lists or numpy array if
  all data is numeric
- paste_mulitline_code pastes multi line code, removing any empty lines.

The latest development version is always available at the project git
repository https://github.com/pyreadline3/pyreadline3
"""

license_name = 'BSD'

authors = {
    'Bassem': ('Bassem Girgis', 'brgirgis@gmail.com'),
    'Jorgen': ('Jorgen Stenarson', 'jorgen.stenarson@kroywen.se'),
    'Gary': ('Gary Bishop', ''),
    'Jack': ('Jack Trainor', ''),
}

url = 'https://pypi.python.org/pypi/pyreadline3/'
download_url = 'https://pypi.python.org/pypi/pyreadline3/'
platforms = [
    'Windows XP/2000/NT',
    'Windows 95/98/ME',
]

keywords = [
    'readline',
    'pyreadline',
    'pyreadline3',
]

classifiers = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 5 - Production/Stable',

    'Environment :: Console',

    'Operating System :: Microsoft :: Windows',

    'License :: OSI Approved :: BSD License',

    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]
