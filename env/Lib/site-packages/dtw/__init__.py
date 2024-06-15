# -*- coding: utf-8 -*-

"""Top-level package for the Comprehensive Dynamic Time Warp library.

Please see the help for the dtw.dtw() function which is the package's
main entry point.

"""

__author__ = """Toni Giorgino"""
__email__ = 'toni.giorgino@gmail.com'
__version__ = '1.4.4'

# There are no comments in this package because it mirrors closely the R sources.

# List of things to export on "from dtw import *"
from dtw.dtw import *
from dtw.stepPattern import *
from dtw.countPaths import *
from dtw.dtwPlot import *
from dtw.mvm import *
from dtw.warp import *
from dtw.warpArea import *
from dtw.window import *
from dtw import dtw_test_data

import sys

# Only print in interactive mode
# https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode/2356427#2356427
import __main__ as main
if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    print("""Importing the dtw module. When using in academic works please cite:
  T. Giorgino. Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package.
  J. Stat. Soft., doi:10.18637/jss.v031.i07.\n""")


