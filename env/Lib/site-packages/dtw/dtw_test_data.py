##
## Copyright (c) 2006-2019 of Toni Giorgino
##
## This file is part of the DTW package.
##
## DTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## DTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with DTW.  If not, see <http://www.gnu.org/licenses/>.
##

"""Miscellaneous test data for DTW"""


import numpy
from pkgutil import get_data

def aami():
    # IMPORT_RDOCSTRING aami
    """ANSI/AAMI EC13 Test Waveforms, 3a and 3b

ANSI/AAMI EC13 Test Waveforms 3a and 3b, as obtained from the PhysioBank
database.

**Details**

The following text is reproduced (abridged) from PhysioBank, page
https://www_physionet_org/content/aami-ec13/1.0.0/. Other recordings
belong to the dataset and can be obtained from the same page.

The files in this set can be used for testing a variety of devices that
monitor the electrocardiogram. The recordings include both synthetic and
real waveforms. For details on these test waveforms and how to use them,
please refer to section 5.1.2.1, paragraphs (e) and (g) in the reference
below. Each recording contains one ECG signal sampled at 720 Hz with
12-bit resolution.

Notes
-----

Timestamps in the datasets have been re-created at the indicated
frequency of 720 Hz, whereas the original timestamps in ms (at least in
text format) only had three decimal digits’ precision, and were
therefore affected by substantial jittering.

References
----------

-  Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark
   RG, Mietus JE, Moody GB, Peng CK, Stanley HE. *PhysioBank,
   PhysioToolkit, and PhysioNet: Components of a New Research Resource
   for Complex Physiologic Signals.* Circulation 101(23):e215-e220; 2000
   (June 13).
-  Cardiac monitors, heart rate meters, and alarms; American National
   Standard (ANSI/AAMI EC13:2002). Arlington, VA: Association for the
   Advancement of Medical Instrumentation, 2002.

Examples
--------
>>> from dtw import *; import numpy as np
>>> (aami3a, aami3b) = dtw_test_data.aami()

Timestamps (ms) are in the first row, values (mV) in the second.

>>> with np.printoptions(precision=3): print(aami3a[0,:])   # doctest: +NORMALIZE_WHITESPACE
[0.000e+00 1.389e+00 2.778e+00 ... 5.983e+04 5.983e+04 5.983e+04]

>>> with np.printoptions(precision=3): print(aami3a[1,:])   # doctest: +NORMALIZE_WHITESPACE
[0.185 0.185 0.169 ... 0.208 0.208 0.208]

"""
    # ENDIMPORT

    ts = lambda v: numpy.arange(len(v))/720.*1000.
    a3a = numpy.fromstring(get_data(__name__, 'data/aami3a.csv'), sep="\n")
    a3b = numpy.fromstring(get_data(__name__, 'data/aami3b.csv'), sep="\n")

    aami3a = numpy.vstack([ts(a3a),a3a])
    aami3b = numpy.vstack([ts(a3b),a3b])
    
    return (aami3a, aami3b)



def sin_cos():
    """Noisy sine vs cosine demo data.

Returns a tuple (query, reference) used in various examples, defined
as follows:

    _idx = numpy.linspace(0,6.28,num=100)
    query = numpy.sin(_idx) + numpy.random.uniform(0,0.1,len(_idx)),
    reference = numpy.cos(_idx)

"""
    _idx = numpy.linspace(0,6.28,num=100)
    return (
        numpy.sin(_idx) + numpy.random.uniform(0,0.1,len(_idx)),
        numpy.cos(_idx)
    )
