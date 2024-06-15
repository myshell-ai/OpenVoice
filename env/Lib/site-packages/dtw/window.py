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


# IMPORT_RDOCSTRING dtwWindowingFunctions
"""Global constraints and windowing functions for DTW

Various global constraints (windows) which can be applied to the
``window_type`` argument of [dtw()], including the Sakoe-Chiba band, the
Itakura parallelogram, and custom functions.

**Details**

Windowing functions can be passed to the ``window_type`` argument in
[dtw()] to put a global constraint to the warping paths allowed. They
take two integer arguments (plus optional parameters) and must return a
boolean value ``True`` if the coordinates fall within the allowed region
for warping paths, ``False`` otherwise.

User-defined functions can read variables ``reference_size``,
``query_size`` and ``window_size``; these are pre-set upon invocation.
Some functions require additional parameters which must be set (e_g.
``window_size``). User-defined functions are free to implement any
window shape, as long as at least one path is allowed between the
initial and final alignment points, i_e., they are compatible with the
DTW constraints.

The ``sakoeChibaWindow`` function implements the Sakoe-Chiba band, i_e.
``window_size`` elements around the ``main`` diagonal. If the window
size is too small, i_e. if ``reference_size``-``query_size`` >
``window_size``, warping becomes impossible.

An ``itakuraWindow`` global constraint is still provided with this
package. See example below for a demonstration of the difference between
a local the two.

The ``slantedBandWindow`` (package-specific) is a band centered around
the (jagged) line segment which joins element ``[1,1]`` to element
``[query_size,reference_size]``, and will be ``window_size`` columns
wide. In other words, the “diagonal” goes from one corner to the other
of the possibly rectangular cost matrix, therefore having a slope of
``M/N``, not 1.

``dtwWindow_plot`` visualizes a windowing function. By default it plots
a 200 x 220 rectangular region, which can be changed via
``reference_size`` and ``query_size`` arguments.

Parameters
----------
iw : 
    index in the query (row) -- automatically set
jw : 
    index in the reference (column) -- automatically set
query_size : 
    size of the query time series -- automatically set
reference_size : 
    size of the reference time series -- automatically set
window_size : 
    window size, used by some windowing functions -- must be set
fun : 
    a windowing function
... : 
    additional arguments passed to windowing functions

Returns
-------

Windowing functions return ``True`` if the coordinates passed as
arguments fall within the chosen warping window, ``False`` otherwise.
User-defined functions should do the same.

Notes
-----

Although ``dtwWindow_plot`` resembles object-oriented notation, there is
not a such a dtwWindow class currently.

A widely held misconception is that the “Itakura parallelogram” (as
described in reference 2) is a *global* constraint, i_e. a window. To
the author’s knowledge, it instead arises from the local slope
restrictions imposed to the warping path, such as the one implemented by
the [typeIIIc()] step pattern.

References
----------

1. Sakoe, H.; Chiba, S., *Dynamic programming algorithm optimization for
   spoken word recognition,* Acoustics, Speech, and Signal Processing,
   IEEE Transactions on , vol.26, no.1, pp. 43-49, Feb 1978
   `doi:10.1109/TASSP.1978.1163055 <https://doi.org/10.1109/TASSP.1978.1163055>`__
2. Itakura, F., *Minimum prediction residual principle applied to speech
   recognition,* Acoustics, Speech, and Signal Processing, IEEE
   Transactions on , vol.23, no.1, pp. 67-72, Feb 1975.
   `doi:10.1109/TASSP.1975.1162641 <https://doi.org/10.1109/TASSP.1975.1162641>`__

Examples
--------
>>> from dtw import *
>>> import numpy as np

Default test data

>>> (query, reference) = dtw_test_data.sin_cos()

Asymmetric step with Sakoe-Chiba band

>>> asyband = dtw(query,reference,
...     keep_internals=True, step_pattern=asymmetric,
...     window_type=sakoeChibaWindow,
...     window_args={'window_size': 30}                  );

>>> dtwPlot(asyband,type="density")  # doctest: +SKIP

Display some windowing functions 

>>> #TODO dtwWindow_plot(itakuraWindow, main="So-called Itakura parallelogram window")
>>> #TODO dtwWindow_plot(slantedBandWindow, window_size=2,
>>> #TODO reference=13, query=17, main="The slantedBandWindow at window_size=2")

"""
# ENDIMPORT



# The functions must be vectorized! The first 2 args are matrices of row and column indices.

def noWindow(iw, jw, query_size, reference_size):
    return (iw | True)


def sakoeChibaWindow(iw, jw, query_size, reference_size, window_size):
    ok = abs(jw - iw) <= window_size
    return ok


def itakuraWindow(iw, jw, query_size, reference_size):
    n = query_size
    m = reference_size
    ok = (jw < 2 * iw) & \
         (iw <= 2 * jw) & \
         (iw >= n - 1 - 2 * (m - jw)) & \
         (jw > m - 1 - 2 * (n - iw))
    return ok


def slantedBandWindow(iw, jw, query_size, reference_size, window_size):
    diagj = (iw * reference_size / query_size)
    return abs(jw - diagj) <= window_size;
