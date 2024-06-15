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

"""Warp one timeseries into the other"""

import numpy
import scipy.interpolate 


# Ties in x are removed and their y mean is used
def _solveTies(x,y):
    n = numpy.bincount(x)
    s = numpy.bincount(x,y)
    return numpy.arange(len(n)), s/n

# Should mimic R's stats::approx as closely as possible
def _interp(x, y):
    xt, yt = _solveTies(x,y)
    return scipy.interpolate.interp1d(xt, yt)
    

def warp(d, index_reference=False):
    # IMPORT_RDOCSTRING warp
    """Apply a warping to a given timeseries

Returns the indexing required to apply the optimal warping curve to a
given timeseries (warps either into a query or into a reference).

**Details**

The warping is returned as a set of indices, which can be used to
subscript the timeseries to be warped (or rows in a matrix, if one wants
to warp a multivariate time series). In other words, ``warp`` converts
the warping curve, or its inverse, into a function in the explicit form.

Multiple indices that would be mapped to a single point are averaged,
with a warning. Gaps in the index sequence are filled by linear
interpolation.

Parameters
----------
d : 
    `dtw` object specifying the warping curve to apply
index_reference : 
    `True` to warp a reference, `False` to warp a query

Returns
-------

A list of indices to subscript the timeseries.

Examples
--------
>>> from dtw import *
>>> import numpy as np

Default test data

>>> (query, reference) = dtw_test_data.sin_cos()

>>> alignment = dtw(query,reference);

>>> wq = warp(alignment,index_reference=False)
>>> wt = warp(alignment,index_reference=True)

>>> import matplotlib.pyplot as plt;		# doctest: +SKIP
... plt.plot(reference);
... plt.plot(query[wq]); 
... plt.gca().set_title("Warping query")

>>> import matplotlib.pyplot as plt;		# doctest: +SKIP
... plt.plot(query);
... plt.plot(reference[wt]); 
... plt.gca().set_title("Warping reference")

Asymmetric step makes it "natural" to warp
the reference, because every query index has
exactly one image (q->t is a function)

>>> alignment = dtw(query,reference,step_pattern=asymmetric)
>>> wt = warp(alignment,index_reference=True);

>>> plt.plot(query, "b-")               	# doctest: +SKIP
... plt.plot(reference[wt], "ok", facecolors='none')

"""
    # ENDIMPORT
    if not index_reference:
        iset = d.index1
        jset = d.index2
    else:
        iset = d.index2
        jset = d.index1

    jmax = numpy.max(jset)

    # Scipy interp1d is buggy. it does not deal with leading
    # duplicated values of x. It returns different values
    # depending on the dtypes of arguments.
    ifun = _interp(x=jset, y=iset)
    ii = ifun(numpy.arange(jmax+1))

    # Quick fix for bug
    if numpy.isnan(ii[0]):
        ii[numpy.isnan(ii)] = iset[0]
    return ii.astype(int)
