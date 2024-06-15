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

"""Count the number of warping paths consistent with the constraints."""

import numpy


def countPaths(d, debug=False):
    # IMPORT_RDOCSTRING countPaths
    """Count the number of warping paths consistent with the constraints.

Count how many possible warping paths exist in the alignment problem
passed as an argument. The object passed as an argument is used to look
up the problem parameters such as the used step pattern, windowing, open
ends, and so on. The actual alignment is ignored.

**Details**

Note that the number of paths grows exponentially with problems size.
The result may be approximate when windowing functions are used.

If ``debug=True``, a matrix used for the computation is returned instead
of the final result.

Parameters
----------
d : 
    an object of class `dtw`
debug : 
    return an intermediate result

Returns
-------

The number of paths.

Examples
--------

>>> from dtw import *
>>> ds = dtw( numpy.arange(3,10), numpy.arange(1,9),
...           keep_internals=True, step_pattern=asymmetric);
>>> countPaths(ds)
126

"""
    # ENDIMPORT
    N = d.N
    M = d.M
    m = numpy.full((N, M), numpy.nan)

    if d.openBegin:
        m[0, :] = 1.0
    else:
        m[0, 0] = 1.0

    dir = d.stepPattern
    npats = dir.get_n_patterns()
    # nsteps = dir.get_n_rows()
    deltas = dir._mkDirDeltas()

    wf = d.windowFunction

    for ii in range(N):
        for jj in range(M):
            if numpy.isfinite(m[ii, jj]):
                continue

            if not wf(ii, jj,
                      query_size=N,
                      reference_size=M,
                      **d.windowArgs):
                m[ii, jj] = 0
                continue

            np = 0
            for k in range(npats):
                ni = ii - deltas[k, 0]
                nj = jj - deltas[k, 1]

                if ni >= 0 and nj >= 0:
                    np += m[ni, nj]

            m[ii, jj] = np

    if debug:
        return m

    if d.openEnd:
        r = numpy.sum(m[-1,])
    else:
        r = m[-1, -1]
    return int(r)
