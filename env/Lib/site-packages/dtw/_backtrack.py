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


import numpy

_INT_MIN = numpy.iinfo(numpy.int32).min

# This is O(n). Let's not make it unreadable.
def _backtrack(gcm):
    n = gcm.costMatrix.shape[0]
    m = gcm.costMatrix.shape[1]
    i = n - 1
    j = gcm.jmin

    # Drop null deltas
    dir = gcm.stepPattern.mx
    dir = dir[numpy.bitwise_or(dir[:, 1] != 0,
                               dir[:, 2] != 0), :]

    # Split by 1st column
    npat = gcm.stepPattern.get_n_patterns()
    stepsCache = dict()
    for q in range(1, npat + 1):
        tmp = dir[dir[:, 0] == q,]
        stepsCache[q] = numpy.array(tmp[:, [1, 2]],
                                    dtype=int)
        stepsCache[q] = numpy.flip(stepsCache[q],0)

    # Mapping lists
    iis = [i]
    ii = [i]
    jjs = [j]
    jj = [j]
    ss = []

    while True:
        if i == 0 and j == 0: break

        # Direction taken, 1-based
        s = gcm.directionMatrix[i, j]

        if s == _INT_MIN: break  # int nan in R

        # undo the steps
        ss.insert(0, s)
        steps = stepsCache[s]

        ns = steps.shape[0]
        for k in range(ns):
            ii.insert(0, i - steps[k, 0])
            jj.insert(0, j - steps[k, 1])

        i -= steps[k, 0]
        j -= steps[k, 1]

        iis.insert(0, i)
        jjs.insert(0, j)

    out = {'index1': numpy.array(ii),
           'index2': numpy.array(jj),
           'index1s': numpy.array(iis),
           'index2s': numpy.array(jjs),
           'stepsTaken': numpy.array(ss)}

    return (out)
