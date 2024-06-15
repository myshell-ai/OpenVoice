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

# cython: language_level=3

"""Utility functions for DTW alignments."""

# Author: Toni Giorgino 2018
#
# If you use this software in academic work, please cite:
#  * T. Giorgino. Computing and Visualizing Dynamic Time Warping
#    Alignments in R: The dtw Package. Journal of Statistical
#    Software, v. 31, Issue 7, p. 1 - 24, aug. 2009. ISSN
#    1548-7660. doi:10.18637/jss.v031.i07. http://www.jstatsoft.org/v31/i07/


import warnings

import numpy as np
cimport numpy as np

from cpython cimport array



__all__ = ["_computeCM_wrapper"]



cdef extern from "dtw_core.h":
  void computeCM(			
	       const int *s,		
	       const int *wm,		
	       const double *lm,	
	       const int *nstepsp,	
	       const double *dir,	
	       double *cm,      # IN+OUT
	       int *sm          # OUT
  ) 




  
def _computeCM_wrapper(int [:,::1] wm not None,
                       double [:,::1] lm not None,
                       int [:] nstepsp not None,
                       double [::1] dir not None,
                       double [:,::1] cm not None,
                       int [:,::1] sm = None  ):

    # Memory ordering is transposed (fortran-like in R). 
    st = np.array([wm.shape[1],
                   wm.shape[0]], dtype=np.int32)
    cdef int [:] s = st

    sm = np.full_like(lm.base, -1, dtype=np.int32)

    computeCM(&s[0],
              &wm[0,0],
              &lm[0,0],
              &nstepsp[0],
              &dir[0],
              &cm[0,0],
              &sm[0,0])

    return { 'costMatrix': cm.base,
             'directionMatrix': sm.base }


    
  
def _test_computeCM(TS=5):

    DTYPE = np.int32
    
    twm = np.ones((TS, TS), dtype=DTYPE)

    tlm = np.zeros( (TS,TS), dtype=np.double)
    for i in range(TS):
        for j in range(TS):
            tlm[i,j]=(i+1)*(j+1)

    tnstepsp = np.array([6], dtype=DTYPE)

    tdir = np.array( (1, 1, 2, 2, 3, 3, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,-1, 1,-1, 1,-1, 1),
                                     dtype=np.double)

    tcm = np.full_like(tlm, np.nan, dtype=np.double)
    tcm[0,0] = tlm[0,0]

    out = _computeCM_wrapper(twm,
                             tlm,
                             tnstepsp,
                             tdir,
                             tcm)
    return out
    
    
