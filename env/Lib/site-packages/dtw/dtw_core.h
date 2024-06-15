//
// Copyright (c) 2006-2019 of Toni Giorgino
//
// This file is part of the DTW package.
//
// DTW is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DTW is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with DTW.  If not, see <http://www.gnu.org/licenses/>.
//



#ifndef _DTW_CORE_H
#define _DTW_CORE_H

void computeCM(                 /* IN */
    const int *s,            /* mtrx dimensions, int */
    const int *wm,           /* windowing matrix, logical=int */
    const double *lm,        /* local cost mtrx, numeric */
    const int *nstepsp,      /* no of steps in stepPattern, int */
    const double *dir,       /* stepPattern description, numeric */
    /* IN+OUT */
    double *cm,              /* cost matrix, numeric */
    /* OUT */
    int *sm                  /* direction mtrx, int */
) ;

#endif
