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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// Define R-like functions - Memory alloc'd by R_alloc is automatically freed
#ifdef DTW_R
#include <R.h>
#define dtw_alloc(n,size) (R_alloc(n,size))
#else
// Either standalone or Python
#include <limits.h>
#define R_NaInt INT_MIN
#if _WIN32
#include <malloc.h>
#define dtw_alloc(n,size) (_alloca((n)*(size)))
#else
#define dtw_alloc(n,size) (alloca((n)*(size)))
#endif
#define error(...) { fprintf (stderr, __VA_ARGS__); exit(-1); }
#endif



#ifndef NAN
#error "This code requires native IEEE NAN support. Verify you are using gcc with -std=gnu99 or recent compilers."
#endif


/* undo R indexing */
#define EP(ii,jj) ((jj)*nsteps+(ii))
#define EM(ii,jj) ((jj)*n+(ii))

#define CLEARCLIST { \
  for(int z=0; z<npats; z++) \
    clist[z]=NAN; }




/*
 * Auxiliary function: return the arg min, ignoring NANs, -1 if all NANs
 * TODO: remove isnan and explain, check, time
*/
static inline
int argmin(const double *list, int n)
{
    int ii=-1;
    double vv=INFINITY;
    for(int i=0; i<n; i++) {
        /* The following is a faster equivalent to
         *    if(!isnan(list[i]) && list[i]<vv)
         * because   (NAN < x) is false
         */
        if(list[i]<vv) {
            ii=i;
            vv=list[i];
        }
    }
    return ii;
}



/*
 *  Compute cumulative cost matrix: replaces kernel in globalCostMatrix.R
 */

/* For now, this code is also valid outside R, as a test unit.
 * This means that we have to refrain to use R-specific functions,
 * such as R_malloc, or conditionally provide replacements.
 */

/* R matrix fastest index is row */

void computeCM(			/* IN */
    const int *s,		/* mtrx dimensions, int */
    const int *wm,		/* windowing matrix, logical=int */
    const double *lm,	/* local cost mtrx, numeric */
    const int *nstepsp,	/* no of steps in stepPattern, int */
    const double *dir,	/* stepPattern description, numeric */
    /* IN+OUT */
    double *cm,		/* cost matrix, numeric */
    /* OUT */
    int *sm			/* direction mtrx, int */
)
{

    /* recover matrix dim */
    int n=s[0],m=s[1];		/* query,template as usual*/
    int nsteps=*nstepsp;


    /* copy steppattern description to ints,
       so we'll do indexing arithmetic on ints
    */
    int *pn,*di,*dj;
    double *sc;

    pn=(int*) dtw_alloc((size_t)nsteps,sizeof(int)); /* pattern id */
    di=(int*) dtw_alloc((size_t)nsteps,sizeof(int)); /* delta i */
    dj=(int*) dtw_alloc((size_t)nsteps,sizeof(int)); /* delta j */
    sc=(double*) dtw_alloc((size_t)nsteps,sizeof(double)); /* step cost */

    for(int i=0; i<nsteps; i++) {
        pn[i]=(int)dir[EP(i,0)]-1;	/* Indexing C-way */
        di[i]=(int)dir[EP(i,1)];
        dj[i]=(int)dir[EP(i,2)];
        sc[i]=dir[EP(i,3)];

        if(pn[i]<0 || pn[i]>=nsteps) {
            error("Error on pattern row %d, pattern number %d out of bounds\n",
                  i,pn[i]+1);
        }
    }

    /* assuming pattern ids are in ascending order */
    int npats=pn[nsteps-1]+1;

    /* prepare a cost list per pattern */
    double *clist=(double*)
                  dtw_alloc((size_t)npats,sizeof(double));

    /* we do not initialize the seed - the caller is supposed
       to do so
       cm[0]=lm[0];
     */

    /* clear the direction matrix */
    for(int i=0; i<m*n; i++)
        sm[i]=R_NaInt;			/* should be NA_INTEGER? */


    /* lets go */
    for(int j=0; j<m; j++) {
        for(int i=0; i<n; i++) {

            /* out of window? */
            if(!wm[EM(i,j)])
                continue;

            /* already initialized? */
            if(!isnan(cm[EM(i,j)]))
                continue;

            CLEARCLIST;
            for(int s=0; s<nsteps; s++) {
                int p=pn[s];		/* indexing C-way */

                int ii=i-di[s];
                int jj=j-dj[s];
                if(ii>=0 && jj>=0) {	/* address ok? C convention */
                    double cc=sc[s];
                    if(cc==-1.0) {
                        clist[p]=cm[EM(ii,jj)];
                    } else {		/* we rely on NAN to propagate */
                        clist[p] += cc*lm[EM(ii,jj)];
                    }
                }
            }

            int minc=argmin(clist,npats);
            if(minc>-1) {
                cm[EM(i,j)]=clist[minc];
                sm[EM(i,j)]=minc+1;	/* convert to 1-based  */
            }
        }
    }
}





/* Test as follows:

   R CMD SHLIB -d computeCM.c

    dyn.load("computeCM.so")
    lm <- matrix(nrow = 6, ncol = 6, byrow = TRUE, c(
      1, 1, 2, 2, 3, 3,
      1, 1, 1, 2, 2, 2,
      3, 1, 2, 2, 3, 3,
      3, 1, 2, 1, 1, 2,
      3, 2, 1, 2, 1, 2,
      3, 3, 3, 2, 1, 2
    ))
    step.matrix <- as.matrix(structure(c(1, 1, 2, 2, 3, 3, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 2,
     0, -1, 1, -1, 1, -1, 1), .Dim = c(6L, 4L), class = "stepPattern", npat = 3, norm = "N"))
    nsteps<-dim(step.matrix)[1]
    wm <- matrix(TRUE,6,6)
    cm <- matrix(NA,6,6)
    cm[1,1] <- lm[1,1];
    sm <- matrix(NA,6,6)

    out<-.C("computeCM",NAOK=TRUE,
       as.integer(dim(cm)),
       as.logical(wm),
       as.double(lm),
       as.integer(nsteps),
       as.double(step.matrix),
       cmo=as.double(cm),
       smo=as.integer(sm))

    cmoo<-matrix(out$cmo,6,6)
    smoo<-matrix(out$smo,6,6)

    storage.mode(wm) <- "logical"
    storage.mode(lm) <- "double"
    storage.mode(cm) <- "double"
    storage.mode(step.matrix) <- "double"

    out2<-.Call("computeCM_Call", wm, lm, cm, step.matrix)

*/








#ifdef TEST_UNIT
/* --------------------------------------------------
 * Unit test - for debugging
 */




/*
 * Printout a matrix.
 * int *s: s[0] - no. of rows
 *         s[1] - no. of columns
 * double *mm: matrix to print
 * double *r: return value
 */

void tm_print(int *s, double *mm, double *r)
{
    int i,j;
    int n=s[0],m=s[1];
    FILE *f=stdout;

    for(i=0; i<n; i++) {
        for(j=0; j<m; j++) {
            double val=mm[j*n+i];
            if(isnan(val)) {
                printf("NAN %d %d\n",i,j);
            }
            fprintf(f,"[%2d,%2d] = %4.2lf    ",i,j,val);
        }
        fprintf(f,"\n");
    }
    *r=-1;
    // fclose(f);
    printf("** tm dump end **\n");
}


/* test  equivalent to the following
   mylm<-outer(1:10,1:10)
   globalCostNative(mylm)->myg2
*/

#define TS 5000
#define TSS (TS*TS)

void test_computeCM()
{
    int ts[]= {TS,TS};
    int *twm;
    double *tlm;
    int tnstepsp[]= {6};
    double tdir[]= {1, 1, 2, 2, 3, 3, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,-1, 1,-1, 1,-1, 1};
    double *tcm;
    int *tsm;

    int i,j;

    twm=malloc(TSS*sizeof(int));
    for( i=0; i<TSS; i++)
        twm[i]=1;

    tlm=malloc(TSS*sizeof(double));
    for( i=0; i<TS; i++)
        for( j=0; j<TS; j++)
            tlm[i*TS+j]=(i+1)*(j+1);


    tcm=malloc(TSS*sizeof(double));
    for( i=0; i<TS; i++)
        for( j=0; j<TS; j++)
            tcm[i*TS+j]=NAN;
    tcm[0]=tlm[0];

    tsm=malloc(TSS*sizeof(int));


//  double r=-2;

//  tm_print(ts,tlm,&r);

    /* pretend we'r R */
    computeCM(ts,twm,tlm,tnstepsp,
              tdir,tcm,tsm);

//  tm_print(ts,tcm,&r);

    free(twm);
    free(tlm);
    free(tcm);
    free(tsm);

}



void test_argmin()
{
    int n=5;

    double t1[]= {10,-2,NAN,2,NAN};
    double t2[]= {10,-2,-3,2,-4};
    double t3[]= {10,INFINITY,-3,2,-4};
    double t4[]= {NAN,NAN,NAN,NAN,NAN};

    printf("argmin(t1,n)==%d, should be 1\n",argmin(t1,n));
    printf("argmin(t2,n)==%d, should be 4\n",argmin(t2,n));
    printf("argmin(t3,n)==%d, should be 4\n",argmin(t3,n));
    printf("argmin(t4,n)==%d, should be -1\n",argmin(t4,n));

}



int main(int argc,char **argv)
{
    test_argmin();
    test_computeCM();
}

#endif

