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

"""Step Pattern handling

See documentation for the StepPattern class.
"""

import numpy


class StepPattern:
    # IMPORT_RDOCSTRING stepPattern
    """Step patterns for DTW

A ``stepPattern`` object lists the transitions allowed while searching
for the minimum-distance path. DTW variants are implemented by passing
one of the objects described in this page to the ``stepPattern``
argument of the [dtw()] call.

**Details**

A step pattern characterizes the matching model and slope constraint
specific of a DTW variant. They also known as local- or
slope-constraints, transition types, production or recursion rules
(GiorginoJSS).

**Pre-defined step patterns**

::

      ## Well-known step patterns
      symmetric1
      symmetric2
      asymmetric

      ## Step patterns classified according to Rabiner-Juang (Rabiner1993)
      rabinerJuangStepPattern(type,slope_weighting="d",smoothed=False)

      ## Slope-constrained step patterns from Sakoe-Chiba (Sakoe1978)
      symmetricP0;  asymmetricP0
      symmetricP05; asymmetricP05
      symmetricP1;  asymmetricP1
      symmetricP2;  asymmetricP2

      ## Step patterns classified according to Rabiner-Myers (Myers1980)
      typeIa;   typeIb;   typeIc;   typeId;
      typeIas;  typeIbs;  typeIcs;  typeIds;  # smoothed
      typeIIa;  typeIIb;  typeIIc;  typeIId;
      typeIIIc; typeIVc;

      ## Miscellaneous
      mori2006;
      rigid;

A variety of classification schemes have been proposed for step
patterns, including Sakoe-Chiba (Sakoe1978); Rabiner-Juang
(Rabiner1993); and Rabiner-Myers (Myers1980). The ``dtw`` package
implements all of the transition types found in those papers, with the
exception of Itakura’s and Velichko-Zagoruyko’s steps, which require
subtly different algorithms (this may be rectified in the future).
Itakura recursion is almost, but not quite, equivalent to ``typeIIIc``.

For convenience, we shall review pre-defined step patterns grouped by
classification. Note that the same pattern may be listed under different
names. Refer to paper (GiorginoJSS) for full details.

**1. Well-known step patterns**

Common DTW implementations are based on one of the following transition
types.

``symmetric2`` is the normalizable, symmetric, with no local slope
constraints. Since one diagonal step costs as much as the two equivalent
steps along the sides, it can be normalized dividing by ``N+M``
(query+reference lengths). It is widely used and the default.

``asymmetric`` is asymmetric, slope constrained between 0 and 2. Matches
each element of the query time series exactly once, so the warping path
``index2~index1`` is guaranteed to be single-valued. Normalized by ``N``
(length of query).

``symmetric1`` (or White-Neely) is quasi-symmetric, no local constraint,
non-normalizable. It is biased in favor of oblique steps.

**2. The Rabiner-Juang set**

A comprehensive table of step patterns is proposed in Rabiner-Juang’s
book (Rabiner1993), tab. 4.5. All of them can be constructed through the
``rabinerJuangStepPattern(type,slope_weighting,smoothed)`` function.

The classification foresees seven families, labelled with Roman numerals
I-VII; here, they are selected through the integer argument ``type``.
Each family has four slope weighting sub-types, named in sec. 4.7.2.5 as
“Type (a)” to “Type (d)”; they are selected passing a character argument
``slope_weighting``, as in the table below. Furthermore, each subtype
can be either plain or smoothed (figure 4.44); smoothing is enabled
setting the logical argument ``smoothed``. (Not all combinations of
arguments make sense.)

::

     Subtype | Rule       | Norm | Unbiased 
     --------|------------|------|---------
        a    | min step   |  --  |   NO 
        b    | max step   |  --  |   NO 
        c    | Di step    |   N  |  YES 
        d    | Di+Dj step | N+M  |  YES 

**3. The Sakoe-Chiba set**

Sakoe-Chiba (Sakoe1978) discuss a family of slope-constrained patterns;
they are implemented as shown in page 47, table I. Here, they are called
``symmetricP<x>`` and ``asymmetricP<x>``, where ``<x>`` corresponds to
Sakoe’s integer slope parameter *P*. Values available are accordingly:
``0`` (no constraint), ``1``, ``05`` (one half) and ``2``. See
(Sakoe1978) for details.

**4. The Rabiner-Myers set**

The ``type<XX><y>`` step patterns follow the older Rabiner-Myers’
classification proposed in (Myers1980) and (MRR1980). Note that this is
a subset of the Rabiner-Juang set (Rabiner1993), and the latter should
be preferred in order to avoid confusion. ``<XX>`` is a Roman numeral
specifying the shape of the transitions; ``<y>`` is a letter in the
range ``a-d`` specifying the weighting used per step, as above;
``typeIIx`` patterns also have a version ending in ``s``, meaning the
smoothing is used (which does not permit skipping points). The
``typeId, typeIId`` and ``typeIIds`` are unbiased and symmetric.

**5. Others**

The ``rigid`` pattern enforces a fixed unitary slope. It only makes
sense in combination with ``open_begin=True``, ``open_end=True`` to find
gapless subsequences. It may be seen as the ``P->inf`` limiting case in
Sakoe’s classification.

``mori2006`` is Mori’s asymmetric step-constrained pattern (Mori2006).
It is normalized by the matched reference length.

[mvmStepPattern()] implements Latecki’s Minimum Variance Matching
algorithm, and it is described in its own page.

**Methods**

``print_stepPattern`` prints an user-readable description of the
recurrence equation defined by the given pattern.

``plot_stepPattern`` graphically displays the step patterns productions
which can lead to element (0,0). Weights are shown along the step
leading to the corresponding element.

``t_stepPattern`` transposes the productions and normalization hint so
that roles of query and reference become reversed.

Parameters
----------
x : 
    a step pattern object
type : 
    path specification, integer 1..7 (see (Rabiner1993), table 4.5)
slope_weighting : 
    slope weighting rule: character `"a"` to `"d"` (see (Rabiner1993), sec. 4.7.2.5)
smoothed : 
    logical, whether to use smoothing (see (Rabiner1993), fig. 4.44)
... : 
    additional arguments to [print()].

Notes
-----

Constructing ``stepPattern`` objects is tricky and thus undocumented.
For a commented example please see source code for ``symmetricP1``.

References
----------

-  (GiorginoJSS) Toni Giorgino. *Computing and Visualizing Dynamic Time
   Warping Alignments in R: The dtw Package.* Journal of Statistical
   Software, 31(7), 1-24.
   `doi:10.18637/jss_v031.i07 <https://doi.org/10.18637/jss_v031.i07>`__
-  (Itakura1975) Itakura, F., *Minimum prediction residual principle
   applied to speech recognition,* Acoustics, Speech, and Signal
   Processing, IEEE Transactions on , vol.23, no.1, pp. 67-72, Feb 1975.
   `doi:10.1109/TASSP.1975.1162641 <https://doi.org/10.1109/TASSP.1975.1162641>`__
-  (MRR1980) Myers, C.; Rabiner, L. & Rosenberg, A. *Performance
   tradeoffs in dynamic time warping algorithms for isolated word
   recognition*, IEEE Trans. Acoust., Speech, Signal Process., 1980, 28,
   623-635.
   `doi:10.1109/TASSP.1980.1163491 <https://doi.org/10.1109/TASSP.1980.1163491>`__
-  (Mori2006) Mori, A.; Uchida, S.; Kurazume, R.; Taniguchi, R.;
   Hasegawa, T. & Sakoe, H. Early Recognition and Prediction of Gestures
   Proc. 18th International Conference on Pattern Recognition ICPR 2006,
   2006, 3, 560-563.
   `doi:10.1109/ICPR.2006.467 <https://doi.org/10.1109/ICPR.2006.467>`__
-  (Myers1980) Myers, Cory S. *A Comparative Study Of Several Dynamic
   Time Warping Algorithms For Speech Recognition*, MS and BS thesis,
   Dept. of Electrical Engineering and Computer Science, Massachusetts
   Institute of Technology, archived Jun 20 1980,
   https://hdl_handle_net/1721.1/27909
-  (Rabiner1993) Rabiner, L. R., & Juang, B.-H. (1993). *Fundamentals of
   speech recognition.* Englewood Cliffs, NJ: Prentice Hall.
-  (Sakoe1978) Sakoe, H.; Chiba, S., *Dynamic programming algorithm
   optimization for spoken word recognition,* Acoustics, Speech, and
   Signal Processing, IEEE Transactions on , vol.26, no.1, pp. 43-49,
   Feb 1978
   `doi:10.1109/TASSP.1978.1163055 <https://doi.org/10.1109/TASSP.1978.1163055>`__

Examples
--------
>>> from dtw import *
>>> import numpy as np

The usual (normalizable) symmetric step pattern
Step pattern recursion, defined as:
 g[i,j] = min(
   g[i,j-1] + d[i,j] ,
   g[i-1,j-1] + 2 * d[i,j] ,
   g[i-1,j] + d[i,j] ,
)

>>> print(symmetric2)		 #doctest: +NORMALIZE_WHITESPACE
Step pattern recursion:
 g[i,j] = min(
     g[i-1,j-1] + 2 * d[i  ,j  ] ,
     g[i  ,j-1] +     d[i  ,j  ] ,
     g[i-1,j  ] +     d[i  ,j  ] ,
 )
<BLANKLINE>
Normalization hint: N+M
<BLANKLINE>

The well-known plotting style for step patterns

>>> import matplotlib.pyplot as plt;		# doctest: +SKIP
... symmetricP2.plot().set_title("Sakoe's Symmetric P=2 recursion")

Same example seen in ?dtw , now with asymmetric step pattern

>>> (query, reference) = dtw_test_data.sin_cos()

Do the computation

>>> asy = dtw(query, reference, keep_internals=True,
... 	  	     step_pattern=asymmetric);

>>> dtwPlot(asy,type="density"			# doctest: +SKIP
...         ).set_title("Sine and cosine, asymmetric step")

Hand-checkable example given in [Myers1980] p 61 - see JSS paper

>>> tm = numpy.reshape( [1, 3, 4, 4, 5, 2, 2, 3, 3, 4, 3, 1, 1, 1, 3, 4, 2,
...                      3, 3, 2, 5, 3, 4, 4, 1], (5,5), "F" )

"""
    # ENDIMPORT

    def __init__(self, mx, hint="NA"):
        self.mx = numpy.array(mx, dtype=numpy.double)
        self.hint = hint

    def get_n_rows(self):
        """Total number of steps in the recursion."""
        return self.mx.shape[0]

    def get_n_patterns(self):
        """Number of rules in the recursion."""
        return int(numpy.max(self.mx[:, 0]))

    def T(self):
        """Transpose a step pattern."""
        tmx = self.mx.copy()
        tmx = tmx[:, [0, 2, 1, 3]]
        th = self.hint
        if th == "N":
            th = "M"
        elif th == "M":
            th = "N"
        tsp = StepPattern(tmx, th)
        return tsp

    def __str__(self):
        np = self.get_n_patterns()
        head = " g[i,j] = min(\n"

        body = ""
        for p in range(1, np + 1):
            steps = self._extractpattern(p)
            ns = steps.shape[0]
            steps = numpy.flip(steps, 0)

            for s in range(ns):
                di, dj, cc = steps[s, :]
                dis = "" if di == 0 else f"{-int(di)}"
                djs = "" if dj == 0 else f"{-int(dj)}"
                dijs = f"i{dis:2},j{djs:2}"

                if cc == -1:
                    gs = f"    g[{dijs}]"
                    body = body + " " + gs
                else:
                    ccs = "    " if cc == 1 else f"{cc:2.2g} *"
                    ds = f"+{ccs} d[{dijs}]"
                    body = body + " " + ds
            body = body + " ,\n"

        tail = " ) \n\n"
        ntxt = f"Normalization hint: {self.hint}\n"

        return "Step pattern recursion:\n" + head + body + tail + ntxt

    def plot(self):
        """Provide a visual description of a StepPattern object"""
        import matplotlib.pyplot as plt
        x = self.mx
        pats = numpy.arange(1, 1 + self.get_n_patterns())

        alpha = .5
        fudge = [0, 0]

        fig, ax = plt.subplots(figsize=(6, 6))
        for i in pats:
            ss = x[:, 0] == i
            ax.plot(-x[ss, 1], -x[ss, 2], lw=2, color="tab:blue")
            ax.plot(-x[ss, 1], -x[ss, 2], 'o', color="black", marker="o", fillstyle="none")

            if numpy.sum(ss) == 1: continue

            xss = x[ss, :]
            xh = alpha * xss[:-1, 1] + (1 - alpha) * xss[1:, 1]
            yh = alpha * xss[:-1, 2] + (1 - alpha) * xss[1:, 2]

            for xx, yy, tt in zip(xh, yh, xss[1:, 3]):
                ax.annotate("{:.2g}".format(tt), (-xx - fudge[0],
                                                  -yy - fudge[1]))

        endpts = x[:, 3] == -1
        ax.plot(-x[endpts, 1], -x[endpts, 2], 'o', color="black")

        ax.set_xlabel("Query index")
        ax.set_ylabel("Reference index")
        ax.set_xticks(numpy.unique(-x[:, 1]))
        ax.set_yticks(numpy.unique(-x[:, 2]))
        return ax

    def _extractpattern(self, sn):
        sp = self.mx
        sbs = sp[:, 0] == sn
        spl = sp[sbs, 1:]
        return numpy.flip(spl, 0)

    def _mkDirDeltas(self):
        out = numpy.array(self.mx, dtype=numpy.int32)
        out = out[out[:, 3] == -1, :]
        out = out[:, [1, 2]]
        return out

    def _get_p(self):
        # Dimensions are reversed wrt R
        s = self.mx[:, [0, 2, 1, 3]]
        return s.T.reshape(-1)



# Alternate constructor for ease of R import
def _c(*v):
    va = numpy.array([*v])
    if len(va) % 4 != 0:
        _error("Internal error in _c constructor")
    va = va.reshape((-1, 4))
    return (va)


# Kludge because lambda: raise doesn't work
def _error(s):
    raise ValueError(s)


##################################################
##################################################

# Reimplementation of the building process

class _P:
    def __init__(self, pid, subtype, smoothing):
        self.subtype = subtype
        self.smoothing = smoothing
        self.pid = pid
        self.i = [0]
        self.j = [0]

    def step(self, di, dj):  # equivalent to .Pstep
        self.i.append(di)
        self.j.append(dj)
        return self

    def get(self):  # eqv to .Pend
        ia = numpy.array(self.i, dtype=numpy.double)
        ja = numpy.array(self.j, dtype=numpy.double)
        si = numpy.cumsum(ia)
        sj = numpy.cumsum(ja)
        ni = numpy.max(si) - si  # ?
        nj = numpy.max(sj) - sj
        if self.subtype == "a":
            w = numpy.minimum(ia, ja)
        elif self.subtype == "b":
            w = numpy.maximum(ia, ja)
        elif self.subtype == "c":
            w = ia
        elif self.subtype == "d":
            w = ia + ja
        else:
            _error("Unsupported subtype")

        if self.smoothing:
            # if self.pid==3:                import ipdb; ipdb.set_trace()
            w[1:] = numpy.mean(w[1:])

        w[0] = -1.0

        nr = len(w)
        mx = numpy.zeros((nr, 4))
        mx[:, 0] = self.pid
        mx[:, 1] = ni
        mx[:, 2] = nj
        mx[:, 3] = w
        return mx


def rabinerJuangStepPattern(ptype, slope_weighting="d", smoothed=False):
    """Construct a pattern classified according to the Rabiner-Juang scheme (Rabiner1993)

See documentation for the StepPattern class.
"""

    f = {
        1: _RJtypeI,
        2: _RJtypeII,
        3: _RJtypeIII,
        4: _RJtypeIV,
        5: _RJtypeV,
        6: _RJtypeVI,
        7: _RJtypeVII
    }.get(ptype, lambda: _error("Invalid type"))

    r = f(slope_weighting, smoothed)
    norm = "NA"
    if slope_weighting == "c":
        norm = "N"
    elif slope_weighting == "d":
        norm = "N+M"

    return StepPattern(r, norm)


def _RJtypeI(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 0).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(0, 1).get()])


def _RJtypeII(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(1, 1).step(0, 1).get()])


def _RJtypeIII(s, m):
    return numpy.vstack([
        _P(1, s, m).step(2, 1).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(1, 2).get()])


def _RJtypeIV(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).get(),
        _P(2, s, m).step(1, 2).step(1, 0).get(),
        _P(3, s, m).step(1, 1).get(),
        _P(4, s, m).step(1, 2).get(),
    ])


def _RJtypeV(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).step(1, 0).get(),
        _P(2, s, m).step(1, 1).step(1, 0).get(),
        _P(3, s, m).step(1, 1).get(),
        _P(4, s, m).step(1, 1).step(0, 1).get(),
        _P(5, s, m).step(1, 1).step(0, 1).step(0, 1).get(),
    ])


def _RJtypeVI(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 1).step(1, 0).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(1, 1).step(1, 1).step(0, 1).get()
    ])


def _RJtypeVII(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).step(1, 0).get(),
        _P(2, s, m).step(1, 2).step(1, 0).step(1, 0).get(),
        _P(3, s, m).step(1, 3).step(1, 0).step(1, 0).get(),
        _P(4, s, m).step(1, 1).step(1, 0).get(),
        _P(5, s, m).step(1, 2).step(1, 0).get(),
        _P(6, s, m).step(1, 3).step(1, 0).get(),
        _P(7, s, m).step(1, 1).get(),
        _P(8, s, m).step(1, 2).get(),
        _P(9, s, m).step(1, 3).get(),
    ])


##########################################################################################
##########################################################################################

## Everything here is semi auto-generated from the R source. Don't
## edit!


##################################################
##################################################


##
## Various step patterns, defined as internal variables
##
## First column: enumerates step patterns.
## Second   	 step in query index
## Third	 step in reference index
## Fourth	 weight if positive, or -1 if starting point
##
## For \cite{} see dtw.bib in the package
##


## Widely-known variants

## White-Neely symmetric (default)
## aka Quasi-symmetric \cite{White1976}
## normalization: no (N+M?)
symmetric1 = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 0, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
));

## Normal symmetric
## normalization: N+M
symmetric2 = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 2,
    2, 0, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N+M");

## classic asymmetric pattern: max slope 2, min slope 0
## normalization: N
asymmetric = StepPattern(_c(
    1, 1, 0, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 0, 1
), "N");

# % \item{\code{symmetricVelichkoZagoruyko}}{symmetric, reproduced from %
# [Sakoe1978]. Use distance matrix \code{1-d}}
# 

## normalization: max[N,M]
## note: local distance matrix is 1-d
## \cite{Velichko}
_symmetricVelichkoZagoruyko = StepPattern(_c(
    1, 0, 1, -1,
    2, 1, 1, -1,
    2, 0, 0, -1.001,
    3, 1, 0, -1));

# % \item{\code{asymmetricItakura}}{asymmetric, slope contrained 0.5 -- 2
# from reference [Itakura1975]. This is the recursive definition % that
# generates the Itakura parallelogram; }
# 

## Itakura slope-limited asymmetric \cite{Itakura1975}
## Max slope: 2; min slope: 1/2
## normalization: N
_asymmetricItakura = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1,
    4, 2, 2, -1,
    4, 1, 0, 1,
    4, 0, 0, 1
));

#############################
## Slope-limited versions
##
## Taken from Table I, page 47 of "Dynamic programming algorithm
## optimization for spoken word recognition," Acoustics, Speech, and
## Signal Processing, vol.26, no.1, pp. 43-49, Feb 1978 URL:
## http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1163055
##
## Mostly unchecked


## Row P=0
symmetricP0 = symmetric2;

## normalization: N ?
asymmetricP0 = StepPattern(_c(
    1, 0, 1, -1,
    1, 0, 0, 0,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N");

## alternative implementation
_asymmetricP0b = StepPattern(_c(
    1, 0, 1, -1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N");

## Row P=1/2
symmetricP05 = StepPattern(_c(
    1, 1, 3, -1,
    1, 0, 2, 2,
    1, 0, 1, 1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 1, 2,
    2, 0, 0, 1,
    3, 1, 1, -1,
    3, 0, 0, 2,
    4, 2, 1, -1,
    4, 1, 0, 2,
    4, 0, 0, 1,
    5, 3, 1, -1,
    5, 2, 0, 2,
    5, 1, 0, 1,
    5, 0, 0, 1
), "N+M");

asymmetricP05 = StepPattern(_c(
    1, 1, 3, -1,
    1, 0, 2, 1 / 3,
    1, 0, 1, 1 / 3,
    1, 0, 0, 1 / 3,
    2, 1, 2, -1,
    2, 0, 1, .5,
    2, 0, 0, .5,
    3, 1, 1, -1,
    3, 0, 0, 1,
    4, 2, 1, -1,
    4, 1, 0, 1,
    4, 0, 0, 1,
    5, 3, 1, -1,
    5, 2, 0, 1,
    5, 1, 0, 1,
    5, 0, 0, 1
), "N");

## Row P=1
## Implementation of Sakoe's P=1, Symmetric algorithm

symmetricP1 = StepPattern(_c(
    1, 1, 2, -1,  # First branch: g(i-1,j-2)+
    1, 0, 1, 2,  # + 2d(i  ,j-1)
    1, 0, 0, 1,  # +  d(i  ,j)
    2, 1, 1, -1,  # Second branch: g(i-1,j-1)+
    2, 0, 0, 2,  # +2d(i,  j)
    3, 2, 1, -1,  # Third branch: g(i-2,j-1)+
    3, 1, 0, 2,  # + 2d(i-1,j)
    3, 0, 0, 1  # +  d(  i,j)
), "N+M");

asymmetricP1 = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 1, .5,
    1, 0, 0, .5,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1
), "N");

## Row P=2
symmetricP2 = StepPattern(_c(
    1, 2, 3, -1,
    1, 1, 2, 2,
    1, 0, 1, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 3, 2, -1,
    3, 2, 1, 2,
    3, 1, 0, 2,
    3, 0, 0, 1
), "N+M");

asymmetricP2 = StepPattern(_c(
    1, 2, 3, -1,
    1, 1, 2, 2 / 3,
    1, 0, 1, 2 / 3,
    1, 0, 0, 2 / 3,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 3, 2, -1,
    3, 2, 1, 1,
    3, 1, 0, 1,
    3, 0, 0, 1
), "N");

################################
## Taken from Table III, page 49.
## Four varieties of DP-algorithm compared

## 1st row:  asymmetric

## 2nd row:  symmetricVelichkoZagoruyko

## 3rd row:  symmetric1

## 4th row:  asymmetricItakura


#############################
## Classified according to Rabiner
##
## Taken from chapter 2, Myers' thesis [4]. Letter is
## the weighting function:
##
##      rule       norm   unbiased
##   a  min step   ~N     NO
##   b  max step   ~N     NO
##   c  x step     N      YES
##   d  x+y step   N+M    YES
##
## Mostly unchecked

# R-Myers     R-Juang
# type I      type II   
# type II     type III
# type III    type IV
# type IV     type VII


typeIa = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 0,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 0
));

typeIb = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 1
));

typeIc = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 0
), "N");

typeId = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 1, 2, -1,
    3, 0, 1, 2,
    3, 0, 0, 1
), "N+M");

## ----------
## smoothed variants of above

typeIas = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, .5,
    1, 0, 0, .5,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, .5,
    3, 0, 0, .5
));

typeIbs = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 1
));

typeIcs = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, .5,
    3, 0, 0, .5
), "N");

typeIds = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1.5,
    1, 0, 0, 1.5,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 1, 2, -1,
    3, 0, 1, 1.5,
    3, 0, 0, 1.5
), "N+M");

## ----------

typeIIa = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 0, 0, 1
));

typeIIb = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 2,
    3, 2, 1, -1,
    3, 0, 0, 2
));

typeIIc = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 0, 0, 2
), "N");

typeIId = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 2,
    2, 1, 2, -1,
    2, 0, 0, 3,
    3, 2, 1, -1,
    3, 0, 0, 3
), "N+M");

## ----------

## Rabiner [3] discusses why this is not equivalent to Itakura's

typeIIIc = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1,
    4, 2, 2, -1,
    4, 1, 0, 1,
    4, 0, 0, 1
), "N");

## ----------

## numbers follow as production rules in fig 2.16

typeIVc = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 1, 3, -1,
    3, 0, 0, 1,
    4, 2, 1, -1,
    4, 1, 0, 1,
    4, 0, 0, 1,
    5, 2, 2, -1,
    5, 1, 0, 1,
    5, 0, 0, 1,
    6, 2, 3, -1,
    6, 1, 0, 1,
    6, 0, 0, 1,
    7, 3, 1, -1,
    7, 2, 0, 1,
    7, 1, 0, 1,
    7, 0, 0, 1,
    8, 3, 2, -1,
    8, 2, 0, 1,
    8, 1, 0, 1,
    8, 0, 0, 1,
    9, 3, 3, -1,
    9, 2, 0, 1,
    9, 1, 0, 1,
    9, 0, 0, 1
), "N");

#############################
## 
## Mori's asymmetric step-constrained pattern. Normalized in the
## reference length.
##
## Mori, A.; Uchida, S.; Kurazume, R.; Taniguchi, R.; Hasegawa, T. &
## Sakoe, H. Early Recognition and Prediction of Gestures Proc. 18th
## International Conference on Pattern Recognition ICPR 2006, 2006, 3,
## 560-563
##

mori2006 = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 3,
    3, 1, 2, -1,
    3, 0, 1, 3,
    3, 0, 0, 3
), "M");

## Completely unflexible: fixed slope 1. Only makes sense with
## open.begin and open.end
rigid = StepPattern(_c(1, 1, 1, -1,
                       1, 0, 0, 1), "N")
