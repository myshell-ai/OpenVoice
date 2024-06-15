import numpy
from dtw.window import noWindow
from dtw._dtw_utils import _computeCM_wrapper


def _globalCostMatrix(lm,
                      step_pattern,
                      window_function,
                      seed,
                      win_args):

    ITYPE = numpy.int32
    n, m = lm.shape

    if window_function == noWindow:  # for performance
        wm = numpy.full_like(lm, 1, dtype=ITYPE)
    else:
        ix, jx = numpy.indices((n,m))
        wm = window_function(ix, jx,
                             query_size=n,
                             reference_size=m,
                             **win_args)
        wm = wm.astype(ITYPE)   # Convert False/True to 0/1

    nsteps = numpy.array([step_pattern.get_n_rows()], dtype=ITYPE)

    dir = numpy.array(step_pattern._get_p(), dtype=numpy.double)

    if seed is not None:
        cm = seed
    else:
        cm = numpy.full_like(lm, numpy.nan, dtype=numpy.double)
        cm[0, 0] = lm[0, 0]

    # All input arguments
    out = _computeCM_wrapper(wm,
                             lm,
                             nsteps,
                             dir,
                             cm)

    out['stepPattern'] = step_pattern;
    return out


def _test_computeCM2(TS=5):
    import numpy as np
    ITYPE = np.int32

    twm = np.ones((TS, TS), dtype=ITYPE)

    tlm = np.zeros((TS, TS), dtype=np.double)
    for i in range(TS):
        for j in range(TS):
            tlm[i, j] = (i + 1) * (j + 1)

    tnstepsp = np.array([6], dtype=ITYPE)

    tdir = np.array((1, 1, 2, 2, 3, 3, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, -1, 1, -1, 1, -1, 1),
                    dtype=np.double)

    tcm = np.full_like(tlm, np.nan, dtype=np.double)
    tcm[0, 0] = tlm[0, 0]

    out = _computeCM_wrapper(twm,
                             tlm,
                             tnstepsp,
                             tdir,
                             tcm)
    return out
