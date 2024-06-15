#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Sequential modeling
===================

Sequence alignment
------------------
.. autosummary::
    :toctree: generated/

    dtw
    rqa

Viterbi decoding
----------------
.. autosummary::
    :toctree: generated/

    viterbi
    viterbi_discriminative
    viterbi_binary

Transition matrices
-------------------
.. autosummary::
    :toctree: generated/

    transition_uniform
    transition_loop
    transition_cycle
    transition_local
"""

import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from .util import pad_center, fill_off_diagonal, tiny, expand_to
from .util.exceptions import ParameterError
from .filters import get_window
from .util.decorators import deprecate_positional_args

__all__ = [
    "dtw",
    "dtw_backtracking",
    "rqa",
    "viterbi",
    "viterbi_discriminative",
    "viterbi_binary",
    "transition_uniform",
    "transition_loop",
    "transition_cycle",
    "transition_local",
]


@deprecate_positional_args
def dtw(
    X=None,
    Y=None,
    *,
    C=None,
    metric="euclidean",
    step_sizes_sigma=None,
    weights_add=None,
    weights_mul=None,
    subseq=False,
    backtrack=True,
    global_constraints=False,
    band_rad=0.25,
    return_steps=False,
):
    """Dynamic time warping (DTW).

    This function performs a DTW and path backtracking on two sequences.
    We follow the nomenclature and algorithmic approach as described in [#]_.

    .. [#] Meinard Mueller
           Fundamentals of Music Processing — Audio, Analysis, Algorithms, Applications
           Springer Verlag, ISBN: 978-3-319-21944-8, 2015.

    Parameters
    ----------
    X : np.ndarray [shape=(..., K, N)]
        audio feature matrix (e.g., chroma features)

        If ``X`` has more than two dimensions (e.g., for multi-channel inputs), all leading
        dimensions are used when computing distance to ``Y``.

    Y : np.ndarray [shape=(..., K, M)]
        audio feature matrix (e.g., chroma features)

    C : np.ndarray [shape=(N, M)]
        Precomputed distance matrix. If supplied, X and Y must not be supplied and
        ``metric`` will be ignored.

    metric : str
        Identifier for the cost-function as documented
        in `scipy.spatial.distance.cdist()`

    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.

    weights_add : np.ndarray [shape=[n, ]]
        Additive weights to penalize certain step sizes.

    weights_mul : np.ndarray [shape=[n, ]]
        Multiplicative weights to penalize certain step sizes.

    subseq : bool
        Enable subsequence DTW, e.g., for retrieval tasks.

    backtrack : bool
        Enable backtracking in accumulated cost matrix.

    global_constraints : bool
        Applies global constraints to the cost matrix ``C`` (Sakoe-Chiba band).

    band_rad : float
        The Sakoe-Chiba band radius (1/2 of the width) will be
        ``int(radius*min(C.shape))``.

    return_steps : bool
        If true, the function returns ``steps``, the step matrix, containing
        the indices of the used steps from the cost accumulation step.

    Returns
    -------
    D : np.ndarray [shape=(N, M)]
        accumulated cost matrix.
        D[N, M] is the total alignment cost.
        When doing subsequence DTW, D[N,:] indicates a matching function.
    wp : np.ndarray [shape=(N, 2)]
        Warping path with index pairs.
        Each row of the array contains an index pair (n, m).
        Only returned when ``backtrack`` is True.
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.
        Only returned when ``return_steps`` is True.

    Raises
    ------
    ParameterError
        If you are doing diagonal matching and Y is shorter than X or if an
        incompatible combination of X, Y, and C are supplied.

        If your input dimensions are incompatible.

        If the cost matrix has NaN values.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('brahms'), offset=10, duration=15)
    >>> X = librosa.feature.chroma_cens(y=y, sr=sr)
    >>> noise = np.random.rand(X.shape[0], 200)
    >>> Y = np.concatenate((noise, noise, X, noise), axis=1)
    >>> D, wp = librosa.sequence.dtw(X, Y, subseq=True)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
    ...                                ax=ax[0])
    >>> ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
    >>> ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    >>> ax[0].legend()
    >>> fig.colorbar(img, ax=ax[0])
    >>> ax[1].plot(D[-1, :] / wp.shape[0])
    >>> ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2],
    ...           title='Matching cost function')
    """
    # Default Parameters
    default_steps = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.uint32)
    default_weights_add = np.zeros(3, dtype=np.float64)
    default_weights_mul = np.ones(3, dtype=np.float64)

    if step_sizes_sigma is None:
        # Use the default steps
        step_sizes_sigma = default_steps

        # Use default weights if none are provided
        if weights_add is None:
            weights_add = default_weights_add

        if weights_mul is None:
            weights_mul = default_weights_mul
    else:
        # If we have custom steps but no weights, construct them here
        if weights_add is None:
            weights_add = np.zeros(len(step_sizes_sigma), dtype=np.float64)

        if weights_mul is None:
            weights_mul = np.ones(len(step_sizes_sigma), dtype=np.float64)

        # Make the default step weights infinite so that they are never
        # preferred over custom steps
        default_weights_add.fill(np.inf)
        default_weights_mul.fill(np.inf)

        # Append custom steps and weights to our defaults
        step_sizes_sigma = np.concatenate((default_steps, step_sizes_sigma))
        weights_add = np.concatenate((default_weights_add, weights_add))
        weights_mul = np.concatenate((default_weights_mul, weights_mul))

    if np.any(step_sizes_sigma < 0):
        raise ParameterError("step_sizes_sigma cannot contain negative values")

    if len(step_sizes_sigma) != len(weights_add):
        raise ParameterError("len(weights_add) must be equal to len(step_sizes_sigma)")
    if len(step_sizes_sigma) != len(weights_mul):
        raise ParameterError("len(weights_mul) must be equal to len(step_sizes_sigma)")

    if C is None and (X is None or Y is None):
        raise ParameterError("If C is not supplied, both X and Y must be supplied")
    if C is not None and (X is not None or Y is not None):
        raise ParameterError("If C is supplied, both X and Y must not be supplied")

    c_is_transposed = False

    # calculate pair-wise distances, unless already supplied.
    # C_local will keep track of whether the distance matrix was supplied
    # by the user (False) or constructed locally (True)
    C_local = False
    if C is None:
        C_local = True
        # take care of dimensions
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Perform some shape-squashing here
        # Put the time axes around front
        X = np.swapaxes(X, -1, 0)
        Y = np.swapaxes(Y, -1, 0)

        # Flatten the remaining dimensions
        # Use F-ordering to preserve columns
        X = X.reshape((X.shape[0], -1), order="F")
        Y = Y.reshape((Y.shape[0], -1), order="F")

        try:
            C = cdist(X, Y, metric=metric)
        except ValueError as exc:
            raise ParameterError(
                "scipy.spatial.distance.cdist returned an error.\n"
                "Please provide your input in the form X.shape=(K, N) "
                "and Y.shape=(K, M).\n 1-dimensional sequences should "
                "be reshaped to X.shape=(1, N) and Y.shape=(1, M)."
            ) from exc

        # for subsequence matching:
        # if N > M, Y can be a subsequence of X
        if subseq and (X.shape[0] > Y.shape[0]):
            C = C.T
            c_is_transposed = True

    C = np.atleast_2d(C)

    # if diagonal matching, Y has to be longer than X
    # (X simply cannot be contained in Y)
    if np.array_equal(step_sizes_sigma, np.array([[1, 1]])) and (
        C.shape[0] > C.shape[1]
    ):
        raise ParameterError(
            "For diagonal matching: Y.shape[-1] >= X.shape[-11] "
            "(C.shape[1] >= C.shape[0])"
        )

    max_0 = step_sizes_sigma[:, 0].max()
    max_1 = step_sizes_sigma[:, 1].max()

    # check C here for nans before building global constraints
    if np.any(np.isnan(C)):
        raise ParameterError("DTW cost matrix C has NaN values. ")

    if global_constraints:
        # Apply global constraints to the cost matrix
        if not C_local:
            # If C was provided as input, make a copy here
            C = np.copy(C)
        fill_off_diagonal(C, radius=band_rad, value=np.inf)

    # initialize whole matrix with infinity values
    D = np.ones(C.shape + np.array([max_0, max_1])) * np.inf

    # set starting point to C[0, 0]
    D[max_0, max_1] = C[0, 0]

    if subseq:
        D[max_0, max_1:] = C[0, :]

    # initialize step matrix with -1
    # will be filled in calc_accu_cost() with indices from step_sizes_sigma
    steps = np.zeros(D.shape, dtype=np.int32)

    # these steps correspond to left- (first row) and up-(first column) moves
    steps[0, :] = 1
    steps[:, 0] = 2

    # calculate accumulated cost matrix
    D, steps = __dtw_calc_accu_cost(
        C, D, steps, step_sizes_sigma, weights_mul, weights_add, max_0, max_1
    )

    # delete infinity rows and columns
    D = D[max_0:, max_1:]
    steps = steps[max_0:, max_1:]

    if backtrack:
        if subseq:
            if np.all(np.isinf(D[-1])):
                raise ParameterError(
                    "No valid sub-sequence warping path could "
                    "be constructed with the given step sizes."
                )
            start = np.argmin(D[-1, :])
            wp = __dtw_backtracking(steps, step_sizes_sigma, subseq, start)
        else:
            # perform warping path backtracking
            if np.isinf(D[-1, -1]):
                raise ParameterError(
                    "No valid sub-sequence warping path could "
                    "be constructed with the given step sizes."
                )

            wp = __dtw_backtracking(steps, step_sizes_sigma, subseq)
            if wp[-1] != (0, 0):
                raise ParameterError(
                    "Unable to compute a full DTW warping path. "
                    "You may want to try again with subseq=True."
                )

        wp = np.asarray(wp, dtype=int)

        # since we transposed in the beginning, we have to adjust the index pairs back
        if subseq and (
            (X is not None and Y is not None and X.shape[0] > Y.shape[0])
            or c_is_transposed
            or C.shape[0] > C.shape[1]
        ):
            wp = np.fliplr(wp)
        return_values = [D, wp]
    else:
        return_values = [D]

    if return_steps:
        return_values.append(steps)

    if len(return_values) > 1:
        return tuple(return_values)
    else:
        return return_values[0]


@jit(nopython=True, cache=True)
def __dtw_calc_accu_cost(
    C, D, steps, step_sizes_sigma, weights_mul, weights_add, max_0, max_1
):  # pragma: no cover
    """Calculate the accumulated cost matrix D.

    Use dynamic programming to calculate the accumulated costs.

    Parameters
    ----------
    C : np.ndarray [shape=(N, M)]
        pre-computed cost matrix
    D : np.ndarray [shape=(N, M)]
        accumulated cost matrix
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.
    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.
    weights_add : np.ndarray [shape=[n, ]]
        Additive weights to penalize certain step sizes.
    weights_mul : np.ndarray [shape=[n, ]]
        Multiplicative weights to penalize certain step sizes.
    max_0 : int
        maximum number of steps in step_sizes_sigma in dim 0.
    max_1 : int
        maximum number of steps in step_sizes_sigma in dim 1.

    Returns
    -------
    D : np.ndarray [shape=(N, M)]
        accumulated cost matrix.
        D[N, M] is the total alignment cost.
        When doing subsequence DTW, D[N,:] indicates a matching function.
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.

    See Also
    --------
    dtw
    """
    for cur_n in range(max_0, D.shape[0]):
        for cur_m in range(max_1, D.shape[1]):
            # accumulate costs
            for cur_step_idx, cur_w_add, cur_w_mul in zip(
                range(step_sizes_sigma.shape[0]), weights_add, weights_mul
            ):
                cur_D = D[
                    cur_n - step_sizes_sigma[cur_step_idx, 0],
                    cur_m - step_sizes_sigma[cur_step_idx, 1],
                ]
                cur_C = cur_w_mul * C[cur_n - max_0, cur_m - max_1]
                cur_C += cur_w_add
                cur_cost = cur_D + cur_C

                # check if cur_cost is smaller than the one stored in D
                if cur_cost < D[cur_n, cur_m]:
                    D[cur_n, cur_m] = cur_cost

                    # save step-index
                    steps[cur_n, cur_m] = cur_step_idx

    return D, steps


@jit(nopython=True, cache=True)
def __dtw_backtracking(steps, step_sizes_sigma, subseq, start=None):  # pragma: no cover
    """Backtrack optimal warping path.

    Uses the saved step sizes from the cost accumulation
    step to backtrack the index pairs for an optimal
    warping path.

    Parameters
    ----------
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.
    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.
    subseq : bool
        Enable subsequence DTW, e.g., for retrieval tasks.
    start : int
        Start column index for backtraing (only allowed for ``subseq=True``)

    Returns
    -------
    wp : list [shape=(N,)]
        Warping path with index pairs.
        Each list entry contains an index pair
        (n, m) as a tuple

    See Also
    --------
    dtw
    """
    if start is None:
        cur_idx = (steps.shape[0] - 1, steps.shape[1] - 1)
    else:
        cur_idx = (steps.shape[0] - 1, start)

    wp = []
    # Set starting point D(N, M) and append it to the path
    wp.append((cur_idx[0], cur_idx[1]))

    # Loop backwards.
    # Stop criteria:
    # Setting it to (0, 0) does not work for the subsequence dtw,
    # so we only ask to reach the first row of the matrix.

    while (subseq and cur_idx[0] > 0) or (not subseq and cur_idx != (0, 0)):
        cur_step_idx = steps[(cur_idx[0], cur_idx[1])]

        # save tuple with minimal acc. cost in path
        cur_idx = (
            cur_idx[0] - step_sizes_sigma[cur_step_idx][0],
            cur_idx[1] - step_sizes_sigma[cur_step_idx][1],
        )

        # If we run off the side of the cost matrix, break here
        if min(cur_idx) < 0:
            break

        # append to warping path
        wp.append((cur_idx[0], cur_idx[1]))

    return wp


@deprecate_positional_args
def dtw_backtracking(steps, *, step_sizes_sigma=None, subseq=False, start=None):
    """Backtrack a warping path.

    Uses the saved step sizes from the cost accumulation
    step to backtrack the index pairs for a warping path.

    Parameters
    ----------
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.
    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.
    subseq : bool
        Enable subsequence DTW, e.g., for retrieval tasks.
    start : int
        Start column index for backtraing (only allowed for ``subseq=True``)

    Returns
    -------
    wp : list [shape=(N,)]
        Warping path with index pairs.
        Each list entry contains an index pair
        (n, m) as a tuple

    See Also
    --------
    dtw
    """
    if subseq is False and start is not None:
        raise ParameterError(
            "start is only allowed to be set if subseq is True "
            "(start={}, subseq={})".format(start, subseq)
        )

    # Default Parameters
    default_steps = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.uint32)

    if step_sizes_sigma is None:
        # Use the default steps
        step_sizes_sigma = default_steps
    else:
        # Append custom steps and weights to our defaults
        step_sizes_sigma = np.concatenate((default_steps, step_sizes_sigma))

    wp = __dtw_backtracking(steps, step_sizes_sigma, subseq, start)
    return np.asarray(wp, dtype=int)


@deprecate_positional_args
def rqa(sim, *, gap_onset=1, gap_extend=1, knight_moves=True, backtrack=True):
    """Recurrence quantification analysis (RQA)

    This function implements different forms of RQA as described by
    Serra, Serra, and Andrzejak (SSA). [#]_  These methods take as input
    a self- or cross-similarity matrix ``sim``, and calculate the value
    of path alignments by dynamic programming.

    Note that unlike dynamic time warping (`dtw`), alignment paths here are
    maximized, not minimized, so the input should measure similarity rather
    than distance.

    The simplest RQA method, denoted as `L` (SSA equation 3) and equivalent
    to the method described by Eckman, Kamphorst, and Ruelle [#]_, accumulates
    the length of diagonal paths with positive values in the input:

        - ``score[i, j] = score[i-1, j-1] + 1``  if ``sim[i, j] > 0``
        - ``score[i, j] = 0`` otherwise.

    The second method, denoted as `S` (SSA equation 4), is similar to the first,
    but allows for "knight moves" (as in the chess piece) in addition to strict
    diagonal moves:

        - ``score[i, j] = max(score[i-1, j-1], score[i-2, j-1], score[i-1, j-2]) + 1``  if ``sim[i, j] >
          0``
        - ``score[i, j] = 0`` otherwise.

    The third method, denoted as `Q` (SSA equations 5 and 6) extends this by
    allowing gaps in the alignment that incur some cost, rather than a hard
    reset to 0 whenever ``sim[i, j] == 0``.
    Gaps are penalized by two additional parameters, ``gap_onset`` and ``gap_extend``,
    which are subtracted from the value of the alignment path every time a gap
    is introduced or extended (respectively).

    Note that setting ``gap_onset`` and ``gap_extend`` to `np.inf` recovers the second
    method, and disabling knight moves recovers the first.

    .. [#] Serrà, Joan, Xavier Serra, and Ralph G. Andrzejak.
        "Cross recurrence quantification for cover song identification."
        New Journal of Physics 11, no. 9 (2009): 093017.

    .. [#] Eckmann, J. P., S. Oliffson Kamphorst, and D. Ruelle.
        "Recurrence plots of dynamical systems."
        World Scientific Series on Nonlinear Science Series A 16 (1995): 441-446.

    Parameters
    ----------
    sim : np.ndarray [shape=(N, M), non-negative]
        The similarity matrix to use as input.

        This can either be a recurrence matrix (self-similarity)
        or a cross-similarity matrix between two sequences.

    gap_onset : float > 0
        Penalty for introducing a gap to an alignment sequence

    gap_extend : float > 0
        Penalty for extending a gap in an alignment sequence

    knight_moves : bool
        If ``True`` (default), allow for "knight moves" in the alignment,
        e.g., ``(n, m) => (n + 1, m + 2)`` or ``(n + 2, m + 1)``.

        If ``False``, only allow for diagonal moves ``(n, m) => (n + 1, m + 1)``.

    backtrack : bool
        If ``True``, return the alignment path.

        If ``False``, only return the score matrix.

    Returns
    -------
    score : np.ndarray [shape=(N, M)]
        The alignment score matrix.  ``score[n, m]`` is the cumulative value of
        the best alignment sequence ending in frames ``n`` and ``m``.
    path : np.ndarray [shape=(k, 2)] (optional)
        If ``backtrack=True``, ``path`` contains a list of pairs of aligned frames
        in the best alignment sequence.

        ``path[i] = [n, m]`` indicates that row ``n`` aligns to column ``m``.

    See Also
    --------
    librosa.segment.recurrence_matrix
    librosa.segment.cross_similarity
    dtw

    Examples
    --------
    Simple diagonal path enhancement (L-mode)

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=30)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> # Use time-delay embedding to reduce noise
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> # Build recurrence, suppress self-loops within 1 second
    >>> rec = librosa.segment.recurrence_matrix(chroma_stack, width=43,
    ...                                         mode='affinity',
    ...                                         metric='cosine')
    >>> # using infinite cost for gaps enforces strict path continuation
    >>> L_score, L_path = librosa.sequence.rqa(rec,
    ...                                        gap_onset=np.inf,
    ...                                        gap_extend=np.inf,
    ...                                        knight_moves=False)
    >>> fig, ax = plt.subplots(ncols=2)
    >>> librosa.display.specshow(rec, x_axis='frames', y_axis='frames', ax=ax[0])
    >>> ax[0].set(title='Recurrence matrix')
    >>> librosa.display.specshow(L_score, x_axis='frames', y_axis='frames', ax=ax[1])
    >>> ax[1].set(title='Alignment score matrix')
    >>> ax[1].plot(L_path[:, 1], L_path[:, 0], label='Optimal path', color='c')
    >>> ax[1].legend()
    >>> ax[1].label_outer()

    Full alignment using gaps and knight moves

    >>> # New gaps cost 5, extending old gaps cost 10 for each step
    >>> score, path = librosa.sequence.rqa(rec, gap_onset=5, gap_extend=10)
    >>> fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(rec, x_axis='frames', y_axis='frames', ax=ax[0])
    >>> ax[0].set(title='Recurrence matrix')
    >>> librosa.display.specshow(score, x_axis='frames', y_axis='frames', ax=ax[1])
    >>> ax[1].set(title='Alignment score matrix')
    >>> ax[1].plot(path[:, 1], path[:, 0], label='Optimal path', color='c')
    >>> ax[1].legend()
    >>> ax[1].label_outer()
    """

    if gap_onset < 0:
        raise ParameterError("gap_onset={} must be strictly positive")
    if gap_extend < 0:
        raise ParameterError("gap_extend={} must be strictly positive")

    score, pointers = __rqa_dp(sim, gap_onset, gap_extend, knight_moves)
    if backtrack:
        path = __rqa_backtrack(score, pointers)
        return score, path

    return score


@jit(nopython=True, cache=True)
def __rqa_dp(sim, gap_onset, gap_extend, knight):  # pragma: no cover
    """RQA dynamic programming implementation"""

    # The output array
    score = np.zeros(sim.shape, dtype=sim.dtype)

    # The backtracking array
    backtrack = np.zeros(sim.shape, dtype=np.int8)

    # These are place-holder arrays to limit the points being considered
    # at each step of the DP
    #
    # If knight moves are enabled, values are indexed according to
    # [(-1,-1), (-1, -2), (-2, -1)]
    #
    # If knight moves are disabled, then only the first entry is used.
    #
    # Using dummy vectors here makes the code a bit cleaner down below.
    sim_values = np.zeros(3)
    score_values = np.zeros(3)
    vec = np.zeros(3)

    if knight:
        # Initial limit is for the base case: diagonal + one knight
        init_limit = 2

        # Otherwise, we have 3 positions
        limit = 3
    else:
        init_limit = 1
        limit = 1

    # backtracking rubric:
    #   0 ==> diagonal move
    #   1 ==> knight move up
    #   2 ==> knight move left
    #  -1 ==> reset without inclusion
    #  -2 ==> reset with inclusion (ie positive value at init)

    # Initialize the first row and column with the data
    score[0, :] = sim[0, :]
    score[:, 0] = sim[:, 0]

    # backtracking initialization: the first row and column are all resets
    # if there's a positive link here, it's an inclusive reset
    for i in range(sim.shape[0]):
        if sim[i, 0]:
            backtrack[i, 0] = -2
        else:
            backtrack[i, 0] = -1

    for j in range(sim.shape[1]):
        if sim[0, j]:
            backtrack[0, j] = -2
        else:
            backtrack[0, j] = -1

    # Initialize the 1-1 case using only the diagonal
    if sim[1, 1] > 0:
        score[1, 1] = score[0, 0] + sim[1, 1]
        backtrack[1, 1] = 0
    else:
        link = sim[0, 0] > 0
        score[1, 1] = max(0, score[0, 0] - (link) * gap_onset - (~link) * gap_extend)
        if score[1, 1] > 0:
            backtrack[1, 1] = 0
        else:
            backtrack[1, 1] = -1

    # Initialize the second row with diagonal and left-knight moves
    i = 1
    for j in range(2, sim.shape[1]):
        score_values[:-1] = (score[i - 1, j - 1], score[i - 1, j - 2])
        sim_values[:-1] = (sim[i - 1, j - 1], sim[i - 1, j - 2])
        t_values = sim_values > 0
        if sim[i, j] > 0:
            backtrack[i, j] = np.argmax(score_values[:init_limit])
            score[i, j] = score_values[backtrack[i, j]] + sim[i, j]  # or + 1 for binary
        else:
            vec[:init_limit] = (
                score_values[:init_limit]
                - t_values[:init_limit] * gap_onset
                - (~t_values[:init_limit]) * gap_extend
            )

            backtrack[i, j] = np.argmax(vec[:init_limit])
            score[i, j] = max(0, vec[backtrack[i, j]])
            # Is it a reset?
            if score[i, j] == 0:
                backtrack[i, j] = -1

    # Initialize the second column with diagonal and up-knight moves
    j = 1
    for i in range(2, sim.shape[0]):
        score_values[:-1] = (score[i - 1, j - 1], score[i - 2, j - 1])
        sim_values[:-1] = (sim[i - 1, j - 1], sim[i - 2, j - 1])
        t_values = sim_values > 0
        if sim[i, j] > 0:
            backtrack[i, j] = np.argmax(score_values[:init_limit])
            score[i, j] = score_values[backtrack[i, j]] + sim[i, j]  # or + 1 for binary

        else:
            vec[:init_limit] = (
                score_values[:init_limit]
                - t_values[:init_limit] * gap_onset
                - (~t_values[:init_limit]) * gap_extend
            )

            backtrack[i, j] = np.argmax(vec[:init_limit])
            score[i, j] = max(0, vec[backtrack[i, j]])
            # Is it a reset?
            if score[i, j] == 0:
                backtrack[i, j] = -1

    # Now fill in the rest of the table
    for i in range(2, sim.shape[0]):
        for j in range(2, sim.shape[1]):
            score_values[:] = (
                score[i - 1, j - 1],
                score[i - 1, j - 2],
                score[i - 2, j - 1],
            )
            sim_values[:] = (sim[i - 1, j - 1], sim[i - 1, j - 2], sim[i - 2, j - 1])
            t_values = sim_values > 0
            if sim[i, j] > 0:
                # if knight is true, it's max of (-1,-1), (-1, -2), (-2, -1)
                # otherwise, it's just the diagonal move (-1, -1)
                # for backtracking purposes, if the max is 0 then it's the start of a new sequence
                # if the max is non-zero, then we extend the existing sequence
                backtrack[i, j] = np.argmax(score_values[:limit])
                score[i, j] = (
                    score_values[backtrack[i, j]] + sim[i, j]
                )  # or + 1 for binary

            else:
                # if the max of our options is negative, then it's a hard reset
                # otherwise, it's a skip move
                vec[:limit] = (
                    score_values[:limit]
                    - t_values[:limit] * gap_onset
                    - (~t_values[:limit]) * gap_extend
                )

                backtrack[i, j] = np.argmax(vec[:limit])
                score[i, j] = max(0, vec[backtrack[i, j]])
                # Is it a reset?
                if score[i, j] == 0:
                    backtrack[i, j] = -1

    return score, backtrack


def __rqa_backtrack(score, pointers):
    """RQA path backtracking

    Given the score matrix and backtracking index array,
    reconstruct the optimal path.
    """

    # backtracking rubric:
    #   0 ==> diagonal move
    #   1 ==> knight move up
    #   2 ==> knight move left
    #  -1 ==> reset (sim = 0)
    #  -2 ==> start of sequence (sim > 0)

    # This array maps the backtracking values to the
    # relative index offsets
    offsets = [(-1, -1), (-1, -2), (-2, -1)]

    # Find the maximum to end the path
    idx = list(np.unravel_index(np.argmax(score), score.shape))

    # Construct the path
    path = []
    while True:
        bt_index = pointers[tuple(idx)]

        # A -1 indicates a non-inclusive reset
        # this can only happen when sim[idx] == 0,
        # and a reset with zero score should not be included
        # in the path.  In this case, we're done.
        if bt_index == -1:
            break

        # Other bt_index values are okay for inclusion
        path.insert(0, idx)

        # -2 indicates beginning of sequence,
        # so we can't backtrack any further
        if bt_index == -2:
            break

        # Otherwise, prepend this index and continue
        idx = [idx[_] + offsets[bt_index][_] for _ in range(len(idx))]

    # If there's no alignment path at all, eg an empty cross-similarity
    # matrix, return a properly shaped and typed array
    if not path:
        return np.empty((0, 2), dtype=np.uint)

    return np.asarray(path, dtype=np.uint)


@jit(nopython=True, cache=True)
def _viterbi(log_prob, log_trans, log_p_init):  # pragma: no cover
    """Core Viterbi algorithm.

    This is intended for internal use only.

    Parameters
    ----------
    log_prob : np.ndarray [shape=(T, m)]
        ``log_prob[t, s]`` is the conditional log-likelihood
        ``log P[X = X(t) | State(t) = s]``
    log_trans : np.ndarray [shape=(m, m)]
        The log transition matrix
        ``log_trans[i, j] = log P[State(t+1) = j | State(t) = i]``
    log_p_init : np.ndarray [shape=(m,)]
        log of the initial state distribution

    Returns
    -------
    None
        All computations are performed in-place on ``state, value, ptr``.
    """
    n_steps, n_states = log_prob.shape

    state = np.zeros(n_steps, dtype=np.uint16)
    value = np.zeros((n_steps, n_states), dtype=np.float64)
    ptr = np.zeros((n_steps, n_states), dtype=np.uint16)

    # factor in initial state distribution
    value[0] = log_prob[0] + log_p_init

    for t in range(1, n_steps):
        # Want V[t, j] <- p[t, j] * max_k V[t-1, k] * A[k, j]
        #    assume at time t-1 we were in state k
        #    transition k -> j

        # Broadcast over rows:
        #    Tout[k, j] = V[t-1, k] * A[k, j]
        #    then take the max over columns
        # We'll do this in log-space for stability

        trans_out = value[t - 1] + log_trans.T

        # Unroll the max/argmax loop to enable numba support
        for j in range(n_states):
            ptr[t, j] = np.argmax(trans_out[j])
            # value[t, j] = log_prob[t, j] + np.max(trans_out[j])
            value[t, j] = log_prob[t, j] + trans_out[j, ptr[t][j]]

    # Now roll backward

    # Get the last state
    state[-1] = np.argmax(value[-1])

    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    logp = value[-1:, state[-1]]

    return state, logp


@deprecate_positional_args
def viterbi(prob, transition, *, p_init=None, return_logp=False):
    """Viterbi decoding from observation likelihoods.

    Given a sequence of observation likelihoods ``prob[s, t]``,
    indicating the conditional likelihood of seeing the observation
    at time ``t`` from state ``s``, and a transition matrix
    ``transition[i, j]`` which encodes the conditional probability of
    moving from state ``i`` to state ``j``, the Viterbi algorithm [#]_ computes
    the most likely sequence of states from the observations.

    .. [#] Viterbi, Andrew. "Error bounds for convolutional codes and an
        asymptotically optimum decoding algorithm."
        IEEE transactions on Information Theory 13.2 (1967): 260-269.

    Parameters
    ----------
    prob : np.ndarray [shape=(..., n_states, n_steps), non-negative]
        ``prob[..., s, t]`` is the probability of observation at time ``t``
        being generated by state ``s``.
    transition : np.ndarray [shape=(n_states, n_states), non-negative]
        ``transition[i, j]`` is the probability of a transition from i->j.
        Each row must sum to 1.
    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, a uniform distribution is assumed.
    return_logp : bool
        If ``True``, return the log-likelihood of the state sequence.

    Returns
    -------
    Either ``states`` or ``(states, logp)``:
    states : np.ndarray [shape=(..., n_steps,)]
        The most likely state sequence.
        If ``prob`` contains multiple channels of input, then each channel is
        decoded independently.
    logp : scalar [float] or np.ndarray
        If ``return_logp=True``, the log probability of ``states`` given
        the observations.

    See Also
    --------
    viterbi_discriminative : Viterbi decoding from state likelihoods

    Examples
    --------
    Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    In this example, we have two states ``healthy`` and ``fever``, with
    initial probabilities 60% and 40%.

    We have three observation possibilities: ``normal``, ``cold``, and
    ``dizzy``, whose probabilities given each state are:

    ``healthy => {normal: 50%, cold: 40%, dizzy: 10%}`` and
    ``fever => {normal: 10%, cold: 30%, dizzy: 60%}``

    Finally, we have transition probabilities:

    ``healthy => healthy (70%)`` and
    ``fever => fever (60%)``.

    Over three days, we observe the sequence ``[normal, cold, dizzy]``,
    and wish to know the maximum likelihood assignment of states for the
    corresponding days, which we compute with the Viterbi algorithm below.

    >>> p_init = np.array([0.6, 0.4])
    >>> p_emit = np.array([[0.5, 0.4, 0.1],
    ...                    [0.1, 0.3, 0.6]])
    >>> p_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> path, logp = librosa.sequence.viterbi(p_emit, p_trans, p_init=p_init,
    ...                                       return_logp=True)
    >>> print(logp, path)
    -4.19173690823075 [0 0 1]
    """

    n_states, n_steps = prob.shape[-2:]

    if transition.shape != (n_states, n_states):
        raise ParameterError(
            "transition.shape={}, must be "
            "(n_states, n_states)={}".format(transition.shape, (n_states, n_states))
        )

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ParameterError(
            "Invalid transition matrix: must be non-negative "
            "and sum to 1 on each row."
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError("Invalid probability values: must be between 0 and 1.")

    # Compute log-likelihoods while avoiding log-underflow
    epsilon = tiny(prob)

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1.0 / n_states)
    elif (
        np.any(p_init < 0)
        or not np.allclose(p_init.sum(), 1)
        or p_init.shape != (n_states,)
    ):
        raise ParameterError(
            "Invalid initial state distribution: " "p_init={}".format(p_init)
        )

    log_trans = np.log(transition + epsilon)
    log_prob = np.log(prob + epsilon)
    log_p_init = np.log(p_init + epsilon)

    def _helper(lp):
        # Transpose input
        _state, logp = _viterbi(lp.T, log_trans, log_p_init)
        # Transpose outputs for return
        return _state.T, logp

    if log_prob.ndim == 2:
        states, logp = _helper(log_prob)
    else:
        # Vectorize the helper
        __viterbi = np.vectorize(
            _helper, otypes=[np.uint16, np.float64], signature="(s,t)->(t),(1)"
        )

        states, logp = __viterbi(log_prob)

        # Flatten out the trailing dimension introduced by vectorization
        logp = logp[..., 0]

    if return_logp:
        return states, logp

    return states


@deprecate_positional_args
def viterbi_discriminative(
    prob, transition, *, p_state=None, p_init=None, return_logp=False
):
    """Viterbi decoding from discriminative state predictions.

    Given a sequence of conditional state predictions ``prob[s, t]``,
    indicating the conditional likelihood of state ``s`` given the
    observation at time ``t``, and a transition matrix ``transition[i, j]``
    which encodes the conditional probability of moving from state ``i``
    to state ``j``, the Viterbi algorithm computes the most likely sequence
    of states from the observations.

    This implementation uses the standard Viterbi decoding algorithm
    for observation likelihood sequences, under the assumption that
    ``P[Obs(t) | State(t) = s]`` is proportional to
    ``P[State(t) = s | Obs(t)] / P[State(t) = s]``, where the denominator
    is the marginal probability of state ``s`` occurring as given by ``p_state``.

    Parameters
    ----------
    prob : np.ndarray [shape=(..., n_states, n_steps), non-negative]
        ``prob[s, t]`` is the probability of state ``s`` conditional on
        the observation at time ``t``.
        Must be non-negative and sum to 1 along each column.
    transition : np.ndarray [shape=(n_states, n_states), non-negative]
        ``transition[i, j]`` is the probability of a transition from i->j.
        Each row must sum to 1.
    p_state : np.ndarray [shape=(n_states,)]
        Optional: marginal probability distribution over states,
        must be non-negative and sum to 1.
        If not provided, a uniform distribution is assumed.
    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, it is assumed to be uniform.
    return_logp : bool
        If ``True``, return the log-likelihood of the state sequence.

    Returns
    -------
    Either ``states`` or ``(states, logp)``:
    states : np.ndarray [shape=(..., n_steps,)]
        The most likely state sequence.
        If ``prob`` contains multiple input channels,
        then each channel is decoded independently.
    logp : scalar [float] or np.ndarray
        If ``return_logp=True``, the log probability of ``states`` given
        the observations.

    See Also
    --------
    viterbi :
        Viterbi decoding from observation likelihoods
    viterbi_binary :
        Viterbi decoding for multi-label, conditional state likelihoods

    Examples
    --------
    This example constructs a simple, template-based discriminative chord estimator,
    using CENS chroma as input features.

    .. note:: this chord model is not accurate enough to use in practice. It is only
            intended to demonstrate how to use discriminative Viterbi decoding.

    >>> # Create templates for major, minor, and no-chord qualities
    >>> maj_template = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0])
    >>> min_template = np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0])
    >>> N_template   = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1.]) / 4.
    >>> # Generate the weighting matrix that maps chroma to labels
    >>> weights = np.zeros((25, 12), dtype=float)
    >>> labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj',
    ...           'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
    ...           'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min',
    ...           'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
    ...           'N']
    >>> for c in range(12):
    ...     weights[c, :] = np.roll(maj_template, c) # c:maj
    ...     weights[c + 12, :] = np.roll(min_template, c)  # c:min
    >>> weights[-1] = N_template  # the last row is the no-chord class
    >>> # Make a self-loop transition matrix over 25 states
    >>> trans = librosa.sequence.transition_loop(25, 0.9)

    >>> # Load in audio and make features
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
    >>> # Suppress percussive elements
    >>> y = librosa.effects.harmonic(y, margin=4)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> # Map chroma (observations) to class (state) likelihoods
    >>> probs = np.exp(weights.dot(chroma))  # P[class | chroma] ~= exp(template' chroma)
    >>> probs /= probs.sum(axis=0, keepdims=True)  # probabilities must sum to 1 in each column
    >>> # Compute independent frame-wise estimates
    >>> chords_ind = np.argmax(probs, axis=0)
    >>> # And viterbi estimates
    >>> chords_vit = librosa.sequence.viterbi_discriminative(probs, trans)

    >>> # Plot the features and prediction map
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2)
    >>> librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', ax=ax[0])
    >>> librosa.display.specshow(weights, x_axis='chroma', ax=ax[1])
    >>> ax[1].set(yticks=np.arange(25) + 0.5, yticklabels=labels, ylabel='Chord')

    >>> # And plot the results
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(probs, x_axis='time', cmap='gray', ax=ax)
    >>> times = librosa.times_like(chords_vit)
    >>> ax.scatter(times, chords_ind + 0.25, color='lime', alpha=0.5, marker='+',
    ...            s=15, label='Independent')
    >>> ax.scatter(times, chords_vit - 0.25, color='deeppink', alpha=0.5, marker='o',
    ...            s=15, label='Viterbi')
    >>> ax.set(yticks=np.unique(chords_vit),
    ...        yticklabels=[labels[i] for i in np.unique(chords_vit)])
    >>> ax.legend()
    """

    n_states, n_steps = prob.shape[-2:]

    if transition.shape != (n_states, n_states):
        raise ParameterError(
            "transition.shape={}, must be "
            "(n_states, n_states)={}".format(transition.shape, (n_states, n_states))
        )

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ParameterError(
            "Invalid transition matrix: must be non-negative "
            "and sum to 1 on each row."
        )

    if np.any(prob < 0) or not np.allclose(prob.sum(axis=-2), 1):
        raise ParameterError(
            "Invalid probability values: each column must "
            "sum to 1 and be non-negative"
        )

    # Compute log-likelihoods while avoiding log-underflow
    epsilon = tiny(prob)

    # Compute marginal log probabilities while avoiding underflow
    if p_state is None:
        p_state = np.empty(n_states)
        p_state.fill(1.0 / n_states)
    elif p_state.shape != (n_states,):
        raise ParameterError(
            "Marginal distribution p_state must have shape (n_states,). "
            "Got p_state.shape={}".format(p_state.shape)
        )
    elif np.any(p_state < 0) or not np.allclose(p_state.sum(axis=-1), 1):
        raise ParameterError(
            "Invalid marginal state distribution: " "p_state={}".format(p_state)
        )

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1.0 / n_states)
    elif (
        np.any(p_init < 0)
        or not np.allclose(p_init.sum(), 1)
        or p_init.shape != (n_states,)
    ):
        raise ParameterError(
            "Invalid initial state distribution: " "p_init={}".format(p_init)
        )

    # By Bayes' rule, P[X | Y] * P[Y] = P[Y | X] * P[X]
    # P[X] is constant for the sake of maximum likelihood inference
    # and P[Y] is given by the marginal distribution p_state.
    #
    # So we have P[X | y] \propto P[Y | x] / P[Y]
    # if X = observation and Y = states, this can be done in log space as
    # log P[X | y] \propto \log P[Y | x] - \log P[Y]
    log_p_init = np.log(p_init + epsilon)
    log_trans = np.log(transition + epsilon)
    log_marginal = np.log(p_state + epsilon)

    # reshape to broadcast against prob
    log_marginal = expand_to(log_marginal, ndim=prob.ndim, axes=-2)

    log_prob = np.log(prob + epsilon) - log_marginal

    def _helper(lp):
        # Transpose input
        _state, logp = _viterbi(lp.T, log_trans, log_p_init)
        # Transpose outputs for return
        return _state.T, logp

    if log_prob.ndim == 2:
        states, logp = _helper(log_prob)
    else:
        # Vectorize the helper
        __viterbi = np.vectorize(
            _helper, otypes=[np.uint16, np.float64], signature="(s,t)->(t),(1)"
        )

        states, logp = __viterbi(log_prob)

    # Flatten out the trailing dimension
    logp = logp[..., 0]

    if return_logp:
        return states, logp

    return states


@deprecate_positional_args
def viterbi_binary(prob, transition, *, p_state=None, p_init=None, return_logp=False):
    """Viterbi decoding from binary (multi-label), discriminative state predictions.

    Given a sequence of conditional state predictions ``prob[s, t]``,
    indicating the conditional likelihood of state ``s`` being active
    conditional on observation at time ``t``, and a 2*2 transition matrix
    ``transition`` which encodes the conditional probability of moving from
    state ``s`` to state ``~s`` (not-``s``), the Viterbi algorithm computes the
    most likely sequence of states from the observations.

    This function differs from `viterbi_discriminative` in that it does not assume the
    states to be mutually exclusive.  `viterbi_binary` is implemented by
    transforming the multi-label decoding problem to a collection
    of binary Viterbi problems (one for each *state* or label).

    The output is a binary matrix ``states[s, t]`` indicating whether each
    state ``s`` is active at time ``t``.

    Parameters
    ----------
    prob : np.ndarray [shape=(..., n_steps,) or (..., n_states, n_steps)], non-negative
        ``prob[s, t]`` is the probability of state ``s`` being active
        conditional on the observation at time ``t``.
        Must be non-negative and less than 1.

        If ``prob`` is 1-dimensional, it is expanded to shape ``(1, n_steps)``.

        If ``prob`` contains multiple input channels, then each channel is decoded independently.

    transition : np.ndarray [shape=(2, 2) or (n_states, 2, 2)], non-negative
        If 2-dimensional, the same transition matrix is applied to each sub-problem.
        ``transition[0, i]`` is the probability of the state going from inactive to ``i``,
        ``transition[1, i]`` is the probability of the state going from active to ``i``.
        Each row must sum to 1.

        If 3-dimensional, ``transition[s]`` is interpreted as the 2x2 transition matrix
        for state label ``s``.

    p_state : np.ndarray [shape=(n_states,)]
        Optional: marginal probability for each state (between [0,1]).
        If not provided, a uniform distribution (0.5 for each state)
        is assumed.

    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, it is assumed to be uniform.

    return_logp : bool
        If ``True``, return the log-likelihood of the state sequence.

    Returns
    -------
    Either ``states`` or ``(states, logp)``:
    states : np.ndarray [shape=(..., n_states, n_steps)]
        The most likely state sequence.
    logp : np.ndarray [shape=(..., n_states,)]
        If ``return_logp=True``, the log probability of each state activation
        sequence ``states``

    See Also
    --------
    viterbi :
        Viterbi decoding from observation likelihoods
    viterbi_discriminative :
        Viterbi decoding for discriminative (mutually exclusive) state predictions

    Examples
    --------
    In this example, we have a sequence of binary state likelihoods that we want to de-noise
    under the assumption that state changes are relatively uncommon.  Positive predictions
    should only be retained if they persist for multiple steps, and any transient predictions
    should be considered as errors.  This use case arises frequently in problems such as
    instrument recognition, where state activations tend to be stable over time, but subject
    to abrupt changes (e.g., when an instrument joins the mix).

    We assume that the 0 state has a self-transition probability of 90%, and the 1 state
    has a self-transition probability of 70%.  We assume the marginal and initial
    probability of either state is 50%.

    >>> trans = np.array([[0.9, 0.1], [0.3, 0.7]])
    >>> prob = np.array([0.1, 0.7, 0.4, 0.3, 0.8, 0.9, 0.8, 0.2, 0.6, 0.3])
    >>> librosa.sequence.viterbi_binary(prob, trans, p_state=0.5, p_init=0.5)
    array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]])
    """

    prob = np.atleast_2d(prob)

    n_states, n_steps = prob.shape[-2:]

    if transition.shape == (2, 2):
        transition = np.tile(transition, (n_states, 1, 1))
    elif transition.shape != (n_states, 2, 2):
        raise ParameterError(
            "transition.shape={}, must be (2, 2) or "
            "(n_states, 2, 2)={}".format(transition.shape, (n_states))
        )

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=-1), 1):
        raise ParameterError(
            "Invalid transition matrix: must be non-negative "
            "and sum to 1 on each row."
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError("Invalid probability values: prob must be between [0, 1]")

    if p_state is None:
        p_state = np.empty(n_states)
        p_state.fill(0.5)
    else:
        p_state = np.atleast_1d(p_state)

    if p_state.shape != (n_states,) or np.any(p_state < 0) or np.any(p_state > 1):
        raise ParameterError(
            "Invalid marginal state distributions: p_state={}".format(p_state)
        )

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(0.5)
    else:
        p_init = np.atleast_1d(p_init)

    if p_init.shape != (n_states,) or np.any(p_init < 0) or np.any(p_init > 1):
        raise ParameterError(
            "Invalid initial state distributions: p_init={}".format(p_init)
        )

    shape_prefix = list(prob.shape[:-2])
    states = np.empty(shape_prefix + [n_states, n_steps], dtype=np.uint16)
    logp = np.empty(shape_prefix + [n_states])

    prob_binary = np.empty(shape_prefix + [2, n_steps])
    p_state_binary = np.empty(2)
    p_init_binary = np.empty(2)

    for state in range(n_states):
        prob_binary[..., 0, :] = 1 - prob[..., state, :]
        prob_binary[..., 1, :] = prob[..., state, :]

        p_state_binary[0] = 1 - p_state[state]
        p_state_binary[1] = p_state[state]

        p_init_binary[0] = 1 - p_init[state]
        p_init_binary[1] = p_init[state]

        states[..., state, :], logp[..., state] = viterbi_discriminative(
            prob_binary,
            transition[state],
            p_state=p_state_binary,
            p_init=p_init_binary,
            return_logp=True,
        )

    if return_logp:
        return states, logp

    return states


def transition_uniform(n_states):
    """Construct a uniform transition matrix over ``n_states``.

    Parameters
    ----------
    n_states : int > 0
        The number of states

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        ``transition[i, j] = 1./n_states``

    Examples
    --------
    >>> librosa.sequence.transition_uniform(3)
    array([[0.333, 0.333, 0.333],
           [0.333, 0.333, 0.333],
           [0.333, 0.333, 0.333]])
    """

    if not isinstance(n_states, (int, np.integer)) or n_states <= 0:
        raise ParameterError("n_states={} must be a positive integer")

    transition = np.empty((n_states, n_states), dtype=np.float64)
    transition.fill(1.0 / n_states)
    return transition


def transition_loop(n_states, prob):
    """Construct a self-loop transition matrix over ``n_states``.

    The transition matrix will have the following properties:

        - ``transition[i, i] = p`` for all ``i``
        - ``transition[i, j] = (1 - p) / (n_states - 1)`` for all ``j != i``

    This type of transition matrix is appropriate when states tend to be
    locally stable, and there is no additional structure between different
    states.  This is primarily useful for de-noising frame-wise predictions.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    prob : float in [0, 1] or iterable, length=n_states
        If a scalar, this is the probability of a self-transition.

        If a vector of length ``n_states``, ``p[i]`` is the probability of self-transition in state ``i``

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    >>> librosa.sequence.transition_loop(3, 0.5)
    array([[0.5 , 0.25, 0.25],
           [0.25, 0.5 , 0.25],
           [0.25, 0.25, 0.5 ]])

    >>> librosa.sequence.transition_loop(3, [0.8, 0.5, 0.25])
    array([[0.8  , 0.1  , 0.1  ],
           [0.25 , 0.5  , 0.25 ],
           [0.375, 0.375, 0.25 ]])
    """

    if not isinstance(n_states, (int, np.integer)) or n_states <= 1:
        raise ParameterError("n_states={} must be a positive integer > 1")

    transition = np.empty((n_states, n_states), dtype=np.float64)

    # if it's a float, make it a vector
    prob = np.asarray(prob, dtype=np.float64)

    if prob.ndim == 0:
        prob = np.tile(prob, n_states)

    if prob.shape != (n_states,):
        raise ParameterError(
            "prob={} must have length equal to n_states={}".format(prob, n_states)
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError(
            "prob={} must have values in the range [0, 1]".format(prob)
        )

    for i, prob_i in enumerate(prob):
        transition[i] = (1.0 - prob_i) / (n_states - 1)
        transition[i, i] = prob_i

    return transition


def transition_cycle(n_states, prob):
    """Construct a cyclic transition matrix over ``n_states``.

    The transition matrix will have the following properties:

        - ``transition[i, i] = p``
        - ``transition[i, i + 1] = (1 - p)``

    This type of transition matrix is appropriate for state spaces
    with cyclical structure, such as metrical position within a bar.
    For example, a song in 4/4 time has state transitions of the form

        1->{1, 2}, 2->{2, 3}, 3->{3, 4}, 4->{4, 1}.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    prob : float in [0, 1] or iterable, length=n_states
        If a scalar, this is the probability of a self-transition.

        If a vector of length ``n_states``, ``p[i]`` is the probability of
        self-transition in state ``i``

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    >>> librosa.sequence.transition_cycle(4, 0.9)
    array([[0.9, 0.1, 0. , 0. ],
           [0. , 0.9, 0.1, 0. ],
           [0. , 0. , 0.9, 0.1],
           [0.1, 0. , 0. , 0.9]])
    """

    if not isinstance(n_states, (int, np.integer)) or n_states <= 1:
        raise ParameterError("n_states={} must be a positive integer > 1")

    transition = np.zeros((n_states, n_states), dtype=np.float64)

    # if it's a float, make it a vector
    prob = np.asarray(prob, dtype=np.float64)

    if prob.ndim == 0:
        prob = np.tile(prob, n_states)

    if prob.shape != (n_states,):
        raise ParameterError(
            "prob={} must have length equal to n_states={}".format(prob, n_states)
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError(
            "prob={} must have values in the range [0, 1]".format(prob)
        )

    for i, prob_i in enumerate(prob):
        transition[i, np.mod(i + 1, n_states)] = 1.0 - prob_i
        transition[i, i] = prob_i

    return transition


@deprecate_positional_args
def transition_local(n_states, width, *, window="triangle", wrap=False):
    """Construct a localized transition matrix.

    The transition matrix will have the following properties:

        - ``transition[i, j] = 0`` if ``|i - j| > width``
        - ``transition[i, i]`` is maximal
        - ``transition[i, i - width//2 : i + width//2]`` has shape ``window``

    This type of transition matrix is appropriate for state spaces
    that discretely approximate continuous variables, such as in fundamental
    frequency estimation.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    width : int >= 1 or iterable
        The maximum number of states to treat as "local".
        If iterable, it should have length equal to ``n_states``,
        and specify the width independently for each state.

    window : str, callable, or window specification
        The window function to determine the shape of the "local" distribution.

        Any window specification supported by `filters.get_window` will work here.

        .. note:: Certain windows (e.g., 'hann') are identically 0 at the boundaries,
            so and effectively have ``width-2`` non-zero values.  You may have to expand
            ``width`` to get the desired behavior.

    wrap : bool
        If ``True``, then state locality ``|i - j|`` is computed modulo ``n_states``.
        If ``False`` (default), then locality is absolute.

    See Also
    --------
    librosa.filters.get_window

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    Triangular distributions with and without wrapping

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=False)
    array([[0.667, 0.333, 0.   , 0.   , 0.   ],
           [0.25 , 0.5  , 0.25 , 0.   , 0.   ],
           [0.   , 0.25 , 0.5  , 0.25 , 0.   ],
           [0.   , 0.   , 0.25 , 0.5  , 0.25 ],
           [0.   , 0.   , 0.   , 0.333, 0.667]])

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=True)
    array([[0.5 , 0.25, 0.  , 0.  , 0.25],
           [0.25, 0.5 , 0.25, 0.  , 0.  ],
           [0.  , 0.25, 0.5 , 0.25, 0.  ],
           [0.  , 0.  , 0.25, 0.5 , 0.25],
           [0.25, 0.  , 0.  , 0.25, 0.5 ]])

    Uniform local distributions with variable widths and no wrapping

    >>> librosa.sequence.transition_local(5, [1, 2, 3, 3, 1], window='ones', wrap=False)
    array([[1.   , 0.   , 0.   , 0.   , 0.   ],
           [0.5  , 0.5  , 0.   , 0.   , 0.   ],
           [0.   , 0.333, 0.333, 0.333, 0.   ],
           [0.   , 0.   , 0.333, 0.333, 0.333],
           [0.   , 0.   , 0.   , 0.   , 1.   ]])
    """

    if not isinstance(n_states, (int, np.integer)) or n_states <= 1:
        raise ParameterError("n_states={} must be a positive integer > 1")

    width = np.asarray(width, dtype=int)
    if width.ndim == 0:
        width = np.tile(width, n_states)

    if width.shape != (n_states,):
        raise ParameterError(
            "width={} must have length equal to n_states={}".format(width, n_states)
        )

    if np.any(width < 1):
        raise ParameterError("width={} must be at least 1")

    transition = np.zeros((n_states, n_states), dtype=np.float64)

    # Fill in the widths.  This is inefficient, but simple
    for i, width_i in enumerate(width):
        trans_row = pad_center(
            get_window(window, width_i, fftbins=False), size=n_states
        )
        trans_row = np.roll(trans_row, n_states // 2 + i + 1)

        if not wrap:
            # Knock out the off-diagonal-band elements
            trans_row[min(n_states, i + width_i // 2 + 1) :] = 0
            trans_row[: max(0, i - width_i // 2)] = 0

        transition[i] = trans_row

    # Row-normalize
    transition /= transition.sum(axis=1, keepdims=True)

    return transition
