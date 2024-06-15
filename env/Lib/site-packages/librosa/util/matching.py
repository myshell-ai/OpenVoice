#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Matching functions"""

import numpy as np
import numba

from .exceptions import ParameterError
from .utils import valid_intervals

__all__ = ["match_intervals", "match_events"]


@numba.jit(nopython=True, cache=True)
def __jaccard(int_a, int_b):  # pragma: no cover
    """Jaccard similarity between two intervals

    Parameters
    ----------
    int_a, int_b : np.ndarrays, shape=(2,)

    Returns
    -------
    Jaccard similarity between intervals
    """
    ends = [int_a[1], int_b[1]]
    if ends[1] < ends[0]:
        ends.reverse()

    starts = [int_a[0], int_b[0]]
    if starts[1] < starts[0]:
        starts.reverse()

    intersection = ends[0] - starts[1]
    if intersection < 0:
        intersection = 0.0

    union = ends[1] - starts[0]

    if union > 0:
        return intersection / union

    return 0.0


@numba.jit(nopython=True, cache=True)
def __match_interval_overlaps(query, intervals_to, candidates):  # pragma: no cover
    """Find the best Jaccard match from query to candidates"""

    best_score = -1
    best_idx = -1
    for idx in candidates:
        score = __jaccard(query, intervals_to[idx])

        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx


@numba.jit(nopython=True, cache=True)
def __match_intervals(intervals_from, intervals_to, strict=True):  # pragma: no cover
    """Numba-accelerated interval matching algorithm."""
    # sort index of the interval starts
    start_index = np.argsort(intervals_to[:, 0])

    # sort index of the interval ends
    end_index = np.argsort(intervals_to[:, 1])

    # and sorted values of starts
    start_sorted = intervals_to[start_index, 0]
    # and ends
    end_sorted = intervals_to[end_index, 1]

    search_ends = np.searchsorted(start_sorted, intervals_from[:, 1], side="right")
    search_starts = np.searchsorted(end_sorted, intervals_from[:, 0], side="left")

    output = np.empty(len(intervals_from), dtype=numba.uint32)
    for i in range(len(intervals_from)):
        query = intervals_from[i]

        # Find the intervals that start after our query ends
        after_query = search_ends[i]
        # And the intervals that end after our query begins
        before_query = search_starts[i]

        # Candidates for overlapping have to (end after we start) and (begin before we end)
        candidates = set(start_index[:after_query]) & set(end_index[before_query:])

        # Proceed as before
        if len(candidates) > 0:
            output[i] = __match_interval_overlaps(query, intervals_to, candidates)
        elif strict:
            # Numba only lets us use compile-time constants in exception messages
            raise ParameterError
        else:
            # Find the closest interval
            # (start_index[after_query] - query[1]) is the distance to the next interval
            # (query[0] - end_index[before_query])
            dist_before = np.inf
            dist_after = np.inf
            if search_starts[i] > 0:
                dist_before = query[0] - end_sorted[search_starts[i] - 1]
            if search_ends[i] + 1 < len(intervals_to):
                dist_after = start_sorted[search_ends[i] + 1] - query[1]
            if dist_before < dist_after:
                output[i] = end_index[search_starts[i] - 1]
            else:
                output[i] = start_index[search_ends[i] + 1]
    return output


def match_intervals(intervals_from, intervals_to, strict=True):
    """Match one set of time intervals to another.

    This can be useful for tasks such as mapping beat timings
    to segments.

    Each element ``[a, b]`` of ``intervals_from`` is matched to the
    element ``[c, d]`` of ``intervals_to`` which maximizes the
    Jaccard similarity between the intervals::

        max(0, |min(b, d) - max(a, c)|) / |max(d, b) - min(a, c)|

    In ``strict=True`` mode, if there is no interval with positive
    intersection with ``[a,b]``, an exception is thrown.

    In ``strict=False`` mode, any interval ``[a, b]`` that has no
    intersection with any element of ``intervals_to`` is instead
    matched to the interval ``[c, d]`` which minimizes::

        min(|b - c|, |a - d|)

    that is, the disjoint interval [c, d] with a boundary closest
    to [a, b].

    .. note:: An element of ``intervals_to`` may be matched to multiple
       entries of ``intervals_from``.

    Parameters
    ----------
    intervals_from : np.ndarray [shape=(n, 2)]
        The time range for source intervals.
        The ``i`` th interval spans time ``intervals_from[i, 0]``
        to ``intervals_from[i, 1]``.
        ``intervals_from[0, 0]`` should be 0, ``intervals_from[-1, 1]``
        should be the track duration.
    intervals_to : np.ndarray [shape=(m, 2)]
        Analogous to ``intervals_from``.
    strict : bool
        If ``True``, intervals can only match if they intersect.
        If ``False``, disjoint intervals can match.

    Returns
    -------
    interval_mapping : np.ndarray [shape=(n,)]
        For each interval in ``intervals_from``, the
        corresponding interval in ``intervals_to``.

    See Also
    --------
    match_events

    Raises
    ------
    ParameterError
        If either array of input intervals is not the correct shape

        If ``strict=True`` and some element of ``intervals_from`` is disjoint from
        every element of ``intervals_to``.

    Examples
    --------
    >>> ints_from = np.array([[3, 5], [1, 4], [4, 5]])
    >>> ints_to = np.array([[0, 2], [1, 3], [4, 5], [6, 7]])
    >>> librosa.util.match_intervals(ints_from, ints_to)
    array([2, 1, 2], dtype=uint32)
    >>> # [3, 5] => [4, 5]  (ints_to[2])
    >>> # [1, 4] => [1, 3]  (ints_to[1])
    >>> # [4, 5] => [4, 5]  (ints_to[2])

    The reverse matching of the above is not possible in ``strict`` mode
    because ``[6, 7]`` is disjoint from all intervals in ``ints_from``.
    With ``strict=False``, we get the following:

    >>> librosa.util.match_intervals(ints_to, ints_from, strict=False)
    array([1, 1, 2, 2], dtype=uint32)
    >>> # [0, 2] => [1, 4]  (ints_from[1])
    >>> # [1, 3] => [1, 4]  (ints_from[1])
    >>> # [4, 5] => [4, 5]  (ints_from[2])
    >>> # [6, 7] => [4, 5]  (ints_from[2])
    """

    if len(intervals_from) == 0 or len(intervals_to) == 0:
        raise ParameterError("Attempting to match empty interval list")

    # Verify that the input intervals has correct shape and size
    valid_intervals(intervals_from)
    valid_intervals(intervals_to)

    try:
        return __match_intervals(intervals_from, intervals_to, strict=strict)
    except ParameterError as exc:
        raise ParameterError(
            "Unable to match intervals with strict={}".format(strict)
        ) from exc


def match_events(events_from, events_to, left=True, right=True):
    """Match one set of events to another.

    This is useful for tasks such as matching beats to the nearest
    detected onsets, or frame-aligned events to the nearest zero-crossing.

    .. note:: A target event may be matched to multiple source events.

    Examples
    --------
    >>> # Sources are multiples of 7
    >>> s_from = np.arange(0, 100, 7)
    >>> s_from
    array([ 0,  7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91,
           98])
    >>> # Targets are multiples of 10
    >>> s_to = np.arange(0, 100, 10)
    >>> s_to
    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> # Find the matching
    >>> idx = librosa.util.match_events(s_from, s_to)
    >>> idx
    array([0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9])
    >>> # Print each source value to its matching target
    >>> zip(s_from, s_to[idx])
    [(0, 0), (7, 10), (14, 10), (21, 20), (28, 30), (35, 30),
     (42, 40), (49, 50), (56, 60), (63, 60), (70, 70), (77, 80),
     (84, 80), (91, 90), (98, 90)]

    Parameters
    ----------
    events_from : ndarray [shape=(n,)]
        Array of events (eg, times, sample or frame indices) to match from.
    events_to : ndarray [shape=(m,)]
        Array of events (eg, times, sample or frame indices) to
        match against.
    left : bool
    right : bool
        If ``False``, then matched events cannot be to the left (or right)
        of source events.

    Returns
    -------
    event_mapping : np.ndarray [shape=(n,)]
        For each event in ``events_from``, the corresponding event
        index in ``events_to``::

            event_mapping[i] == arg min |events_from[i] - events_to[:]|

    See Also
    --------
    match_intervals

    Raises
    ------
    ParameterError
        If either array of input events is not the correct shape
    """
    if len(events_from) == 0 or len(events_to) == 0:
        raise ParameterError("Attempting to match empty event list")

    # If we can't match left or right, then only strict equivalence
    # counts as a match.
    if not (left or right) and not np.all(np.in1d(events_from, events_to)):
        raise ParameterError(
            "Cannot match events with left=right=False "
            "and events_from is not contained "
            "in events_to"
        )

    # If we can't match to the left, then there should be at least one
    # target event greater-equal to every source event
    if (not left) and max(events_to) < max(events_from):
        raise ParameterError(
            "Cannot match events with left=False "
            "and max(events_to) < max(events_from)"
        )

    # If we can't match to the right, then there should be at least one
    # target event less-equal to every source event
    if (not right) and min(events_to) > min(events_from):
        raise ParameterError(
            "Cannot match events with right=False "
            "and min(events_to) > min(events_from)"
        )

    # array of matched items
    output = np.empty_like(events_from, dtype=np.int32)

    return __match_events_helper(output, events_from, events_to, left, right)


@numba.jit(nopython=True, cache=True)
def __match_events_helper(
    output, events_from, events_to, left=True, right=True
):  # pragma: no cover
    # mock dictionary for events
    from_idx = np.argsort(events_from)
    sorted_from = events_from[from_idx]

    to_idx = np.argsort(events_to)
    sorted_to = events_to[to_idx]

    # find the matching indices
    matching_indices = np.searchsorted(sorted_to, sorted_from)

    # iterate over indices in matching_indices
    for ind, middle_ind in enumerate(matching_indices):
        left_flag = False
        right_flag = False

        left_ind = -1
        right_ind = len(matching_indices)

        left_diff = 0
        right_diff = 0
        mid_diff = 0

        middle_ind = matching_indices[ind]
        sorted_from_num = sorted_from[ind]

        # Prevent oob from chosen index
        if middle_ind == len(sorted_to):
            middle_ind -= 1

        # Permitted to look to the left
        if left and middle_ind > 0:
            left_ind = middle_ind - 1
            left_flag = True

        # Permitted to look to right
        if right and middle_ind < len(sorted_to) - 1:
            right_ind = middle_ind + 1
            right_flag = True

        mid_diff = abs(sorted_to[middle_ind] - sorted_from_num)
        if left and left_flag:
            left_diff = abs(sorted_to[left_ind] - sorted_from_num)
        if right and right_flag:
            right_diff = abs(sorted_to[right_ind] - sorted_from_num)

        if left_flag and (
            not right
            and (sorted_to[middle_ind] > sorted_from_num)
            or (not right_flag and left_diff < mid_diff)
            or (left_diff < right_diff and left_diff < mid_diff)
        ):
            output[ind] = to_idx[left_ind]

        # Check if right should be chosen
        elif right_flag and (right_diff < mid_diff):
            output[ind] = to_idx[right_ind]

        # Selected index wins
        else:
            output[ind] = to_idx[middle_ind]

    # Undo sorting
    solutions = np.empty_like(output)
    solutions[from_idx] = output

    return solutions
