# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t


def find_first_difference(sequence: t.Iterable, value, offset: int = 0):
    """Find the index of the first element in the sequence that is different
    from the value.

    >>> find_first_difference([1, 1, 2, 1], 1)
    2
    """
    for i in range(offset, len(sequence) - 1):
        if sequence[i] != value:
            return i
    return None


def find_first_repeat(sequence: t.Iterable, value, offset: int = 0):
    """Find the index where a given value is first repeated in the sequence.

    >>> find_first_repeat([1, 2, 1, 1, 1, 1], 1)
    2
    """
    for i in range(offset, len(sequence) - 1):
        if sequence[i] == value and sequence[i + 1] == value:
            return i
    return None


def segment_deviations(sequence: t.Iterable, value):
    """Segment a sequence into parts that are constant and parts that are not.
    If possible, the deviating parts are surrounded by constant values. The
    function returns a sequence of tuples `(first, last)` indicating the index
    of the first and last element of the segment.

    In this example, the sequence `[1, 2, 1, 1, 1, 1]` is divided into two
    segments: `[1, 2, 1]` and `[1, 1, 1]`:

    >>> segment_deviations([1, 2, 1, 1, 1, 1], 1)
    [(0, 2), (3, 5)]
    """
    start = 0
    segments = []
    while start < len(sequence):
        diff = find_first_difference(sequence[start:], value)
        if diff is None:
            segments.append((start, len(sequence) - 1))
            break
        else:
            end = find_first_repeat(sequence[start:], value, offset=max(0, diff - 2))
            if end is None:
                segments.append((start, len(sequence) - 1))
                break
            else:
                if start == 0:
                    segment = (0, end)
                else:
                    segment = (start, start + end)
                segments.append(segment)
                start += end + 1

    return segments
