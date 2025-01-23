# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------


def dict_first(d: dict):
    """Retrieve the value associated with the first key in a dictionary.

    Parameters
    ----------
    d
        The input dictionary from which to retrieve the first value.

    Raises
    ------
    StopIteration
        If the dictionary is empty, an exception is raised when trying to
        retrieve the first key.

    Examples
    --------
    >>> dict_first({'a': 1, 'b': 2})
    1
    """
    first_key = next(iter(d.keys()))
    return d[first_key]


def dict_last(d: dict):
    """Retrieve the value associated with the last key in a dictionary.

    Parameters
    ----------
    d
        A dictionary from which to retrieve the value of the last key.

    Raises
    ------
    StopIteration
        If the dictionary is empty, an exception is raised when trying to
        retrieve the next key.

    Examples
    --------
    >>> dict_last({'a': 1, 'b': 2})
    2
    """
    last_key = next(reversed(d.keys()))
    return d[last_key]


def dict_swap(d: dict):
    """Return a new dictionary in which they keys and values are swapped.

    Parameters
    ----------
    d
        A dictionary where the values are unique and hashable.

    Examples
    --------
    >>> dict_swap({'a': 1, 'b': 2})
    {1: 'a', 2: 'b'}
    """
    return {v: k for k, v in d.items()}
