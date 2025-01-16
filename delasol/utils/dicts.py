# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------


def dict_first(d: dict):
    """Retrieve the value associated with the first key in a dictionary.

    Parameters
    ----------
    d : dict
        The input dictionary from which to retrieve the first value.

    Returns
    -------
    The value associated with the first key in the dictionary.

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
    d : dict
        A dictionary from which to retrieve the value of the last key.

    Returns
    -------
    The value associated with the last key in the dictionary.

    Notes
    -----
    If the dictionary is empty, this function will raise a
    `StopIteration` error due to the use of `next()` on an empty iterator.

    Examples
    --------
    >>> dict_last({'a': 1, 'b': 2})
    2
    """
    last_key = next(reversed(d.keys()))
    return d[last_key]


def dict_swap(d: dict):
    """Swap the keys and values of a dictionary.

    Parameters
    ----------
    d : dict
        A dictionary where the values are unique and hashable.

    Returns
    -------
    dict
        A new dictionary with keys and values swapped. The original dictionary
        remains unchanged.

    Examples
    --------
    >>> dict_swap({'a': 1, 'b': 2})
    {1: 'a', 2: 'b'}
    """
    return {v: k for k, v in d.items()}
