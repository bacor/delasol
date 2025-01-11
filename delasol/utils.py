# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

import networkx as nx
import numpy as np
import matplotlib.cm as cm
from collections.abc import Iterable
from music21.pitch import Pitch
from music21.stream import Stream
from music21.note import Note

# Local imports
from delasol.custom_types import PitchLike


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


def as_pitch_list(pitch_string: str, sep: str = " ") -> list[Pitch]:
    """Convert a string of pitches into a list of Pitch objects.

    Parameters
    ----------
    pitch_string : str
        A string containing pitch values separated by a specified separator.
    sep : str, optional
        The separator used to split the pitch_string into individual pitches.
        Default is a space (" ").

    Returns
    -------
    list[Pitch]
        A list of Pitch objects created from the input string.

    Examples
    --------
    >>> as_pitch_list("C4 D4 E4 F4")
    [<music21.pitch.Pitch C4>, <music21.pitch.Pitch D4>, <music21.pitch.Pitch E4>, <music21.pitch.Pitch F4>]
    """
    return [Pitch(p) for p in pitch_string.split(sep)]


def as_stream(pitch_string: str, sep: str = " ") -> Stream:
    """Convert a string of pitch representations into a Stream of Note objects.

    Parameters
    ----------
    pitch_string : str
        A string containing pitch representations, separated by the specified
        separator.
    sep : str, optional
        The separator used to split the pitch_string into individual pitches.
        Default is a space (" ").

    Returns
    -------
    Stream
        A Stream object containing Note objects created from the pitches in the
        input string.

    Examples
    --------
    >>> stream = as_stream("C4 D4 E4 F4")
    >>> stream[0]
    <music21.note.Note C>
    >>> stream[1]
    <music21.note.Note D>
    """
    pitches = as_pitch_list(pitch_string, sep=sep)
    notes = [Note(p) for p in pitches]
    return Stream(notes)


def as_pitch(pitch: PitchLike) -> Pitch:
    """Convert a given pitch representation to a Pitch object.

    Parameters
    ----------
    pitch : PitchLike
        The input pitch representation, which can be a string or a Pitch
        object.

    Returns
    -------
    Pitch
        A Pitch object corresponding to the input representation.

    Raises
    ------
    ValueError
        If the input is neither a string nor a Pitch object.

    Examples
    --------
    >>> as_pitch("A4")
    <music21.pitch.Pitch A4>

    >>> as_pitch(Pitch("C#5"))
    <music21.pitch.Pitch C#5>
    >>> as_pitch(1)
    Traceback (most recent call last):
        ...
    ValueError: Expected a Pitch object or string, got <class 'int'>
    """
    if isinstance(pitch, str):
        pitch = Pitch(pitch)
    elif not isinstance(pitch, Pitch):
        raise ValueError(f"Expected a Pitch object or string, got {type(pitch)}")
    return pitch


def extract_lyrics(notes: Iterable[Note], number: int) -> list[str]:
    """Extract the lyrics at a given line number from an iterable of notes"""
    extracted = []
    for note in notes:
        lyrics = {lyric.number: lyric for lyric in note.lyrics}
        if number in lyrics:
            extracted.append(lyrics[number].text)
        else:
            extracted.append(None)
    return extracted


def num_lyrics(stream: Stream) -> int:
    """Returns the maximum number of lyrics in a stream"""
    num_lyrics = 0
    for note in stream.flat.notes:
        numbers = [lyric.number for lyric in note.lyrics]
        if len(numbers) > 0:
            num_lyrics = max(max(numbers), num_lyrics)
    return num_lyrics


def annotate_note(
    note: Note, text: str = None, color: str = None, number: int = 1
) -> None:
    """Add lyrics to a note and set its color"""
    if text is not None:
        note.addLyric(text, lyricNumber=number)
    lyrics = {lyric.number: lyric for lyric in note.lyrics}
    if color is not None:
        if number in lyrics:
            lyrics[number].style.color = color


def set_lyrics_color(
    notes: Iterable[Note], number: int, color: str = "#000000"
) -> None:
    """Set the color of a lyric in a list of notes."""
    for note in notes:
        annotate_note(note, color=color, number=number)


def find_first_difference(sequence, value, offset: int = 0):
    """Find the index of the first element in the sequence that is different
    from the value.

    >>> find_first_difference([1, 1, 2, 1], 1)
    2
    """
    for i in range(offset, len(sequence) - 1):
        if sequence[i] != value:
            return i
    return None


def find_first_repeat(sequence, value, offset: int = 0):
    """Find the index where a given value is first repeated in the sequence.

    >>> find_first_repeat([1, 2, 1, 1, 1, 1], 1)
    2
    """
    for i in range(offset, len(sequence) - 1):
        if sequence[i] == value and sequence[i + 1] == value:
            return i
    return None


def segment_deviations(sequence, value):
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


def latexify_subscript(input: str, as_text: bool = True) -> str:
    """Convert a string to a LaTeX formatted subscript representation.

    Parameters
    ----------
    input : str
        The input string which may contain a subscript indicated by an
        underscore ('_'). If the underscore is present, the part before the
        underscore is treated as the main text, and the part after is treated
        as the subscript.
    as_text : bool, optional
        If True, the function returns a LaTeX formatted string with the text
        and subscript wrapped in a text environment. Default is True.

    Returns
    -------
    str
        A LaTeX formatted string.

    Examples
    --------
    >>> latexify_subscript("C4")
    'C4'
    >>> latexify_subscript("C_4")
    '$\\\\text{C}_{\\\\text{4}}$'
    >>> latexify_subscript("C_4", as_text=False)
    '$C_{4}$'
    """
    if "_" in input:
        name, subscript = input.split("_")
        if as_text:
            return f"$\\text{{{name}}}_{{\\text{{{subscript}}}}}$"
        else:
            return f"${name}_{{{subscript}}}$"
    else:
        return input


def draw_graph(
    graph: nx.Graph,
    labels: t.Union[
        t.Literal["latex_name", "name", "syllable", "index"], str, dict[t.Any, str]
    ] = "latex_name",
    pos: dict[t.Any, (float, float)] = None,
    weights: t.Iterable[float] = None,
    show_loops: bool = False,
    ax: "matplotlib.axis.Axes" = None,
    color_mapper: t.Callable[[float], t.Any] = lambda w: cm.Reds(0.9 * w + 0.1),
    label_kws={},
    edge_kws={},
    **shared_kws,
):
    """Draw a graph using NetworkX and Matplotlib.

    Parameters
    ----------
    graph : nx.Graph
        The graph to be drawn.

    labels : {str, dict, optional}
        The labels to use for the nodes:
        - str: The attribute of the nodes to use as labels. The attributes
          "name", "syllable", and "index" are predefined, but you can use
          custom attributes. A ValueError is raised if the the attribute
          specified does not exist.
        - dict: A dictionary mapping nodes to labels.
        Default is "name".

    pos : dict, optional
        A dictionary mapping nodes to their positions in the plot. If None,
        positions will be retrieved from the graph's node attributes.

    weights : iterable of float, optional
        A collection of weights for the edges. If None, weights will be
        extracted from the graph's edges.

    show_loops : bool, optional
        If True, self-loops will be included in the drawing. Default is False.

    ax : matplotlib.axes.Axes, optional
        The axes to draw the graph on. If None, a new figure will be created.

    label_kws : dict, optional
        Additional keyword arguments for node label styling.

    edge_kws : dict, optional
        Additional keyword arguments for edge styling.

    color_mapper : callable, optional
        A function that maps edge weights to colors. Default is a function
        that scales weights to a red color gradient.

    Returns
    -------
    None
        The function draws the graph directly and does not return any value.
    """
    if pos is None:
        pos = nx.get_node_attributes(graph, "position")

    if labels == "latex_name":
        names = nx.get_node_attributes(graph, "name")
        labels = {node: latexify_subscript(label) for node, label in names.items()}
    elif isinstance(labels, str):
        labels = nx.get_node_attributes(graph, labels)
        if len(labels) == 0:
            raise ValueError(f"No labels found for attribute {labels}")

    kws = dict(
        font_size=9,
        bbox=dict(
            facecolor="white",
            linewidth=0.5,
            boxstyle="round,pad=.5,rounding_size=1",
        ),
    )
    kws.update(**label_kws)
    nx.draw_networkx_labels(graph, labels=labels, pos=pos, ax=ax, **kws, **shared_kws)

    # Edges
    if show_loops:
        edges = graph.edges
    else:
        edges = [e for e in graph.edges if e[0] != e[1]]
    if weights is None:
        weights = np.array([graph.edges[e]["weight"] for e in edges])
        weights = weights / weights.max()
    kws = dict(
        width=1,
        min_target_margin=10,
        arrowsize=7,
        node_size=500,
        connectionstyle="arc3,rad=-.2",
    )
    kws.update(**edge_kws)
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=edges,
        edge_color=[color_mapper(w) for w in weights],
        ax=ax,
        **kws,
        **shared_kws,
    )
