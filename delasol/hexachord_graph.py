# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
from functools import cached_property

# Libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from music21.pitch import Pitch

# Local imports
from delasol.custom_types import PitchLike, HexachordGraphNode
from delasol.utils import draw_graph, as_pitch, dict_swap

# Conventional hexachords numbering
octaves = range(0, 10)
base_notes = ["C", "F", "G"]
G2_index = len(base_notes) * (2 - octaves[0]) + base_notes.index("G")

HEXACHORD_NUMBERING = {
    Pitch(f"{note}{octave}"): i - G2_index + 1
    for i, (note, octave) in enumerate(
        (note, octave) for octave in octaves for note in base_notes
    )
}
"""
dict[Pitch, int]: Conventional hexachords numbering so that F2 is 0, G2 is 1, 
etc. The numbering is extended for down and up beyond the conventional 
numbering so that e.g. C2 is -1 and C5 is 8.
"""

SYLLABLES = ["ut", "re", "mi", "fa", "sol", "la", "fi"]
"""
list of str: The syllables used to internally name the nodes in a hexachord graph.
"""


class HexachordGraph(nx.DiGraph):
    """
    A directed graph representing a hexachord.

    Nodes in the graph are Pitch objects and have the following attributes:
    - index (int): the zero-based index of the pitch in the hexachord (e.g. ut
      is 0, fa-super-la is 6)
    - name (string): the (globally) unique name of the node (e.g. "ut_C3", "re_F4", etc.)
    - syllable (string): the syllable of the hexachord (e.g. "ut", "re", etc.)

    Attributes
    ----------
    base : Pitch
        The base pitch of the hexachord.
    fa_super_la : bool
        Indicates whether the hexachord includes the fa super la.
    pitches : list of Pitch
        The list of pitches in the hexachord.

    Parameters
    ----------
    base : PitchLike
        The base of the hexachord: the lowest note in the hexachord or the ut.
    fa_super_la : bool, optional
        A flag indicating whether to include the fa super la in the hexachord.
        If so, it becomes a 'heptachordal hexachord', otherwise it is a normal
        hexachord. Default is True.
    **kwargs : keyword arguments
        Additional keyword arguments to be passed to the build method.

    Notes
    -----
    This class builds a sequence of pitches based on the specified base pitch
    and the defined intervals. The intervals are applied in the order: major
    second, major second, minor second, major second, major second, and
    optionally a minor second if `fa_super_la` is True.

    Examples
    --------
    >>> hex = HexachordGraph("G2")
    >>> hex.number
    1
    >>> hex.quality
    'hard'
    >>> hex.names["re_G2"]
    <music21.pitch.Pitch A2>
    >>> hex.syllables["fi"]
    <music21.pitch.Pitch F3>
    >>> hex.pitches
    [<music21.pitch.Pitch G2>, <music21.pitch.Pitch A2>, <music21.pitch.Pitch B2>, <music21.pitch.Pitch C3>, <music21.pitch.Pitch D3>, <music21.pitch.Pitch E3>, <music21.pitch.Pitch F3>]
    """

    def __init__(
        self,
        base: PitchLike,
        fa_super_la: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._names = None
        self.fa_super_la = fa_super_la
        self.base = as_pitch(base)

        # Build the graph
        intervals = ["M2", "M2", "m2", "M2", "M2"]
        if fa_super_la:
            intervals.append("m2")
        # TODO is it necessary to store the pitches? They are the same as the nodes after all
        self.pitches = [self.base]
        for interval in intervals:
            self.pitches.append(self.pitches[-1].transpose(interval))
        self.build(**kwargs)

    def __repr__(self):
        return f"<HexachordGraph on {self.base.nameWithOctave}>"

    @property
    def number(self) -> int | None:
        """The number of a hexachord, if the hexachord starts on C, F, or G.

        Conventionally those hexachords are numbered, from 1 for the hexachord
        on G2 (Gamma ut) up to 7 for the hexachord on G4. Lower and higher
        hexachords simply continue the numbering, so the hexachord on F2 is
        number 0, and the hexachord on C5 is number 8. If the hexachord has
        a different base, None is returned.
        """
        return HEXACHORD_NUMBERING.get(self.base, None)

    @property
    def quality(self) -> str | None:
        """The quality (natural, soft or hard) of the hexachord.

        The quality is "natural" for hexachords on Cs, "soft" for those on Fs,
        "hard" for those on Gs, and None otherwise.

        Examples
        --------
        >>> HexachordGraph("G2").quality
        'hard'
        >>> HexachordGraph("C3").quality
        'natural'
        >>> HexachordGraph("F3").quality
        'soft'
        >>> HexachordGraph("B-2").quality is None
        True
        """
        qualities = dict(C="natural", F="soft", G="hard")
        return qualities.get(self.base.name, None)

    @cached_property
    def names(self) -> dict[str, HexachordGraphNode]:
        """A dictionary mapping node names to the corresponding nodes.

        Returns
        -------
        dict of str: HexachordGraphNode
            A dictionary where the keys are the names of the nodes and the
            values are the corresponding graph nodes (Pitch objects).

        Examples
        --------
        >>> hex = HexachordGraph("F3")
        >>> hex.names
        {'ut_F3': <music21.pitch.Pitch F3>, 're_F3': <music21.pitch.Pitch G3>, 'mi_F3': <music21.pitch.Pitch A3>, 'fa_F3': <music21.pitch.Pitch B-3>, 'sol_F3': <music21.pitch.Pitch C4>, 'la_F3': <music21.pitch.Pitch D4>, 'fi_F3': <music21.pitch.Pitch E-4>}
        """
        return dict_swap(nx.get_node_attributes(self, "name"))

    @cached_property
    def syllables(self) -> dict[str, HexachordGraphNode]:
        """A dictionary mapping syllables to their corresponding nodes.

        Returns
        -------
        dict of str: HexachordGraphNode
            A dictionary where the keys are the syllables of the nodes and the
            values are the corresponding graph nodes (Pitch objects).

        Examples
        --------
        >>> hex = HexachordGraph("G2")
        >>> hex.syllables
        {'ut': <music21.pitch.Pitch G2>, 're': <music21.pitch.Pitch A2>, 'mi': <music21.pitch.Pitch B2>, 'fa': <music21.pitch.Pitch C3>, 'sol': <music21.pitch.Pitch D3>, 'la': <music21.pitch.Pitch E3>, 'fi': <music21.pitch.Pitch F3>}
        """
        return dict_swap(nx.get_node_attributes(self, "syllable"))

    def build(
        self,
        step_weight: float = 1,
        loop_weight: float = 0.5,
        fa_super_la_weight: float = 1.5,
        weights: np.ndarray = None,
    ):
        """Build a weighted hexachord graph with the specified weights.

        Nodes in the graph are Pitch objects, and have attributes containing their index, name, and position. See class documentation for details.

        Parameters
        ----------
        step_weight : float, optional
            The weight assigned to step transitions between nodes. Default is 1.
            Ignored if `weights` is provided.
        loop_weight : float, optional
            The weight assigned to loop transitions (self-loops) for each node.
            Default is 0.5. Ignored if `weights` is provided.
        fa_super_la_weight : float, optional
            The weight assigned to the transition between the la and fa super la.
            Default is 1.5. Ignored if `weights` is provided.
        weights : np.ndarray, optional
            A pre-defined weight matrix for the edges. It should be a 7x7 matrix,
            so that the element at position (i, j) in the matrix represents the
            weight of the edge from node i to node j. A weight of 0 means there
            is no edge between the nodes. If None, a weight matrix is generated
            based on the other parameters.
        """
        if weights is None:
            N = len(self.pitches)
            weights = np.eye(N) * loop_weight
            np.fill_diagonal(weights[1:], step_weight)
            np.fill_diagonal(weights.T[1:], step_weight)
            weights[N - 1, N - 2] = fa_super_la_weight
            weights[N - 2, N - 1] = fa_super_la_weight

        for i, pitch in enumerate(self.pitches):
            self.add_node(
                pitch,
                name=f"{SYLLABLES[i]}_{self.base.nameWithOctave}",
                syllable=SYLLABLES[i],
                index=i,
            )

        for i, pitch1 in enumerate(self.pitches):
            for j, pitch2 in enumerate(self.pitches):
                if weights[i, j] > 0:
                    self.add_edge(pitch1, pitch2, weight=weights[i, j])

    def positions(
        self, y: float = 0, offset_x: float = 0
    ) -> dict[HexachordGraphNode, tuple[float, float]]:
        """Calculate the positions of nodes in a hexachord graph.

        Parameters
        ----------
        y : float, optional
            The y-coordinate for the positions of the nodes. Default is 0.
        offset_x : float, optional
            The x-coordinate offset to be added to the diatonic note number of
            each node. Default is 0.

        Returns
        -------
        dict[HexachordGraphNode, tuple[float, float]]
            A dictionary mapping each node to its corresponding (x, y) position
            as a tuple, where x is the diatonic note number plus an optional
            offset and y is the provided parameter.
        """
        return {node: (offset_x + node.diatonicNoteNum, y) for node in self.nodes}

    def draw(
        self,
        ax: "matplotlib.axes.Axes" = None,
        styling: bool = True,
        pos_kws={},
        **kws,
    ) -> None:
        """Draws a graphical representation of the hexachord graph.

        Parameters
        ----------
        fig : plt.Figure, optional
            A matplotlib figure object to draw on. If None, a new figure will
            be created. Default is None.
        pos_kws : dict, optional
            Additional keyword arguments passed to the positions method. Default is {}.
        **kws : keyword arguments
            Additional keyword arguments passed to the drawing function.

        Returns
        -------
        None
            This function does not return a value. It modifies the current
            matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(len(self), 1))
        if "pos" not in kws:
            kws["pos"] = self.positions(**pos_kws)

        draw_graph(self, ax=ax, **kws)

        if styling:
            ax.axis("off")
            ys = np.array([y for _, y in kws["pos"].values()])
            ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)


# Ensure that doctest also evaluates these cached properties
__test__ = {
    "HexachordGraph.names": HexachordGraph.names,
    "HexachordGraph.syllables": HexachordGraph.syllables,
}
