# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from functools import cached_property

# Libraries
import networkx as nx
import matplotlib.pyplot as plt
from music21.pitch import Pitch

# Local imports
from delasol.utils import draw_graph, dict_first, dict_last, dict_swap
from delasol.hexachord_graph import HexachordGraph
from delasol.custom_types import GamutGraphNode, PitchLike


class GamutGraph(nx.DiGraph):
    """
    A graph representing a gamut of hexachords.

    Parameters
    ----------
    hexachords : Iterable[Union[HexachordGraph, PitchLike]], optional
        An iterable of hexachords, that can be specified by a HexachordGraph
        or a PitchLike object representing the base of the hexachord.
        If None, the instance will start with an empty set of hexachords.
    mutations : dict, optional
        A dictionary of mutations to apply to the hexachords. If None, no
        mutations will be applied.
    mutation_weight : float, optional
        The weight to apply to the mutations. Default is 2.

    Attributes
    ----------
    hexachords : dict[Pitch, HexachordGraph]
        A dictionary mapping base pitches to the corresponding hexachords.

    Notes
    -----
    This constructor calls the superclass's initializer and populates the
    hexachords and mutations if provided.

    Examples
    --------
    >>> H1 = HexachordGraph("G2")
    >>> H2 = HexachordGraph("C3")
    >>> GamutGraph(hexachords=[H1, H2])
    <GamutGraph hexachords=[G2, C3]>

    Or simply

    >>> GamutGraph(["G2", "C3"])
    <GamutGraph hexachords=[G2, C3]>

    """

    _cached_properties = ["names", "pitches", "overlapping_hexachords"]

    def __init__(
        self,
        hexachords: t.Iterable[t.Union[HexachordGraph, PitchLike]],
        mutations: dict = None,
        mutation_weight: float = 2,
    ):
        super().__init__()
        self.hexachords = dict()
        if hexachords is not None:
            for hexachord in hexachords:
                self.add_hexachord(hexachord)
        if mutations:
            self.add_mutations(mutations, weight=mutation_weight)

    def __repr__(self):
        bases = [base.nameWithOctave for base in self.hexachords.keys()]
        if len(bases) > 7:
            bases_str = ", ".join(bases[:7])
            return f"<GamutGraph hexachords=[{bases_str[:7]}, ...]>"
        else:
            bases_str = ", ".join(bases)
            return f"<GamutGraph hexachords=[{bases_str}]>"

    @cached_property
    def names(self) -> dict[str, GamutGraphNode]:
        """A dictionary mapping node names to the actual nodes in the graph.

        Returns
        -------
        dict[str, GamutGraphNode]
            A dictionary where the keys are node names (strings) and
            the values are the corresponding GamutGraphNode objects.

        Examples
        --------
        >>> gamut = GamutGraph(["G2", "C3"])
        >>> gamut.names["ut_G2"]
        (<music21.pitch.Pitch G2>, <music21.pitch.Pitch G2>)
        >>> gamut.names["fi_C3"]
        (<music21.pitch.Pitch C3>, <music21.pitch.Pitch B-3>)
        """
        return dict_swap(nx.get_node_attributes(self, "name"))

    @cached_property
    def pitches(self) -> dict[Pitch, list[GamutGraphNode]]:
        """A dictionary mapping pitches to the corresponding nodes in the graph.

        Returns
        -------
        dict[Pitch, list[GamutGraphNode]]
            A dictionary where the keys are unique Pitch objects and the values
            are lists of GamutGraphNode instances associated with each pitch.

        Examples
        --------
        >>> gamut = GamutGraph(["G2", "C3"])
        >>> gamut.pitches[Pitch("C3")]
        [(<music21.pitch.Pitch G2>, <music21.pitch.Pitch C3>), (<music21.pitch.Pitch C3>, <music21.pitch.Pitch C3>)]
        """
        pitches = {}
        for base, pitch in self.nodes:
            if pitch not in pitches:
                pitches[pitch] = []
            pitches[pitch].append((base, pitch))
        return pitches

    @property
    def first_hexachord(self) -> HexachordGraph:
        """The first registered hexachord according to its base pitch.

        Returns
        -------
        HexachordGraph
            The first hexachord in the gamut.

        Examples
        --------
        >>> gamut = GamutGraph(["C3", "G2"])
        >>> gamut.first_hexachord.base
        <music21.pitch.Pitch G2>
        """
        return dict_first(self.hexachords)

    @property
    def last_hexachord(self) -> HexachordGraph:
        """The last registered hexachord according to its base pitch.

        Returns
        -------
        HexachordGraph
            The last hexachord in the gamut.

        Examples
        --------
        >>> gamut = GamutGraph(["G3", "F3"])
        >>> gamut.last_hexachord.base
        <music21.pitch.Pitch G3>
        """
        return dict_last(self.hexachords)

    @property
    def lowest_node(self) -> GamutGraphNode:
        """The lowest node of the graph: the lowest pitch of the first
        hexachord, where hexachords are sorted according to their base pitch.

        Returns
        -------
        GamutGraphNode
            The lowest node of the graph
        """
        hex = self.first_hexachord
        return (hex.base, hex.pitches[0])

    @property
    def highest_node(self) -> GamutGraphNode:
        """The highest node of the graph: the highest pitch of the last
        hexachord, where hexachords are sorted according to their base pitch.

        Returns
        -------
        GamutGraphNode
            The lowest node of the graph
        """
        hex = self.last_hexachord
        return (hex.base, hex.pitches[-1])

    @cached_property
    def overlapping_hexachords(self) -> dict[HexachordGraph, list[HexachordGraph]]:
        """Dictionary identifying overlapping hexachords.

        This method identifies hexachords that share common pitches and returns a
        dictionary where each key is a hexachord and the corresponding value is a
        list of hexachords that overlap with it.

        Returns
        -------
        dict[HexachordGraph, list[HexachordGraph]]
            A dictionary mapping each hexachord to a list of hexachords that
            share pitches with it. If the overlapping hexachords have already been
            computed, the cached result is returned.

        Examples
        >>> gamut = GamutGraph(["G2", "C3", "C4"])
        >>> gamut.overlapping_hexachords
        {<HexachordGraph on G2>: [<HexachordGraph on C3>], <HexachordGraph on C3>: [<HexachordGraph on G2>], <HexachordGraph on C4>: []}
        """
        hexachords = list(self.hexachords.values())
        neighbors = {hex: [] for hex in hexachords}
        for i in range(len(hexachords)):
            for j in range(i + 1, len(hexachords)):
                hex1 = hexachords[i]
                hex2 = hexachords[j]
                if any(p in hex1.pitches for p in hex2.pitches):
                    neighbors[hex1].append(hex2)
                    neighbors[hex2].append(hex1)
        return neighbors

    def add_hexachord(self, hexachord: t.Union[HexachordGraph, PitchLike]) -> None:
        if isinstance(hexachord, str) or isinstance(hexachord, Pitch):
            hexachord = HexachordGraph(hexachord)
        elif not isinstance(hexachord, HexachordGraph):
            raise ValueError("Invalid value for hexachord")

        base: Pitch = hexachord.base
        if base in self.hexachords:
            raise ValueError(
                f"A hexachord on {base.nameWithOctave} has already been added to the gamut graph"
            )

        # Register the hexachords, while ensuring they are ordered by their bases
        sorted_hexachords = sorted(
            [hexachord, *self.hexachords.values()], key=lambda h: h.base
        )
        self.hexachords = dict((hex.base, hex) for hex in sorted_hexachords)

        # Add nodes to the graph, copying all attributes from the hexachord graphs
        for node in hexachord.nodes:
            attrs = dict(**hexachord.nodes[node])
            self.add_node((base, node), **attrs)
        weights = nx.get_edge_attributes(hexachord, "weight")
        weighted_edges = [
            ((base, u), (base, v), weight) for (u, v), weight in weights.items()
        ]
        self.add_weighted_edges_from(weighted_edges)

    def add_mutations(self, mutations: dict, weight: float = 2):
        """
        Add mutations between hexachords to the graph using a mutation dictionary.

        The dictionary describes the degrees at which you can mutate from each
        type of hexachord to each other type, in both ascending and descending
        direction. The dictionary should have the following structure:

        ```
        {
            "natural": {
                "up": {"hard": [moves]},
                "down": {"hard": [moves]}
            },
            "hard": {
                "up": {"natural": [moves]},
                "down": {"natural": [moves]}
            }
            # ...
        }
        ```

        Parameters
        ----------
        mutations : dict
            A dictionary defining the mutation rules between hexachords.
        weight : float, optional
            The weight to assign to the edges created by mutations (default is 2).

        Returns
        -------
        None
            This function modifies the graph in place by adding edges based on
            the specified mutations.
        """
        for hexachord in self.hexachords.values():
            for neighbor in self.overlapping_hexachords[hexachord]:
                direction = "up" if neighbor.base > hexachord.base else "down"
                moves = mutations[hexachord.quality][direction].get(
                    neighbor.quality, []
                )
                for move in moves:
                    source_node = (hexachord.base, hexachord.pitches[move[0] - 1])
                    target_node = (neighbor.base, neighbor.pitches[move[1] - 1])
                    edge_weight = weight if len(move) == 2 else move[2]
                    self.add_edge(source_node, target_node, weight=edge_weight)

    def add_edges_by_names(
        self,
        edges: list[tuple[GamutGraphNode, GamutGraphNode, t.Optional[float]]],
        default_weight: float = 1,
    ):
        """Add edges using names of nodes instead of nodes themselves.

        Parameters
        ----------
        edges : list of tuple
            A list of tuples representing the edges to be added. Each tuple can
            either be of the form (source, target) or (source, target, weight).
        default_weight : float, optional
            The default weight to assign to the edges if not specified in the
            tuples. Default is 1.

        Examples
        --------
        >>> G = GamutGraph(["G2", "C3"])
        >>> G.add_edges_by_names([("fa_G2", "re_C3"), ("fa_C3", "la_G2", 1.5)])
        >>> (G.names["fa_G2"], G.names["re_C3"]) in G.edges
        True
        """
        for edge in edges:
            edge_weight = default_weight if len(edge) == 2 else edge[2]
            self.add_edge(self.names[edge[0]], self.names[edge[1]], weight=edge_weight)

    # def solmize(self, node: GamutGraphNode):
    #     """Return the solmization of a pitch in the gamut graph"""
    #     return self.nodes[node]["syllable"]

    def positions(
        self,
        pos_x: t.Literal["order", "diatonic", "ps"] = "diatonic",
        pos_y: t.Literal["order", "diatonic", "ps"] = "order",
        offset_x: float = 0,
        offset_y: float = 0,
    ) -> dict[GamutGraphNode, tuple[float, float]]:
        """Calculate the positions of nodes in the gamut graph.

        Parameters
        ----------
        pos_x : {'order', 'diatonic', 'ps'}, optional
            The method to determine the x-coordinate of the position.
            'order' uses the index of the pitch in the hexachord,
            'diatonic' uses the diatonic note number, and 'ps' uses
            the pitch class. Default is 'diatonic'.

        pos_y : {'order', 'diatonic', 'ps'}, optional
            The method to determine the y-coordinate of the position.
            'order' uses the index of the hexachord, 'diatonic' uses
            the diatonic note number of the base pitch, and 'ps' uses
            the pitch class of the base pitch. Default is 'order'.

        offset_x : float, optional
            An offset to be added to the x-coordinate. Default is 0.

        offset_y : float, optional
            An offset to be added to the y-coordinate. Default is 0.

        Returns
        -------
        dict[GamutGraphNode, tuple[float, float]]
            A dictionary mapping each (base, pitch) pair to its
            corresponding (x, y) position as a tuple of floats.

        Raises
        ------
        ValueError
            If an invalid value is provided for pos_x or pos_y.
        """
        positions = {}
        pitches = list(self.pitches.keys())
        for i, (base, hexachord) in enumerate(self.hexachords.items()):
            for j, pitch in enumerate(hexachord.pitches):
                match pos_x:
                    case "order":
                        x = pitches.index(pitch)
                    case "diatonic":
                        x = pitch.diatonicNoteNum
                    case "ps":
                        x = pitch.ps
                    case _:
                        raise ValueError(f"Invalid value for pos_x: {pos_x}")

                match pos_y:
                    case "order":
                        y = i
                    case "diatonic":
                        y = base.diatonicNoteNum
                    case "ps":
                        y = base.ps
                    case _:
                        raise ValueError(f"Invalid value for pos_y: {pos_y}")

                positions[(base, pitch)] = (x + offset_x, y + offset_y)
        return positions

    def draw(
        self,
        show_axes: bool = True,
        ax: "matplotlib.axes.Axes" = None,
        pos_x: t.Literal["order", "diatonic", "ps"] = "diatonic",
        pos_y: t.Literal["order", "diatonic", "ps"] = "order",
        pos_kws={},
        **kws,
    ):
        """Draw a graphical representation of hexachords.

        Parameters
        ----------
        show_axes : bool, optional
            If True, display the axes. Default is True.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the graph. If None, a new figure and axes
            will be created.
        pos_x : {'order', 'diatonic', 'ps'}, optional
            The positioning method for the x-axis. Default is 'diatonic'.
        pos_y : {'order', 'diatonic', 'ps'}, optional
            The positioning method for the y-axis. Default is 'order'.
        pos_kws : dict, optional
            Additional keyword arguments for positioning.
        **kws : keyword arguments
            Additional keyword arguments passed to the drawing function.

        Returns
        -------
        None
            This function does not return a value but modifies the provided axes
            to display the hexachord graph.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(len(self) * 0.4, len(self.hexachords)))

        # Determine positions
        if "pos" not in kws:
            _ = pos_kws.pop("pos_x", None)
            _ = pos_kws.pop("pos_y", None)
            kws["pos"] = self.positions(pos_x=pos_x, pos_y=pos_y, **pos_kws)

        # Draw graph!
        draw_graph(self, ax=ax, **kws)

        # Decorate with nice axes
        if show_axes:
            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axis_on()
            ax.xaxis.grid(color=".9")
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

            # X-ticks
            pitches = self.pitches.keys()
            lowest = self.lowest_node[1]
            highest = self.highest_node[1]
            match pos_x:
                case "diatonic":
                    ax.set_xticks([p.diatonicNoteNum for p in pitches])
                    ax.set_xlim(lowest.diatonicNoteNum - 1, highest.diatonicNoteNum + 1)
                case "ps":
                    ax.set_xticks([p.ps for p in pitches])
                    ax.set_xlim(lowest.ps - 1, highest.ps + 1)
                case "order":
                    ax.set_xticks(list(range(len(pitches))))
                    ax.set_xlim(-1, len(pitches))

            xtick_labels = [
                p.unicodeNameWithOctave if p.name in "CEG" else None
                for p in self.pitches.keys()
            ]
            ax.set_xticklabels(xtick_labels)
            ax.set_xlabel("pitch")

            # Y-ticks
            bases = self.hexachords.keys()
            match pos_y:
                case "diatonic":
                    ax.set_yticks([p.diatonicNoteNum for p in bases])
                case "ps":
                    ax.set_yticks([p.ps for p in bases])
                case _:
                    ax.set_yticks(list(range(len(bases))))

            ytick_labels = [base.unicodeNameWithOctave for base in bases]
            ax.set_yticklabels(ytick_labels)
            ax.set_ylabel("base of hexachord")


# Ensure that doctest also evaluates these cached properties
__test__ = {
    "GamutGraph.pitches": GamutGraph.pitches,
    "GamutGraph.names": GamutGraph.names,
    "GamutGraph.overlapping_hexachords": GamutGraph.overlapping_hexachords,
}
