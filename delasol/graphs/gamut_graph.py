# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from functools import cached_property

# Library imports
import networkx as nx
from music21.pitch import Pitch

# Local imports
from delasol.utils.dicts import dict_first, dict_last, dict_swap
from delasol.graphs.hexachord_graph import HexachordGraph
from delasol.custom_types import GamutGraphNode, PitchLike


class GamutGraph(nx.DiGraph):
    """
    A graph representing a gamut of hexachords.

    Examples
    --------
    >>> H1 = HexachordGraph("G2")
    >>> H2 = HexachordGraph("C3")
    >>> GamutGraph(hexachords=[H1, H2])
    <GamutGraph hexachords=[G2, C3]>

    Or simply

    >>> GamutGraph(["G2", "C3"])
    <GamutGraph hexachords=[G2, C3]>

    Parameters
    ----------
    hexachords
        An iterable of hexachords, that can be specified by a HexachordGraph
        or a PitchLike object representing the base of the hexachord.
        If None, the instance will start with an empty set of hexachords.
    mutations
        A dictionary of mutations to apply to the hexachords. If None, no
        mutations will be applied.
    mutation_weight
        The weight to apply to the mutations. Default is 2.


    Attributes
    ----------
    hexachords : dict[Pitch, HexachordGraph]
        A dictionary mapping base pitches to the corresponding hexachords.
    """

    _cached_properties = ["name_to_node", "pitch_to_node", "overlapping_hexachords"]

    def __init__(
        self,
        hexachords: t.Iterable[t.Union[HexachordGraph, PitchLike]],
        mutations: list[dict] = None,
        mutation_weight: float = 2,
    ):
        super().__init__()
        self.hexachords = dict()
        if hexachords is not None:
            for hexachord in hexachords:
                self.add_hexachord(hexachord)
        if mutations:
            self.add_mutations(mutations, default_weight=mutation_weight)

    def __repr__(self):
        bases = [base.nameWithOctave for base in self.hexachords.keys()]
        if len(bases) > 7:
            bases_str = ", ".join(bases[:7])
            return f"<GamutGraph hexachords=[{bases_str[:7]}, ...]>"
        else:
            bases_str = ", ".join(bases)
            return f"<GamutGraph hexachords=[{bases_str}]>"

    # Properties

    @cached_property
    def name_to_node(self) -> dict[str, GamutGraphNode]:
        """A dictionary in which the keys are node names (strings) and
        the values are the corresponding GamutGraphNode objects.

        Examples
        --------
        >>> gamut = GamutGraph(["G2", "C3"])
        >>> gamut.name_to_node["ut_G2"]
        (<music21.pitch.Pitch G2>, <music21.pitch.Pitch G2>)
        >>> gamut.name_to_node["fi_C3"]
        (<music21.pitch.Pitch C3>, <music21.pitch.Pitch B-3>)
        """
        return dict_swap(nx.get_node_attributes(self, "name"))

    @cached_property
    def pitch_to_node(self) -> dict[Pitch, list[GamutGraphNode]]:
        """A dictionary in which the keys are unique Pitch objects and the values
        are lists of GamutGraphNode instances associated with each pitch.

        Examples
        --------
        >>> gamut = GamutGraph(["G2", "C3"])
        >>> gamut.pitch_to_node[Pitch("C3")]
        [(<music21.pitch.Pitch G2>, <music21.pitch.Pitch C3>), (<music21.pitch.Pitch C3>, <music21.pitch.Pitch C3>)]
        """
        pitches = {}
        for base, pitch in self.nodes:
            if pitch not in pitches:
                pitches[pitch] = []
            pitches[pitch].append((base, pitch))
        return pitches

    @property
    def names(self) -> list[str]:
        """A list of node names"""
        return list(self.name_to_node.keys())

    @property
    def pitches(self) -> list[Pitch]:
        """The list of pitches found in the graph.
        Note that the order corresponds to the order in self.nodes, and so
        the pitches may not be sorted by pitch."""
        return list(self.pitch_to_node.keys())

    @property
    def first_hexachord(self) -> HexachordGraph:
        """The first registered hexachord according to its base pitch.

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

        Examples
        --------
        >>> gamut = GamutGraph(["G3", "F3"])
        >>> gamut.last_hexachord.base
        <music21.pitch.Pitch G3>
        """
        return dict_last(self.hexachords)

    @property
    def lowest_node(self) -> GamutGraphNode:
        """The lowest node of the graph. This is the lowest pitch of the first
        hexachord, where hexachords are sorted according to their base pitch.
        """
        hex = self.first_hexachord
        return (hex.base, hex.pitches[0])

    @property
    def highest_node(self) -> GamutGraphNode:
        """The highest node of the graph. This is the highest pitch of the last
        hexachord, where hexachords are sorted according to their base pitch.
        """
        hex = self.last_hexachord
        return (hex.base, hex.pitches[-1])

    @cached_property
    def overlapping_hexachords(self) -> dict[HexachordGraph, list[HexachordGraph]]:
        """Identifies hexachords that share common pitches and returns a
        dictionary where each key is a hexachord and the corresponding value is a
        list of hexachords that overlap with it.
        Returns a dictionary mapping each hexachord to a list of hexachords that
        share pitches with it. If the overlapping hexachords have already been
        computed, the cached result is returned.

        Examples
        --------

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

    # Building methods

    def add_hexachord(self, hexachord: t.Union[HexachordGraph, PitchLike]) -> None:
        """Add a hexachord to the gamut graph.

        Parameters
        ----------
        hexachord
            The hexachord to be added. This can be a string representation of a
            hexachord, a Pitch object, or an instance of HexachordGraph. If a
            string or Pitch is provided, it will be converted to a HexachordGraph.

        Raises
        ------
        ValueError
            If the provided hexachord is invalid or if a hexachord with the same
            base has already been added to the gamut graph.
        """
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

    def add_edge_by_names(
        self,
        source: str,
        target: str,
        weight: float = 1,
    ):
        """Add an edge using names of source and target.

        Parameters
        ----------
        source
            The name of the source node
        target
            The name of the target node
        weight
            The weight to use for the edge

        Examples
        --------
        >>> G = GamutGraph(["G2", "C3"])
        >>> G.add_edge_by_names("fa_G2", "re_C3", weight=3)
        >>> (G.get_node(name="fa_G2"), G.get_node(name="re_C3")) in G.edges
        True
        """
        source_node = self.get_node(name=source)
        target_node = self.get_node(name=target)
        self.add_edge(source_node, target_node, weight=weight)

    def add_mutations(self, mutations, default_weight: float = 2):
        """Add mutations between hexachords based on hexachord qualities and direction.
        The mutations list specifies each mutation as a dictionary that
        indicates the source hexachord, target hexachord, direction of movement
        (e.g. moving from natural _up_ to a hard hexachord), and finally the
        actual moves between those two hexachords, specified as syllable pairs.
        For example, these are the mutations in 16th century continental style:

        .. code-block:: python

            mutations = [
                # Mutations from natural hexachords
                dict(source="natural", dir="up", target="hard", moves=[("sol", "re")]),
                dict(source="natural", dir="up", target="hard", moves=[("sol", "re")]),
                dict(source="natural", dir="up", target="soft", moves=[("fa", "re")]),
                dict(source="natural", dir="down", target="hard", moves=[("fa", "la")]),
                dict(source="natural", dir="down", target="soft", moves=[("mi", "la")]),

                # Mutations from hard hexachords
                dict(source="hard", dir="up", target="natural", moves=[("fa", "re")]),
                dict(source="hard", dir="down", target="natural", moves=[("mi", "la")]),

                # Mutations from soft hexachords
                dict(source="soft", dir="up", target="natural", moves=[("sol", "re")]),
                dict(source="soft", dir="down", target="natural", moves=[("fa", "la")]),
            ]

        Examples
        ------
        >>> G = GamutGraph(["G2", "C3", "G3"])
        >>> nat_up_hard = dict(source="natural", dir="up", target="hard", moves=[("sol", "re")])
        >>> nat_down_hard = dict(source="natural", dir="down", target="hard", moves=[("fa", "la", 10)])
        >>> mutations = [nat_up_hard, nat_down_hard]
        >>> G.add_mutations(mutations, default_weight=3)

        Now check the mutation from sol in the natural up to the re of the
        hard hexachord, which will have the default weight of 3

        >>> sol_C3 = G.get_node(name="sol_C3")
        >>> re_G3 = G.get_node(name="re_G3")
        >>> (sol_C3, re_G3) in G.edges
        True
        >>> G[sol_C3][re_G3]
        {'weight': 3}

        And the mutation from the fa in the natural, down to the la of the hard
        hexachord will have a weight of 4, as specified:

        >>> fa_C3 = G.get_node(name="fa_C3")
        >>> la_G2 = G.get_node(name="la_G2")
        >>> (fa_C3, la_G2) in G.edges
        True
        >>> G[fa_C3][la_G2]
        {'weight': 10}


        Parameters
        ----------
        mutations
            A list of mutation dictionaries, where each dictionary contains the
            following keys:

            - 'source' : The quality of the source hexachord ('soft', 'hard', or 'natural').
            - 'target' : The quality of the target hexachord ('soft', 'hard', or 'natural').
            - 'dir' : The direction of the mutation ('up' or 'down').
            - 'moves' : A list of moves associated with the mutation, where each
            move is a tuple of the form :py:`(source_syll, target_syll, optional_weight)`:
            for example, :py:`('sol', 're', 3)` specifies a move from the sol in the source
            hexachord to the re in the target hexachord, and assigns the mutation
            weight 3. If the weight is omitted, the default_weight is used.

        default_weight
            The default weight to assign to edges if not specified in the moves.
            Defaults to 2.

        """
        for source in self.hexachords.values():
            for target in self.overlapping_hexachords[source]:
                direction = "up" if target.base > source.base else "down"
                for mut in mutations:
                    if not (
                        mut["source"] == source.quality
                        and mut["target"] == target.quality
                        and mut["dir"] == direction
                    ):
                        continue

                    for move in mut["moves"]:
                        source_name = f"{move[0]}_{source.base_name}"
                        target_name = f"{move[1]}_{target.base_name}"
                        weight = default_weight if len(move) == 2 else move[2]
                        self.add_edge_by_names(source_name, target_name, weight=weight)

    # Utilities

    def get_node(self, name: str = None, pitch: Pitch = None) -> GamutGraphNode:
        """Retrieve a node from the graph based on the specified name or pitch.

        Parameters
        ----------
        name
            The name of the node to retrieve. If provided, the function will
            return the corresponding node from the name-to-node mapping.

        pitch
            The pitch of the node to retrieve. If provided, the function will
            return the corresponding node from the pitch-to-node mapping.

        Raises
        ------
        KeyError
            If the specified name or pitch does not exist in the respective
            mappings.
        """
        if name:
            return self.name_to_node[name]
        elif pitch:
            return self.pitch_to_node[pitch]

    # Drawing

    def node_positions(
        self,
        pos_x: t.Literal["order", "diatonic", "ps"] = "diatonic",
        pos_y: t.Literal["order", "diatonic", "ps"] = "order",
        offset_x: float = 0,
        offset_y: float = 0,
    ) -> dict[GamutGraphNode, tuple[float, float]]:
        """Calculate positions for the nodes that can be used for
        plotting the graph. Returns a dictionary mapping each
        (base, pitch) pair to its corresponding (x, y) position as a tuple of floats.


        Parameters
        ----------
        pos_x
            The method to determine the x-coordinate of the position.
            'order' uses the index of the pitch in the hexachord,
            'diatonic' uses the diatonic note number, and 'ps' uses
            the pitch class. Default is 'diatonic'.

        pos_y
            The method to determine the y-coordinate of the position.
            'order' uses the index of the hexachord, 'diatonic' uses
            the diatonic note number of the base pitch, and 'ps' uses
            the pitch class of the base pitch. Default is 'order'.

        offset_x
            An offset to be added to the x-coordinate. Default is 0.

        offset_y
            An offset to be added to the y-coordinate. Default is 0.

        Raises
        ------
        ValueError
            If an invalid value is provided for pos_x or pos_y.
        """
        positions = {}
        for i, (base, hexachord) in enumerate(self.hexachords.items()):
            for j, pitch in enumerate(hexachord.pitches):
                match pos_x:
                    case "order":
                        x = self.pitches.index(pitch)
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

    def draw(self, **kws):
        """Draws a gamut graph. This is a shorthand for :func:`delasol.utils.drawing.draw_gamut_graph`.
        Note that the drawing function is lazy-loaded: so only imported when
        you call the `GamutGraph.draw` method.

        Parameters
        ----------
        **kws
            See :func:`delasol.utils.drawing.draw_gamut_graph`
        """
        from delasol.utils.drawing import draw_gamut_graph

        draw_gamut_graph(self, **kws)


###############################################################################

# Gamut graph registry

GAMUTS = {}
"""dict[str, GamutGraph]: A dictionary of registered gamut graphs"""


def register_gamut(gamut: GamutGraph) -> None:
    """Register a GamutGraph. See :class:`delasol.solmizers.continental_16c` for an example
    of how to register a new GamutGraph.

    Parameters
    ----------
    gamut : GamutGraph
        the GamutGraph class to be registered. It must have a 'name' attribute.

    Raises
    ------
    ValueError
        If the provided gamut is not an instance of GamutGraph or if it does
        not have a 'name' attribute.
    """
    if not issubclass(gamut, GamutGraph):
        raise ValueError("The gamut graph must be an instance of GamutGraphs")
    if not hasattr(gamut, "name"):
        raise ValueError("A gamut graph must have a 'name' attribute")

    GAMUTS[gamut.name] = gamut


def get_gamut(name: str, **kws) -> GamutGraph:
    """Get a new gamut graph instance by its name.

    Parameters
    ----------
    name : str
        The name of the gamut graph to retrieve.

    **kws : keyword arguments
        Additional keyword arguments to pass to the gamut graph class
        constructor. See :class:`delasol.graphs.gamut_graph.GamutGraph`
        for all keywords.

    Raises
    ------
    ValueError
        If no gamut graph with the given name has been registered.
    """
    gamut_class = GAMUTS.get(name)
    if not gamut_class:
        raise ValueError(f"Gamut graph '{name}' not found.")
    return gamut_class(**kws)


###############################################################################


# Ensure that doctest also evaluates these cached properties
__test__ = {
    "GamutGraph.pitches": GamutGraph.pitches,
    "GamutGraph.names": GamutGraph.names,
    "GamutGraph.overlapping_hexachords": GamutGraph.overlapping_hexachords,
}
