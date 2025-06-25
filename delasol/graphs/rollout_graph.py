# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from functools import lru_cache, cached_property

# Library imports
import networkx as nx

# Local imports
from delasol.custom_types import BaseGraphNode, SeqType, RolloutGraphNode, BaseGraphPath


def match_node(node: BaseGraphNode, target: SeqType) -> bool:
    return node == target


def match_second_el(node: BaseGraphNode, target: SeqType) -> bool:
    return node[1] == target


# TODO add a simpler non-interpolated rollout graph and actually compare
# performance. Isn't it more efficient to interpolate the melody diatonically
# and then an actual stepwise rollout graph, exactly mirroring a weighted
# finite state automaton?


class RolloutGraph(nx.DiGraph):
    """
    An (interpolated) rollout graph.
    A rollout graph basically shows all ways a sequence can be 'parsed' by
    the base graph. The sequence contains values that are somehow related to
    the nodes in the base graph (e.g. pitches, names, etc.). Note that the
    sequence can contain jumps, i.e. values that are not directly connected
    in the base graph. The rollout graph will attempt to fill those gaps with
    stepwise movements.

    Starting from a special START node, the rollout adds paths to all nodes in
    the base graph that match the first value in the sequence. Then, for each
    subsequent value in the sequence, it adds paths to the nodes in the base
    graph that match the value, and that are reachable from the previous nodes.
    The rollout graph ends with a special END node.

    Every node in the rollout graph has a coordinate `(time, index)` where
    `time` indicates the time step. Note that because the sequence can contain
    jumps, the length of the rollout graph is the number of timesteps, and this
    is usually greater than the length of the sequence (and of course the START
    and END nodes). The width of the rollout graph at a given timestep is the
    number of nodes at that timestep.

    Examples
    --------

    Here's an example using a complete graph with three nodes (no loops)

    >>> import networkx as nx
    >>> graph = nx.complete_graph(3)
    >>> rollout = RolloutGraph(graph, [0, 2, 1])
    >>> rollout
    <RolloutGraph base=Graph seq=[0, 2, 1]>
    >>> rollout.sequence
    [0, 2, 1]
    >>> rollout.width
    [1, 1, 1, 1, 1]
    >>> rollout.nodes
    NodeView(((0, 'START'), (1, 0), (2, 2), (3, 1), (4, 'END')))

    And here we have a circulant graph with a symmetric, jumpy path from the
    first node to the third, and back:

    >>> graph = nx.circulant_graph(6, [1])
    >>> rollout = RolloutGraph(graph, [0, 3, 0])
    >>> len(rollout)
    9
    >>> rollout.width
    [1, 1, 2, 2, 1, 2, 2, 1, 1]

    Parameters
    ----------
    graph
        The base graph.

    sequence
        A sequence of values to be unrolled. Must not be empty.

    match_fn
        A function that takes a node from the base graph and value
        and returns a boolean indicating a match. Defaults to `match_node`.

    prune_dead_ends
        A flag indicating whether to prune dead-ends: branches that cannot
        be extended to parse the input sequence. Defaults to True.

    default_weight
        The default weight used if an edge in the base graph is unweighed.

    **kwargs
        Additional keyword arguments to be passed to the nx.DiGraph
        initializer.

    Raises
    ------
    ValueError
        If the provided sequence is None or empty.

    Attributes
    ----------
    base : nx.Graph
        The base graph.
    match_fn : Callable[[BaseGraphNode, SeqType], bool]
        The function used to match nodes to values
    input_timesteps : Iterable[int] | None
        A list of timesteps at which the input values are found
    default_weight : float
        The default weight used if an edge in the base graph is unweighed
    start : RolloutGraphNode
        The very first START node at time 0: `(0, 'START')`.
    end : RolloutGraphNode
        The very last END node at the end of the rollout: `(len(self), 'END')`.
    """

    input_timesteps: t.Iterable[int] | None = None
    """A list of timesteps at which the input values are found"""

    def __init__(
        self,
        graph: nx.Graph,
        sequence: t.Iterable[SeqType],
        match_fn: t.Callable[[BaseGraphNode, SeqType], bool] = match_node,
        prune_dead_ends: bool = True,
        default_weight: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence is None or len(sequence) == 0:
            raise ValueError("You must provide a sequence to parse.")

        self.base = graph
        self.__seq = sequence
        self.match_fn = match_fn
        self.default_weight = default_weight
        self.__build(prune_dead_ends=prune_dead_ends)

    def __repr__(self):
        limit = 4
        name = self.base.__class__.__name__
        seq = ", ".join([str(item) for item in self.sequence][:limit])
        if len(self.sequence) > limit:
            return f"<RolloutGraph base={name} seq=[{seq}, ...]>"
        else:
            return f"<RolloutGraph base={name} seq=[{seq}]>"

    def __len__(self):
        return max(*self.timesteps.keys()) + 1

    @property
    def sequence(self) -> t.Iterable[SeqType]:
        """The sequence parsed by this parse graph. Note that this is a read-only property."""
        return self.__seq

    # TODO rename this to something like slices so that timesteps returns the actual timesteps only.
    @cached_property
    def timesteps(self) -> dict[int, list[RolloutGraphNode]]:
        """A dictionary mapping timesteps to a list of nodes found at that time slice."""
        timesteps = {}
        for node in self.nodes:
            time = node[0]
            if time not in timesteps:
                timesteps[time] = []
            timesteps[time].append(node)
        return timesteps

    @cached_property
    def width(self) -> list[int]:
        """The width of the rollout at each timestep. This function returns a list
        where each element represents the number of nodes at each timestep.

        Examples
        --------
        >>> rollout = RolloutGraph(nx.complete_graph(3), [0, 2, 3, 1])
        >>> rollout.width
        [1, 1, 2, 2, 1, 2, 2, 1, 1]
        """
        # TODO does the doctest work correctly?
        width = [1] * len(self)
        for time, nodes in self.timesteps.items():
            width[time] = len(nodes)
        return width

    def slice(self, time: int) -> list["RolloutGraphNode"]:
        """Return the nodes at a given time step.

        Parameters
        ----------
        time
            The time step for which to retrieve the nodes.
        """
        if time not in self.timesteps:
            raise ValueError(f"Time step {time} is out of range (0–{len(self)}).")
        return self.timesteps[time]

    ## Search the base graph

    @lru_cache(maxsize=None)
    def search_base(
        self, target: SeqType, nodes: t.Iterable[BaseGraphNode] = None
    ) -> list["BaseGraphNode"]:
        """Search for nodes in the original graph that match a given target.

        Parameters
        ----------
        target
            The target to search for in the nodes of the graph. This must be
            of the same datatype as the sequence for which the parse graph was
            constructed.
        nodes
            An iterable of nodes in the original graph to search through. If
            None, all nodes in the original graph will be searched.

        Raises
        ------
        ValueError
            If a node is passed that is not in the original graph.
        """
        if nodes is None:
            nodes = self.base.nodes
        matches = []
        for node in nodes:
            if not node in self.base:
                raise ValueError(f"Node {node} is not in the original graph.")
            if self.match_fn(node, target):
                matches.append(node)
        return matches

    @lru_cache(maxsize=None)
    def shortest_paths_base(
        self, source_value: BaseGraphNode, target_value: BaseGraphNode
    ) -> list["BaseGraphPath"]:
        """Return the shortest paths between two nodes in the base graph.
        This function searches for all shortest paths between the specified source
        and target nodes in the graph. It memoizes the results to optimize future
        calls. If no paths exist, an empty list is returned.

        Parameters
        ----------
        source_value
            The starting node for the path search.
        target_value
            The ending node for the path search.
        """
        # TODO this is inefficient: whenever we search for a shortest path, we should
        # immediately memoize all intermediate paths. Next, the memoization should not
        # be based on the values but on the base nodes (since multiple values can
        # map to the same node.)
        # Perhaps make a PathFinder/Interpolator class to implement different strategies?
        source_matches = self.search_base(source_value)
        target_matches = self.search_base(target_value)
        all_paths = []
        for source in source_matches:
            for target in target_matches:
                paths = nx.all_shortest_paths(self.base, source, target)
                all_paths.extend(paths)

        if not all_paths:
            return []

        # Store the shortest paths
        shortest_length = min([len(path) for path in all_paths])
        all_paths = [path for path in all_paths if len(path) == shortest_length]
        return all_paths

    ## Private construction methods

    def __add_node(self, time: int, base_node: BaseGraphNode) -> RolloutGraphNode:
        """Add a node from the base graph to the rollout at a particular time.
        The newly added node is returned.

        Parameters
        ----------
        time
            The time at which the node is to be added.
        base_node
            The node from the base graph that will be added to the rollout.
        """
        node = (time, base_node)
        attributes = dict(**self.base.nodes[base_node])
        self.add_node(node, **attributes)
        return node

    def __add_path(
        self, start_time: int, path: list[BaseGraphNode]
    ) -> list[RolloutGraphNode]:
        """Add a path of nodes to the graph starting from a given time.
        The newly created nodes are all returned.

        Parameters
        ----------
        start_time
            The starting time for the first node in the path.
        path
            A list of base graph nodes to be added to the graph.

        Notes
        -----
        This function creates new nodes in the graph for each base node in the
        provided path, starting from the specified start time. It also adds
        edges between consecutive nodes, using the weight from the base graph
        if available, or a default weight otherwise.
        """

        new_nodes = []
        for i, base_node in enumerate(path):
            time = start_time + i
            new_node = (time, base_node)
            if new_node not in self.nodes:
                self.__add_node(time, base_node)
            if i >= 1:
                orig_weight = self.base[new_nodes[-1][1]][new_node[1]].get(
                    "weight", self.default_weight
                )
                self.add_edge(new_nodes[-1], new_node, weight=orig_weight)

            new_nodes.append(new_node)
        return new_nodes

    def __build(self, prune_dead_ends: bool = True) -> None:
        """Builds the rollout graph from a given sequence.
        This method initializes the graph by adding nodes and edges based on the
        provided sequence. It starts from a designated starting node and connects
        to subsequent nodes based on the shortest paths found between values in
        the sequence. The option to prune dead-end nodes is available.

        Parameters
        ----------
        prune_dead_ends
            If True, dead-end nodes will be removed from the graph after each
            iteration. Default is True.

        Raises
        ------
        Exception
            If the sequence cannot be parsed due to missing paths between values.
        """
        # Starting node
        self.start = (0, "START")
        self.add_node(self.start, name="START", coordinates=(0, 0))

        # First step: from start to first matching nodes (in the original graph)
        matches = self.search_base(self.__seq[0])
        for base_node in matches:
            new_node = self.__add_node(1, base_node)
            self.add_edge(self.start, new_node, weight=0)

        pos = 1
        prev_nodes = [node for node in matches]
        self.input_timesteps = [1]
        for prev_value, next_value in zip(self.__seq, self.__seq[1:]):
            # Find all paths from prev_value to next_value
            paths = self.shortest_paths_base(prev_value, next_value)
            paths = [p for p in paths if p[0] in prev_nodes]
            if len(paths) == 0:
                raise Exception("This sequence could not be parsed")

            # Add paths to the next nodes
            next_nodes = []
            for path in paths:
                prev_node = path[0]
                if prev_value != next_value:
                    path = path[1:]
                new_nodes = self.__add_path(pos + 1, path)
                base_weight = self.base[prev_node][new_nodes[0][1]].get(
                    "weight", self.default_weight
                )
                self.add_edge((pos, prev_node), new_nodes[0], weight=base_weight)
                next_nodes.append(new_nodes[-1][1])

            # Prune paths
            if prune_dead_ends:
                for prev_node in prev_nodes:
                    if self.out_degree[(pos, prev_node)] == 0:
                        self.prune_branch((pos, prev_node))

            pos = new_nodes[-1][0]
            self.input_timesteps.append(pos)
            prev_nodes = set(next_nodes)

        # Finish up: connect to end node
        self.end = (pos + 1, "END")
        self.add_node(self.end, name="END", coordinates=(pos + 1, 0))
        for prev_node in prev_nodes:
            self.add_edge((pos, prev_node), self.end, weight=0)

        # Set the node coordinates
        for time, nodes in self.timesteps.items():
            nodes = sorted(nodes)
            for idx, node in enumerate(nodes):
                self.nodes[node]["coordinates"] = (time, idx)

    # Pruning

    def prune_branch(self, source: RolloutGraphNode) -> None:
        """Remove all predecessors of a node that have only one successor.
        This method prunes branches in the graph that cannot parse the sequence
        by removing nodes that only have a single successor. If a predecessor is
        found to have only one successor, it is recursively pruned. If the source
        node has no successors after pruning, it is also removed.

        Parameters
        ----------
        source
            The node from which to start pruning predecessors.
        """
        predecessors = list(self.predecessors(source))
        for predecessor in predecessors:
            if self.out_degree[predecessor] == 1:
                self.prune_branch(predecessor)
                self.remove_node(predecessor)

        if self.out_degree[source] == 0:
            self.remove_node(source)

    # Drawing

    def node_positions(self) -> dict[RolloutGraphNode, tuple[float, float]]:
        """Retrieve the positions of all nodes in the graph. Returns a dictionary
        in which keys are node identifiers and the values are the corresponding
        coordinates of each node."""
        return {node: self.nodes[node]["coordinates"] for node in self.nodes}

    def draw(self, **kws) -> None:
        """Draws a a parse graph. This is a shorthand for
        :func:`delasol.utils.drawing.draw_rollout_graph`. Note that the
        drawing function is lazy-loaded: so only imported when you call the
        `RolloutGraph.draw` method.

        Parameters
        ----------
        **kws : keyword arguments
            See :func:`delasol.utils.drawing.draw_rollout_graph`
        """
        from delasol.utils.drawing import draw_rollout_graph

        draw_rollout_graph(self, **kws)
