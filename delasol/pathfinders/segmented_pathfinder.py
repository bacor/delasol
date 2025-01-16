# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from functools import cached_property

# Library imports
import numpy as np
import networkx as nx

# Local imports
from delasol.utils.sequence import segment_deviations
from delasol.custom_types import RolloutGraphPath, RolloutGraphNode
from delasol.graphs.rollout_graph import RolloutGraph
from delasol.pathfinders.pathfinder import Pathfinder


class Segment(object):
    """A segment class

    A segment is a window of the base graph that has only a single start and
    end node. This means that all paths through the base graph will agree at
    the start and end of a segment. All shortest paths through a segment are
    immediately computed.

    Parameters
    ----------
    graph : RolloutGraph
        The base graph to segment. The graph needs to have a timesteps attribute.
    start : int
        The starting time of the segment. There must be exactly one node in
        this time slice.
    end : int
        The ending node of the segment. There must be exactly one node in
        this time slice.

    Attributes
    ----------
    graph : RolloutGraph
        The base graph to segment.
    start : int
        The starting time of the segment.
    end : int
        The ending time of the segment.
    start_node : RolloutGraphNode
        The node at the start of the segment.
    end_node : RolloutGraphNode
        The node at the end of the segment.
    paths : list of RolloutGraphPath
        A list of all shortest paths through the segment.
    weights : list of float
        The weights of the paths in the paths attribute.

    Raises
    ------
    ValueError
        If the provided graph does not have a timesteps attribute, or
        if the start and end nodes are not unique.
    """

    def __init__(
        self,
        graph: nx.Graph,
        start: int,
        end: int,
    ):
        """Initialize a new instance of the class."""
        if not hasattr(graph, "timesteps"):
            raise ValueError("The graph must have a timesteps attibute.")

        if len(graph.timesteps[start]) > 1 or len(graph.timesteps[end]) > 1:
            raise ValueError("Segments must have a unique end and starting node")

        # Store variables
        self.graph = graph
        self.start = start
        self.end = end

        # Find start and end node
        self.start_node = graph.timesteps[start][0]
        self.end_node = graph.timesteps[end][0]

        # Find all shortest simple paths and their weights
        if self.start_node == self.end_node:
            self.paths = [[self.start_node]]
            self.weights = [0]
        else:
            self.paths = list(
                nx.shortest_simple_paths(
                    graph, self.start_node, self.end_node, "weight"
                )
            )
            self.weights = [
                nx.path_weight(graph, path, "weight") for path in self.paths
            ]

    def __repr__(self):
        return f"<Segment {self.start}–{self.end} of {repr(self.graph)}>"

    def __len__(self):
        return self.end - self.start


class SegmentsGraph(nx.DiGraph):
    """
    Initialize a new instance of the class.

    A segment is a window of the base graph that has only a single start and
    end node. This means that all paths through the base graph will agree at
    the start and end of a segment. In a segments graph, each node corresponds
    to a path in a segment, and the weight of the incoming edge is the total
    weight of that path. Similar to the underlying rollout, the graph is linear
    and the shortest path from the START to END node will have the exact same
    length as the shortest path in the base graph.

    Parameters
    ----------
    base : RolloutGraph
        The base rollout graph to be segmented.

    Attributes
    ----------
    segments : list of Segment
        A list of segments created from the base rollout graph.

    Notes
    -----
    This constructor divides the base graph into segments based on the
    calculated deviations and builds the graph accordingly.
    """

    def __init__(self, base: RolloutGraph):
        super().__init__()
        self.base = base

        # Divide the graph in segments
        self.segments = []
        positions = segment_deviations(base.width, value=1)
        for start, end in positions:
            segment = Segment(base, start, end)
            self.segments.append(segment)

        # Build the graph
        self.__build()

    def __repr__(self) -> str:
        return f"<SegmentsGraph len={len(self.segments)} base={repr(self.base)}>"

    def __build(self) -> None:
        """Builds a segments graph."""
        # Create start node
        self.add_node("START")
        prev_nodes = ["START"]

        # Create nodes for each path in every segment
        for s, segment in enumerate(self.segments):
            new_nodes = []
            for p, weight in enumerate(segment.weights):
                self.add_node((s, p))
                new_nodes.append((s, p))
                for prev_node in prev_nodes:
                    self.add_edge(prev_node, (s, p), weight=weight)
            prev_nodes = new_nodes

        # Connect to end node
        self.add_node("END")
        for prev_node in prev_nodes:
            self.add_edge(prev_node, "END", weight=0)

    def iter_shortest_paths(self) -> t.Iterator[RolloutGraphPath]:
        """Iterates over all paths through the base graph, ranked by path length (weight).

        Yields
        ------
        RolloutGraphPath
            A path through the base graph, represented as a list of nodes.
        """
        segment_paths = nx.shortest_simple_paths(self, source="START", target="END")
        for segment_path in segment_paths:
            path = []
            for s, p in segment_path[1:-1]:
                segment = self.segments[s]
                path.extend(segment.paths[p])
            yield path

    # Drawing

    def node_positions(self) -> dict[RolloutGraphNode, t.Tuple[int, int]]:
        """Calculate the positions of nodes to be used in plotting.

        The "START" node is positioned at (-1, 0) and the "END" node is
        positioned at (length of segments, 0). All other nodes are
        returned as they are.

        Returns
        -------
        dict
            A dictionary mapping each node to its corresponding position
        """
        positions = {}
        for node in self.nodes:
            if node == "START":
                positions[node] = np.array([-1, 0])
            elif node == "END":
                positions[node] = np.array([len(self.segments), 0])
            else:
                positions[node] = node
        return positions

    def draw(self) -> None:
        """Draw the graph"""
        nx.draw(self, pos=self.node_positions(), with_labels=True)


class SegmentedPathfinder(Pathfinder):
    """A segmented pathfinder.

    This pathfinder divides the base graph into segments, where each segment
    is a window of the base graph that has only a single start and end node.
    This means that all paths through the base graph will agree at the start
    and end of a segment. The pathfinder then finds the shortest path through
    the base graph by finding the shortest path through each segment.

    Moreover, it constructs a segments graph, where nodes are paths through
    segments. The shortest path in the segments graph will correspond to the
    shortest path through the base graph.

    The main benefit of this approach is that it allows you to print all
    alternative solmizations per segment, rather than for the entire input.

    Parameters
    ----------
    graph : RolloutGraph
        The graph to find paths in. Must be a RolloutGraph.

    **kws : keyword arguments
        Additional arguments passed to AbstractPathfinder.

    Raises
    ------
    ValueError
        If the provided graph is not a RolloutGraph.
    """

    def __init__(self, graph: RolloutGraph, **kws):
        if not isinstance(graph, RolloutGraph):
            raise ValueError("The graph must be a RolloutGraph")

        # Initialize a segments graph
        self.segments_graph = SegmentsGraph(graph)
        self.segments = self.segments_graph.segments

        super().__init__(graph, **kws)

    def find_paths(self, *args, **kws):
        for path in self.segments_graph.iter_shortest_paths():
            yield path, dict()

    @cached_property
    def time_to_segment(self):
        time_to_segment = {}
        for segment in self.segments:
            for pos in range(segment.start, segment.end + 1):
                time_to_segment[pos] = segment
        return time_to_segment

    # DEPRECATED

    # def step(self, time: int):
    #     segment = self.time_to_segment[time]
    #     return segment[time]

    # def iter_steps(self, timesteps: t.Iterable[int], return_orig_node: bool = True):
    #     last_segment = None
    #     for time in timesteps:
    #         segment = self.time_to_segment[time]
    #         step = segment[time]
    #         if return_orig_node:
    #             step["nodes"] = [n[1] for n in step["nodes"]]
    #         step["is_first"] = segment != last_segment
    #         last_segment = segment
    #         yield step

    # def iter_selected_paths(
    #     self,
    #     selector: t.Callable[[Segment], int],
    #     positions: t.Iterable[int] = None,
    #     return_orig_node: bool = True,
    # ):
    #     for index, segment in enumerate(self.segments):
    #         index = selector(index, segment)
    #         for i, step in enumerate(segment):
    #             pos = segment.start + i
    #             if positions is None or pos in positions:
    #                 node = step["nodes"][index]
    #                 yield node[1] if return_orig_node else node

    # def iter_nth_path(self, n: int = 0, **kwargs):
    #     selector = lambda index, segment: min(n, len(segment.paths))
    #     return self.iter_selected_paths(selector, **kwargs)

    # def iter_best_path(self, **kwargs):
    #     return self.iter_nth_path(0, **kwargs)
