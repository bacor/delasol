# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
from functools import cached_property

import numpy as np
import networkx as nx
from typing import Callable, Iterable, Any

from .utils import segment_deviations

OrigGraphNode = Any
ParseGraphNode = tuple[int, OrigGraphNode]
SequenceItem = Any
Path = list[ParseGraphNode]


class Segment:
    def __init__(self, graph: nx.Graph, start: int, end: int):
        # if not isinstance(graph, ParseGraph):
        #     raise Exception("Only segments of parse graphs are currently supported.")
        if len(graph.positions[start]) > 1 or len(graph.positions[end]) > 1:
            raise ValueError("Segments must have a unique end and starting node")

        self.graph = graph
        self.start = start
        self.end = end
        self.start_node = self.graph.positions[start][0]
        self.end_node = self.graph.positions[end][0]
        if self.start_node == self.end_node:
            paths = [[self.start_node]]
            weights = [0]
        else:
            paths = list(
                nx.all_simple_paths(self.graph, self.start_node, self.end_node)
            )
            weights = [nx.path_weight(self.graph, path, "weight") for path in paths]

        self.ranking = np.argsort(weights)
        self.paths = [paths[i] for i in self.ranking]
        self.weights = [weights[i] for i in self.ranking]

    def __repr__(self):
        return f"<Segment {self.start}–{self.end} of {repr(self.graph)}>"

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        for pos in range(self.start, self.end + 1):
            yield self[pos]

    def __getitem__(self, index: int) -> dict:
        index = index - self.start
        return dict(
            nodes=[path[index] for path in self.paths],
            weights=self.weights,
            pos_in_segment=index,
        )


class SegmentedGraph(nx.DiGraph):
    _cached_properties = ["positions", "width", "segments", "pos_to_segment"]

    @property
    def length(self):
        return len(self.positions)

    @cached_property
    def positions(self) -> dict[int, list[ParseGraphNode]]:
        positions = {}
        for node in self.nodes:
            pos, _ = node
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(node)
        return positions

    @cached_property
    def width(self) -> np.ndarray:
        """A numpy array with the number of nodes at each position."""
        width = np.zeros(len(self))
        for pos, _ in self.nodes:
            width[pos] += 1
        return width

    @cached_property
    def segments(self) -> list[Segment]:
        segments = []
        positions = segment_deviations(self.width, value=1)
        for start, end in positions:
            segment = Segment(self, start, end)
            segments.append(segment)
        return segments

    @cached_property
    def pos_to_segment(self):
        pos_to_segment = {}
        for segment in self.segments:
            for pos in range(segment.start, segment.end + 1):
                pos_to_segment[pos] = segment
        pos_to_segment

    def reset_cached_properties(self):
        """Reset all the cached properties"""
        # TODO does this work?
        for key in self._cached_properties:
            if key in self.__dict__:
                del self.__dict__[key]

    def step(self, position: int):
        segment = self.pos_to_segment[position]
        return segment[position]

    def iter_steps(
        self, positions: Iterable[int] | None = None, return_orig_node: bool = True
    ):
        last_segment = None
        for position in positions:
            segment = self.pos_to_segment[position]
            step = segment[position]
            if return_orig_node:
                step["nodes"] = [n[1] for n in step["nodes"]]
            step["is_first"] = segment != last_segment
            last_segment = segment
            yield step

    def iter_selected_paths(
        self,
        selector: Callable[[Segment], int],
        positions: Iterable[int] = None,
        return_orig_node: bool = True,
    ):
        for index, segment in enumerate(self.segments):
            index = selector(index, segment)
            for i, step in enumerate(segment):
                pos = segment.start + i
                if positions is None or pos in positions:
                    node = step["nodes"][index]
                    yield node[1] if return_orig_node else node

    def iter_nth_path(self, n: int = 0, **kwargs):
        selector = lambda index, segment: min(n, len(segment.paths))
        return self.iter_selected_paths(selector, **kwargs)

    def iter_best_path(self, **kwargs):
        return self.iter_nth_path(0, **kwargs)
