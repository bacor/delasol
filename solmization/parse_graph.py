# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from music21.pitch import Pitch
from typing import Callable, Iterable, Any

from .utils import segment_deviations, draw_graph
from .gamut_graph import GamutGraph

OrigGraphNode = Any
ParseGraphNode = tuple[int, OrigGraphNode]
SequenceItem = Any


def match_fn(node: OrigGraphNode, target: SequenceItem) -> bool:
    return node[1] == target


class ParseGraph(nx.DiGraph):
    def __init__(
        self,
        graph: nx.Graph,
        sequence: Iterable[SequenceItem] = None,
        match_fn: Callable[[OrigGraphNode, SequenceItem], bool] = match_fn,
        prune: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.orig = graph
        self.match_fn = match_fn
        self._shortest_paths = {}
        self._segments = None
        self._width = None
        self._positions = None
        if sequence is not None:
            self.build(sequence, prune=prune)

    def __repr__(self):
        return f"<ParseGraph of {self.orig.__class__.__name__}>"

    def __len__(self):
        return max(*self.positions.keys()) + 1

    @property
    def positions(self) -> dict[int, list[ParseGraphNode]]:
        if self._positions is None:
            self._positions = {}
            for node in self.nodes:
                pos, _ = node
                if pos not in self._positions:
                    self._positions[pos] = []
                self._positions[pos].append(node)
        return self._positions

    @property
    def width(self) -> np.ndarray:
        """A numpy array with the number of nodes at each position."""
        if self._width is None:
            self._width = np.zeros(len(self))
            for pos, _ in self.nodes:
                self._width[pos] += 1
        return self._width

    @property
    def segments(self):
        """Return a list of segments, based on the width of the parse graph. A segment is a portion of the graph delimited by a unique starting node and a unique ending node. All paths in the graph thus agree on the starting and ending node of a segment."""
        if self._segments is None:
            segmentIndices = segment_deviations(self.width, value=1)
            self._segments = []
            for start, end in segmentIndices:
                # Note that start and endnodes are unique
                self._segments.append(
                    dict(
                        start=start,
                        end=end,
                        startNode=self.positions[start][0],
                        endNode=self.positions[end][0],
                    )
                )
        return self._segments

    # Todo memoize?
    def search(
        self, target: Any, nodes: Iterable[OrigGraphNode] = None
    ) -> list[tuple[OrigGraphNode, dict]]:
        """Search for nodes matching a certain target value using the match function."""
        if nodes is None:
            nodes = self.orig.nodes
        matches = []
        for node in nodes:
            if not node in self.orig:
                raise ValueError(f"Node {node} is not in the original graph.")
            if self.match_fn(node, target):
                matches.append(node)
        return matches

    def shortest_paths(self, source_value, target_value):
        if (source_value, target_value) not in self._shortest_paths:
            source_matches = self.search(source_value)
            target_matches = self.search(target_value)
            all_paths = []
            for source in source_matches:
                for target in target_matches:
                    paths = nx.all_shortest_paths(self.orig, source, target)
                    all_paths.extend(paths)

            # Store the shortest paths
            shortest_length = min([len(path) for path in all_paths])
            all_paths = [path for path in all_paths if len(path) == shortest_length]
            self._shortest_paths[(source, target)] = all_paths

        return self._shortest_paths[(source, target)]

    def add_node_from_orig(self, pos: int, orig_node: OrigGraphNode) -> ParseGraphNode:
        """Add a node to the parse graph based on the position and a node in the original graph."""
        node = (pos, orig_node)
        attrs = dict(**self.orig.nodes[orig_node])
        self.add_node(node, **attrs)
        return node

    def _add_path(self, start, path):
        new_nodes = []
        for i, orig_node in enumerate(path):
            pos = start + i
            new_node = (pos, orig_node)
            if new_node not in self.nodes:
                self.add_node_from_orig(pos, orig_node)
            if i >= 1:
                orig_weight = self.orig[new_nodes[-1][1]][new_node[1]]["weight"]
                self.add_edge(new_nodes[-1], new_node, weight=orig_weight)

            new_nodes.append(new_node)
        return new_nodes

    def prune_branch(self, source: ParseGraphNode):
        """Remove all predecessors of a node that have only one successor. This allows us to
        prune branches that cannot parse the sequence anyway."""
        predecessors = list(self.predecessors(source))
        for predecessor in predecessors:
            if self.out_degree[predecessor] == 1:
                self.prune_branch(predecessor)
                self.remove_node(predecessor)

        if self.out_degree[source] == 0:
            self.remove_node(source)

    def set_node_positions(self):
        def orig_y_position(node):
            if node[1] in self.orig.nodes:
                orig_node_attrs = self.orig.nodes[node[1]]
                return orig_node_attrs.get("position", (0, 0))[1]
            return 0

        for pos, nodes in self.positions.items():
            nodes = sorted(nodes, key=orig_y_position)
            for i, node in enumerate(nodes):
                self.nodes[node]["position"] = (pos, i)

    def build(self, sequence: Iterable[SequenceItem], prune: bool = True):
        self.clear()
        self.seq = sequence
        self.start = (0, "START")
        self.add_node(self.start, label="START")

        # First step: from start to first matching nodes (in the original graph)
        matches = self.search(self.seq[0])
        for orig_node in matches:
            new_node = self.add_node_from_orig(1, orig_node)
            self.add_edge(self.start, new_node, weight=0)

        pos = 1
        prev_nodes = [node for node in matches]
        self.input_positions = [1]
        for prev_value, next_value in zip(self.seq, self.seq[1:]):
            # Find all paths from prev_value to next_value
            paths = self.shortest_paths(prev_value, next_value)
            paths = [p for p in paths if p[0] in prev_nodes]
            if len(paths) == 0:
                raise Exception("This sequence could not be parsed")

            # Fix repetitions: append the same node to the path again, hacky but works
            if prev_value == next_value:
                for path in paths:
                    path.append(path[0])

            # Add paths to the next nodes
            next_nodes = []
            for prev_node in prev_nodes:
                for path in paths:
                    if path[0] == prev_node:
                        new_nodes = self._add_path(pos + 1, path[1:])
                        orig_weight = self.orig[prev_node][new_nodes[0][1]]["weight"]
                        self.add_edge(
                            (pos, prev_node), new_nodes[0], weight=orig_weight
                        )
                        next_nodes.append(new_nodes[-1][1])

            # Prune paths
            if prune:
                for prev_node in prev_nodes:
                    if self.out_degree[(pos, prev_node)] == 0:
                        self.prune_branch((pos, prev_node))

            pos = new_nodes[-1][0]
            self.input_positions.append(pos)
            prev_nodes = set(next_nodes)

        # Finish up: connect to end node
        self.end = (pos + 1, "END")
        self.add_node(self.end, label="END")
        for prev_node in prev_nodes:
            self.add_edge((pos, prev_node), self.end, weight=0)
        self.set_node_positions()

    def path_segments(self):
        segments = []
        for segment in self.segments:
            startNode = segment["startNode"]
            endNode = segment["endNode"]
            if startNode == endNode:
                paths = [[startNode]]
                weights = [0]
            else:
                paths = list(nx.all_simple_paths(self, startNode, endNode))
                weights = [nx.path_weight(self, path, "weight") for path in paths]
            ranking = np.argsort(weights)
            segments.append(
                dict(
                    paths=[paths[i] for i in ranking],
                    weights=[weights[i] for i in ranking],
                    **segment,
                )
            )
        return segments

    def draw(self, fig=None, show_segments: bool = True, show_axes: bool = True, **kws):
        if fig is None:
            plt.figure(figsize=(len(self), self.width.max()))
        if show_segments:
            for segment in self.segments[1:]:
                plt.gca().axvline(
                    segment["start"] - 0.5, color="k", lw=0.5, linestyle="--"
                )
        draw_graph(self, **kws)
        if show_axes:
            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axis_on()
            ax.xaxis.grid(color=".9")
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            ax.set_xticks(range(len(self)))
            xlabels = ["start"] + [f"{i}" for i in range(1, len(self) - 1)] + ["end"]
            for i, pos in enumerate(self.input_positions):
                xlabels[pos] += f"\n{self.seq[i]}"
            ax.set_xticklabels(xlabels)
            ax.set_yticks(np.arange(0, self.width.max()))
            ax.set_yticklabels(np.arange(1, self.width.max() + 1, dtype=int))
            plt.ylabel("width")
            plt.ylim(-0.75, self.width.max() - 0.75)
        else:
            plt.axis("off")


class GamutParseGraph(ParseGraph):
    def __init__(
        self,
        gamut: GamutGraph,
        sequence: Iterable[Pitch] = None,
        mismatch_penalty: float = 0,
        match_fn: Any = None,
        **kwargs,
    ):
        if not isinstance(gamut, GamutGraph):
            raise ValueError("The graph should be a GamutGraph.")
        if sequence is not None and not isinstance(sequence[0], Pitch):
            raise ValueError("The sequence should be a list of pitches.")
        if match_fn is not None:
            raise Warning("The match function is ignored for GamutParseGraph.")

        self.gamut = gamut
        self.mismatch_penalty = mismatch_penalty
        super().__init__(graph=gamut, sequence=sequence, **kwargs)

    def search(
        self, target: Pitch, nodes: Iterable[OrigGraphNode] = None
    ) -> list[tuple[OrigGraphNode, dict]]:
        """Search for nodes with a matching pitch"""
        if nodes is None:
            nodes = self.orig.nodes
        return [n for n in nodes if n[1].diatonicNoteNum == target.diatonicNoteNum]

    def build(self, sequence: Iterable[Pitch], prune: bool = True):
        super().build(sequence, prune=prune)

        # Add a mismatch penalty to all nodes that do not exactly match the target pitch
        for pos, target in zip(self.input_positions, sequence):
            for node in self.positions[pos]:
                _, (_, pitch) = node
                if pitch != target:
                    for predecessor in self.predecessors(node):
                        self[predecessor][node]["weight"] += self.mismatch_penalty
