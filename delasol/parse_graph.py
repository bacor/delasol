# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from delasol.segmented_graph import Segment, SegmentedGraph
from delasol.utils import draw_graph

OrigGraphNode = t.Any
ParseGraphNode = tuple[int, OrigGraphNode]
SequenceItem = t.Any
Path = list[ParseGraphNode]


def match_fn(node: OrigGraphNode, target: SequenceItem) -> bool:
    return node[1] == target


class ParseGraph(SegmentedGraph):
    input_positions = None
    _shortest_paths = {}

    def __init__(
        self,
        graph: nx.Graph,
        sequence: t.Iterable[SequenceItem],
        match_fn: t.Callable[[OrigGraphNode, SequenceItem], bool] = match_fn,
        prune: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.orig = graph
        self.match_fn = match_fn
        if sequence is not None:
            self.build(sequence, prune=prune)

    def __repr__(self):
        return f"<ParseGraph of {self.orig.__class__.__name__}>"

    def __len__(self):
        return max(*self.positions.keys()) + 1

    ## Parent search operations

    def search(
        self, target: t.Any, nodes: t.Iterable[OrigGraphNode] = None
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

    def shortest_paths(
        self, source_value: OrigGraphNode, target_value: OrigGraphNode
    ) -> dict[tuple[OrigGraphNode, OrigGraphNode], list[list[OrigGraphNode]]]:
        """Return the shortest paths between two nodes in the original graph. This function memoizes the results."""
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
            self._shortest_paths[(source_value, target_value)] = all_paths

        return self._shortest_paths[(source_value, target_value)]

    ## Construction

    def _add_node(self, pos: int, orig_node: OrigGraphNode) -> ParseGraphNode:
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
                self._add_node(pos, orig_node)
            if i >= 1:
                orig_weight = self.orig[new_nodes[-1][1]][new_node[1]]["weight"]
                self.add_edge(new_nodes[-1], new_node, weight=orig_weight)

            new_nodes.append(new_node)
        return new_nodes

    def prune_branch(self, source: ParseGraphNode):
        """Remove all predecessors of a node that have only one successor. This allows us to
        prune branches that cannot parse the sequence t.Anyway."""
        predecessors = list(self.predecessors(source))
        for predecessor in predecessors:
            if self.out_degree[predecessor] == 1:
                self.prune_branch(predecessor)
                self.remove_node(predecessor)

        if self.out_degree[source] == 0:
            self.remove_node(source)

    def clear(self):
        super().reset_cached_properties()
        self._shortest_paths = {}
        super().clear()

    def build(self, sequence: t.Iterable[SequenceItem], prune: bool = True):
        self.clear()
        self.seq = sequence
        self.start = (0, "START")
        self.add_node(self.start, name="START", position=(0, 0))

        # First step: from start to first matching nodes (in the original graph)
        matches = self.search(self.seq[0])
        for orig_node in matches:
            new_node = self._add_node(1, orig_node)
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

            # Add paths to the next nodes
            next_nodes = []
            for prev_node in prev_nodes:
                for path in paths:
                    if path[0] == prev_node:
                        if prev_value != next_value:
                            path = path[1:]
                        new_nodes = self._add_path(pos + 1, path)
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
        self.add_node(self.end, name="END", position=(pos + 1, 0))
        for prev_node in prev_nodes:
            self.add_edge((pos, prev_node), self.end, weight=0)
        self.set_node_positions()

    ## Iterating segments

    def iter_selected_paths(
        self,
        selector: t.Callable[[Segment], int],
        input_only=True,
        **kwargs,
    ):
        if input_only:
            kwargs["positions"] = self.input_positions
        return super().iter_selected_paths(selector, **kwargs)

    def iter_steps(self, input_only=True, **kwargs):
        if input_only:
            kwargs["positions"] = self.input_positions
        return super().iter_steps(**kwargs)

    # Drawing

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

    def draw(
        self,
        fig=None,
        show_segments: bool = True,
        show_axes: bool = True,
        width_factor: float = 0.7,
        **kws,
    ):
        if fig is None:
            plt.figure(figsize=((len(self) - 1) * width_factor, self.width.max()))
        if show_segments:
            for segment in self.segments[1:]:
                plt.gca().axvline(
                    segment.start - 0.5, color="k", lw=0.5, linestyle="--"
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
            plt.ylim(-0.5, self.width.max() - 0.5)
            plt.xlim(-1, len(self))
        else:
            plt.axis("off")
        plt.tight_layout()
