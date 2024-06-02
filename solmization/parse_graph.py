# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Callable, Iterable, Any

from .utils import segment_deviations, draw_graph

OrigGraphNode = Any
ParseGraphNode = tuple[int, Any]
SequenceItem = Any


def match_fn(node: OrigGraphNode, target: SequenceItem) -> bool:
    return node[1] == target


class ParseGraph(nx.DiGraph):
    def __init__(
        self,
        graph: nx.Graph,
        sequence: Iterable[SequenceItem],
        match_fn: Callable[[OrigGraphNode, SequenceItem], bool] = match_fn,
        prune: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.orig = graph
        self.seq = sequence
        self.match_fn = match_fn
        self._segments = None
        self._width = None
        self._positions = None
        self.build(prune=prune)

    def __repr__(self):
        return f"<ParseGraph of {self.orig.__class__.__name__}>"

    def __len__(self):
        return len(self.seq) + 2

    @property
    def start(self) -> ParseGraphNode:
        return (0, "START")

    @property
    def end(self) -> ParseGraphNode:
        return (len(self) - 1, "END")

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

    def filter(
        self,
        nodes: Iterable[ParseGraphNode],
        target: Any,
    ) -> list[ParseGraphNode]:
        """Return only those notes that match the target value using the match function."""
        return [node for node in nodes if self.match_fn(node, target)]

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

    def build(self, prune: bool = True):
        self.add_node(self.start, position=(0, 0), label="START")
        self.add_node(self.end, position=(len(self.seq) + 1, 0), label="END")

        # First step: from start to first matching nodes (in the original graph)
        nodes = self.filter(self.orig.nodes, target=self.seq[0])
        for i, node in enumerate(nodes):
            attrs = dict(**self.orig.nodes[node])
            attrs["position"] = (1, i)
            self.add_node((1, node), **attrs)
            self.add_edge(self.start, (1, node), weight=0)

        # Add intermediate nodes
        for pos in range(1, len(self.seq)):
            target = self.seq[pos]
            neighborCount = 0
            nextNodes = []
            for node in nodes:
                neighbors = self.filter(self.orig[node], target=target)
                if len(neighbors) > 0:
                    for neighbor in neighbors:
                        newNeighbor = (pos + 1, neighbor)
                        if newNeighbor not in self.nodes:
                            attrs = dict(**self.orig.nodes[neighbor])
                            attrs["position"] = (pos + 1, neighborCount)
                            self.add_node(newNeighbor, **attrs)
                            nextNodes.append(neighbor)
                            neighborCount += 1
                        weight = self.orig[node][neighbor]["weight"]
                        self.add_edge((pos, node), newNeighbor, weight=weight)
                elif prune:
                    self.prune_branch((pos, node))

            if len(nextNodes) == 0:
                raise ValueError(
                    f"Cannot unroll the graph for this sequence: no matching nodes at position {pos} for target {target}."
                )
            else:
                nodes = nextNodes

        # Last step: from last matching nodes to end
        for node in nodes:
            self.add_edge((len(self.seq), node), self.end, weight=0)

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
            plt.figure(figsize=(len(self.seq), self.width.max()))
        for segment in self.segments[1:]:
            plt.gca().axvline(segment["start"] - 0.5, color="k", lw=0.5, linestyle=":")
        draw_graph(self, **kws)
        if show_axes:
            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axis_on()
            ax.xaxis.grid(color=".9")
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            ax.set_xticks(range(len(self)))
            xlabels = ["start"] + self.seq + ["end"]
            ax.set_xticklabels([f"{i}\n{lab}" for i, lab in enumerate(xlabels)])
            ax.set_yticks(np.arange(0, self.width.max()))
            ax.set_yticklabels(np.arange(1, self.width.max() + 1, dtype=int))
            plt.ylabel("width")
            plt.ylim(-0.5, self.width.max() - 0.5)
        else:
            plt.axis("off")
