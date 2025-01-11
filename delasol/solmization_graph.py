# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Libraries
from music21.pitch import Pitch

# Local imports
from delasol.gamut_graph import GamutGraph
from delasol.parse_graph import ParseGraph

OrigGraphNode = t.Any


class SolmizationGraph(ParseGraph):
    def __init__(
        self,
        gamut: GamutGraph,
        pitches: t.Iterable[Pitch],
        mismatch_penalty: float = 0,
        match_fn: t.Any = None,
        **kwargs,
    ):
        if not isinstance(gamut, GamutGraph):
            raise ValueError("The graph should be a GamutGraph.")
        if pitches is not None and not isinstance(pitches[0], Pitch):
            raise ValueError("The sequence should be a list of pitches.")
        if match_fn is not None:
            raise Warning("The match function is ignored for SolmizationGraph.")

        self.gamut = gamut
        self.mismatch_penalty = mismatch_penalty
        super().__init__(graph=gamut, sequence=pitches, **kwargs)

    def search(
        self, target: Pitch, nodes: t.Iterable[OrigGraphNode] | None = None
    ) -> list[tuple[OrigGraphNode, dict]]:
        """Search for nodes with a matching pitch"""
        if nodes is None:
            nodes = self.orig.nodes
        return [n for n in nodes if n[1].diatonicNoteNum == target.diatonicNoteNum]

    def build(self, sequence: t.Iterable[Pitch], prune: bool = True):
        super().build(sequence, prune=prune)

        # Add a mismatch penalty to all nodes that do not exactly match the target pitch
        for pos, target in zip(self.input_positions, sequence):
            for node in self.positions[pos]:
                _, (_, pitch) = node
                if pitch != target:
                    for predecessor in self.predecessors(node):
                        self[predecessor][node]["weight"] += self.mismatch_penalty
