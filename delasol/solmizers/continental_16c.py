# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Library imports
from music21.pitch import Pitch

# Local imports
from delasol.graphs.gamut_graph import GamutGraph, register_gamut
from delasol.graphs.hexachord_graph import HexachordGraph
from delasol.graphs.rollout_graph import RolloutGraph
from delasol.pathfinders.simple_pathfinder import SimplePathfinder
from delasol.custom_types import GamutGraphNode
from delasol.solmizers.solmizer import Solmizer, register_solmizer

###############################################################################

CONTINENTAL_MUTATIONS = [
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


class HardContinental16CenturyGamutGraph(GamutGraph):
    name = "hard_continental_16c"

    def __init__(
        self,
        hexachords: t.Optional[list[HexachordGraph]] = None,
        mutations: t.Optional[dict] = CONTINENTAL_MUTATIONS,
        hexachord_kws: t.Optional[dict] = {},
        **kwargs,
    ):
        if hexachords is None:
            bases = ["G2", "C3", "G3", "C4", "G4", "C5"]
            hexachords = [HexachordGraph(base, **hexachord_kws) for base in bases]
        super().__init__(hexachords=hexachords, mutations=mutations, **kwargs)


register_gamut(HardContinental16CenturyGamutGraph)


class SoftContinental16CenturyGamutGraph(GamutGraph):
    name = "soft_continental_16c"

    def __init__(
        self,
        hexachords: t.Optional[list[HexachordGraph]] = None,
        mutations: t.Optional[dict] = CONTINENTAL_MUTATIONS,
        extend_below: bool = True,
        hexachord_kws: t.Optional[dict] = {},
        **kwargs,
    ):
        if hexachords is None:
            bases = ["F2", "C3", "F3", "C4", "F4"]
            if not extend_below:
                bases = bases[1:]
            hexachords = [HexachordGraph(base, **hexachord_kws) for base in bases]
        super().__init__(hexachords=hexachords, mutations=mutations, **kwargs)


register_gamut(SoftContinental16CenturyGamutGraph)


def match_diatonically(node: GamutGraphNode, target: Pitch) -> bool:
    """Check if a node matches a target pitch diatonically.

    Parameters
    ----------
    node : GamutGraphNode
        A node in the graph.
    target : Pitch
        The target pitch to match against.

    Returns
    -------
    bool
        True if the diatonic note numbers of the node and the target pitch match.
    """
    return node[1].diatonicNoteNum == target.diatonicNoteNum


###############################################################################


class Continental16cSolmizer(Solmizer):
    name = "continental_16c"

    def __init__(
        self,
        input,
        key: t.Optional[int] = None,
        mismatch_penalty: float = 0,
        **kws,
    ):
        super().__init__(input, key=key, mismatch_penalty=mismatch_penalty, **kws)

    def preprocess_input_and_opts(self, input, **kws):
        input, opts = super().preprocess_input_and_opts(input, **kws)

        if opts.get("key", None) is None:
            raise ValueError(
                "Please specify the 'key' or make sure the stream has a KeySignature."
            )
        elif opts.get("key") not in [0, -1]:
            raise ValueError(f"Unsupported key signature ({opts['key']}).")

        return input, opts

    def get_gamut_graph(self):
        if self.opts["key"] == -1:
            gamut = SoftContinental16CenturyGamutGraph()
        else:
            gamut = HardContinental16CenturyGamutGraph()
        return gamut

    def get_rollout_graph(self, gamut, pitches, **kws):
        rollout = RolloutGraph(gamut, pitches, match_fn=match_diatonically, **kws)

        # Adjust the rollout
        for time, target in zip(rollout.input_timesteps, rollout.sequence):
            for node in rollout.timesteps[time]:
                _, (_, pitch) = node
                if pitch != target:
                    for predecessor in rollout.predecessors(node):
                        rollout[predecessor][node]["weight"] += self.opts.get(
                            "mismatch_penalty"
                        )

        return rollout

    def get_pathfinder(self, rollout, **kws):
        return SimplePathfinder(rollout, **kws)


register_solmizer(Continental16cSolmizer)
