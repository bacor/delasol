# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Local imports
from delasol.graphs.gamut_graph import GamutGraph, register_gamut
from delasol.graphs.hexachord_graph import HexachordGraph
from delasol.solmizers.continental_16c import match_diatonically
from delasol.solmizers.solmizer import Solmizer, register_solmizer
from delasol.graphs.rollout_graph import RolloutGraph
from delasol.pathfinders.simple_pathfinder import SimplePathfinder
from delasol.custom_types import GamutGraphNode

ENGLISH_MUTATIONS = [
    dict(source="natural", dir="up", target="hard", moves=[("la", "mi")]),
    dict(source="natural", dir="up", target="soft", moves=[("la", "fa")]),
    dict(source="natural", dir="down", target="hard", moves=[("fa", "la")]),
    dict(source="natural", dir="down", target="soft", moves=[("mi", "la")]),
    dict(source="hard", dir="up", target="natural", moves=[("la", "fa")]),
    dict(source="hard", dir="down", target="natural", moves=[("mi", "la")]),
    dict(source="soft", dir="up", target="natural", moves=[("la", "mi")]),
    dict(source="soft", dir="down", target="natural", moves=[("fa", "la")]),
]


class HardEnglish16CenturyGamutGraph(GamutGraph):
    name = "hard_english_16c"

    def __init__(
        self,
        hexachords: t.Optional[t.Iterable[HexachordGraph]] = None,
        mutations: t.Optional[dict] = ENGLISH_MUTATIONS,
        mutation_weight: t.Optional[float] = 0.75,
        hexachord_kws={},
        **kwargs,
    ):
        kws = dict(
            fa_super_la=False,
            loop_weight=0.5,
            step_weight=1,
            fa_super_la_weight=1,
        )
        kws.update(**hexachord_kws)
        if hexachords is None:
            bases = ["G2", "C3", "G3", "C4", "G4"]
            hexachords = [HexachordGraph(base, **kws) for base in bases]

        super().__init__(
            hexachords=hexachords,
            mutations=mutations,
            mutation_weight=mutation_weight,
            **kwargs,
        )


register_gamut(HardEnglish16CenturyGamutGraph)


class SoftEnglish16CenturyGamutGraph(GamutGraph):
    name = "soft_english_16c"

    def __init__(
        self,
        hexachords: t.Optional[t.Iterable[HexachordGraph]] = None,
        mutations: t.Optional[dict] = ENGLISH_MUTATIONS,
        mutation_weight: t.Optional[float] = 0.75,
        hexachord_kws={},
        **kwargs,
    ):
        kws = dict(
            fa_super_la=False,
            loop_weight=0.5,
            step_weight=1,
            fa_super_la_weight=1,
        )
        kws.update(**hexachord_kws)
        if hexachords is None:
            bases = ["C3", "F3", "C4", "F4"]
            hexachords = [HexachordGraph(base, **kws) for base in bases]
        super().__init__(
            hexachords=hexachords,
            mutations=mutations,
            mutation_weight=mutation_weight,
            **kwargs,
        )


register_gamut(SoftEnglish16CenturyGamutGraph)


###############################################################################


class English16cSolmizer(Solmizer):
    name = "english_16c"

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
            gamut = SoftEnglish16CenturyGamutGraph()
        else:
            gamut = HardEnglish16CenturyGamutGraph()
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


register_solmizer(English16cSolmizer)
