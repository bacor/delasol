# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Local imports
from delasol.gamut_graph import GamutGraph
from delasol.hexachord_graph import HexachordGraph

ENGLISH_MUTATIONS = {
    "natural": {
        "up": {
            "hard": [(6, 3)],
            "soft": [(6, 4)],
        },
        "down": {
            "hard": [(4, 6)],
            "soft": [(3, 6)],
        },
    },
    "hard": {
        "up": {
            "natural": [(6, 4)],
        },
        "down": {
            "natural": [(3, 6)],
        },
    },
    "soft": {
        "up": {
            "natural": [(6, 3)],
        },
        "down": {
            "natural": [(4, 6)],
        },
    },
}


class HardEnglish16CenturyGamutGraph(GamutGraph):
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
            hexachords = [
                HexachordGraph("G2", **kws),
                HexachordGraph("C3", **kws),
                HexachordGraph("G3", **kws),
                HexachordGraph("C4", **kws),
                HexachordGraph("G4", **kws),
            ]
        super().__init__(
            hexachords=hexachords,
            mutations=mutations,
            mutation_weight=mutation_weight,
            **kwargs,
        )


class SoftEnglish16CenturyGamutGraph(GamutGraph):
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
            hexachords = [
                HexachordGraph("C3", **kws),
                HexachordGraph("F3", **kws),
                HexachordGraph("C4", **kws),
                HexachordGraph("F4", **kws),
            ]
        super().__init__(
            hexachords=hexachords,
            mutations=mutations,
            mutation_weight=mutation_weight,
            **kwargs,
        )
