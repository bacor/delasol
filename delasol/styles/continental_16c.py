# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Libraries

# Local imports
from delasol.gamut_graph import GamutGraph
from delasol.hexachord_graph import HexachordGraph


# Mutations
CONTINENTAL_MUTATIONS = {
    "natural": {
        "up": {
            "hard": [(5, 2)],
            "soft": [(4, 2)],
        },
        "down": {
            "hard": [(4, 6)],
            "soft": [(3, 6)],
        },
    },
    "hard": {
        "up": {
            "natural": [(4, 2)],
        },
        "down": {
            "natural": [(3, 6)],
        },
    },
    "soft": {
        "up": {
            "natural": [(5, 2)],
        },
        "down": {
            "natural": [(4, 6)],
        },
    },
}


class HardContinental16CenturyGamutGraph(GamutGraph):
    def __init__(
        self,
        hexachords: t.Optional[list[HexachordGraph]] = None,
        mutations: t.Optional[dict] = CONTINENTAL_MUTATIONS,
        hexachord_kws: t.Optional[dict] = {},
        **kwargs,
    ):
        if hexachords is None:
            hexachords = [
                HexachordGraph("G2", **hexachord_kws),
                HexachordGraph("C3", **hexachord_kws),
                HexachordGraph("G3", **hexachord_kws),
                HexachordGraph("C4", **hexachord_kws),
                HexachordGraph("G4", **hexachord_kws),
                HexachordGraph("C5", **hexachord_kws),
            ]
        super().__init__(hexachords=hexachords, mutations=mutations, **kwargs)


class SoftContinental16CenturyGamutGraph(GamutGraph):
    def __init__(
        self,
        hexachords: t.Optional[list[HexachordGraph]] = None,
        mutations: t.Optional[dict] = CONTINENTAL_MUTATIONS,
        extend_below: bool = True,
        hexachord_kws: t.Optional[dict] = {},
        **kwargs,
    ):
        if hexachords is None:
            hexachords = [
                HexachordGraph("C3", **hexachord_kws),
                HexachordGraph("F3", **hexachord_kws),
                HexachordGraph("C4", **hexachord_kws),
                HexachordGraph("F4", **hexachord_kws),
            ]
        if extend_below:
            hexachords = [HexachordGraph("F2", **hexachord_kws)] + hexachords
        super().__init__(hexachords=hexachords, mutations=mutations, **kwargs)
