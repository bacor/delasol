# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Libraries
from music21.key import KeySignature

# Local imports
from delasol.gamut_graph import GamutGraph
from delasol.styles.continental_16c import (
    HardContinental16CenturyGamutGraph,
    SoftContinental16CenturyGamutGraph,
)
from delasol.styles.english_16c import (
    HardEnglish16CenturyGamutGraph,
    SoftEnglish16CenturyGamutGraph,
)

GAMUTS = {
    "hard-continental": HardContinental16CenturyGamutGraph,
    "soft-continental": SoftContinental16CenturyGamutGraph,
    "hard-english": HardEnglish16CenturyGamutGraph,
    "soft-english": SoftEnglish16CenturyGamutGraph,
}

SHARPS_TO_GAMUT_NAME = {
    "continental": {
        0: "hard-continental",
        -1: "soft-continental",
    },
    "english": {
        0: "hard-english",
        -1: "soft-english",
    },
}


def get_gamut(
    name: t.Optional[str] = None,
    style: t.Optional[str] = None,
    sharps: t.Optional[int] = None,
    key: t.Optional[KeySignature] = None,
    **kws,
) -> GamutGraph:
    """Returns a gamut by its name, the style or the number of sharps"""
    if key is not None:
        sharps = key.sharps

    if sharps is not None:
        if style is None:
            raise ValueError(
                "No solmization style specified. This is required if you provide a key signature or the number of sharps"
            )
        if style not in SHARPS_TO_GAMUT_NAME:
            raise ValueError("Invalid style {style}")
        if sharps not in SHARPS_TO_GAMUT_NAME[style]:
            raise ValueError(
                f"Number of sharps ({sharps}) is not supported for style {style}."
            )
        name = SHARPS_TO_GAMUT_NAME[style][sharps]

    if name not in GAMUTS:
        raise ValueError(
            f"Invalid gamut name '{name}'. Suppored names are: {', '.join(GAMUTS.keys())}"
        )
    else:
        return GAMUTS[name](**kws)
