# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
from typing import Union
from music21.pitch import Pitch

PitchLike = Union[str, Pitch]
"""
A Pitch-like data input, can be either a string or a music21 Pitch object
"""

# Graph nodes

HexachordGraphNode = Pitch

GamutGraphNode = tuple[Pitch, HexachordGraphNode]
