# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Library imports
from music21.pitch import Pitch

# General
PitchLike = t.Union[str, Pitch]

# Hexachord graph types
HexachordGraphNode = Pitch

# Gamut graph types
GamutGraphNode = tuple[Pitch, HexachordGraphNode]
GamutGraphPath = list[GamutGraphNode]

# Rollout graph types
SeqType = t.TypeVar("SeqType")
BaseGraphNode = t.TypeVar("BaseNode")
BaseGraphPath = list[BaseGraphNode]
RolloutGraphNode = t.Tuple[int, BaseGraphNode | t.Literal["START"] | t.Literal["END"]]
RolloutGraphPath = list[RolloutGraphNode]

# To be removed
OrigGraphNode = t.TypeVar("OrigGraphNode")
ParseGraphNode = tuple[int, OrigGraphNode]
SequenceItem = t.TypeVar("SequenceItem")
Path = t.Iterable[ParseGraphNode]
