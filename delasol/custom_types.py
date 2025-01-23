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
"""A pitch-like object, either a pitch-string or a :class:`music21.pitch.Pitch` object."""

# Hexachord graph types

HexachordGraphNode = t.TypeVar("HexachordGraphNode")
"""Node in a hexachord graph, represented by a :class:`music21.pitch.Pitch` object."""
# Pitch

# Gamut graph types

GamutGraphNode = t.TypeVar("GamutGraphNode")
"""Node in a gamut graph. This is a pair :py:`(base, pitch)` where :py:`base` is a :class:`music21.pitch.Pitch` object representing the base of a hexachord, and :py:`pitch` another pitch object, representing the pitch in that hexachord."""
# t.Tuple[Pitch, HexachordGraphNode]

GamutGraphPath = t.TypeVar("GamutGraphNode")
"""A path through the gamut graph, so a list of :class:`GamutGraphNode` objects."""
# t.List[GamutGraphNode]

# Rollout graph types

SeqType = t.TypeVar("SeqType")
"""The data type of the sequence that is used to make the rollout graph.
In this case, these are typically :class:`music21.pitch.Pitch` objects. """

BaseGraphNode = t.TypeVar("BaseGraphNode")
"""The type for the nodes in the base graph, from which a rollout graph is constructed.
Here, these will typically be :class:`GamutGraphNode` objects."""

BaseGraphPath = list[BaseGraphNode]
"""A path through a base graph, so a list of :class:`BaseGraphNode` objects."""

RolloutGraphNode = t.TypeVar("RolloutGraphNode")
"""A node in the rollout graph. This is a pair :py:`(time, node)` where :py:`time` is 
an integer, and :py:`time` can be either a :class:`BaseGraphNode`, :py:`"START"`, or :py:`"END"`."""
# t.TypeVar[t.Tuple[int, BaseGraphNode | t.Literal["START"] | t.Literal["END"]]]

RolloutGraphPath = t.TypeVar("RolloutGraphPath")
"""A path in the rollout graph, so a list of :class:`RolloutGraphNode` objects."""
# list[RolloutGraphNode]
