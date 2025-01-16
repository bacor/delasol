# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Libraries
from music21.pitch import Pitch


SYLLABLES = ["ut", "re", "mi", "fa", "sol", "la", "fa"]
"""
list of str: The syllables used to name the notes in a hexachord.
"""


INTERNAL_SYLLABLES = ["ut", "re", "mi", "fa", "sol", "la", "fi"]
"""
list of str: The syllables used to internally name the nodes in a hexachord graph.
"""

# Conventional hexachords numbering
octaves = range(0, 10)
base_notes = ["C", "F", "G"]
G2_index = len(base_notes) * (2 - octaves[0]) + base_notes.index("G")

HEXACHORD_NUMBERING = {
    Pitch(f"{note}{octave}"): i - G2_index + 1
    for i, (note, octave) in enumerate(
        (note, octave) for octave in octaves for note in base_notes
    )
}
"""
dict[Pitch, int]: Conventional hexachords numbering so that F2 is 0, G2 is 1, 
etc. The numbering is extended for down and up beyond the conventional 
numbering so that e.g. C2 is -1 and C5 is 8.
"""


UNICODE_SUBSCRIPTS = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
"""
list of str: Unicode subscripts, can be used to represent the hexachord numbering.
"""
