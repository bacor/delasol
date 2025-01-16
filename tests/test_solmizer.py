# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
import os
from music21 import converter
from music21.stream import Stream
from music21.note import Note
from music21.key import KeySignature

# Local imports
from delasol.utils.music import as_pitch_list, as_stream
from delasol.solmizers.solmizer import solmize
from delasol.solmizers.continental_16c import Continental16cSolmizer
from delasol.solmizers.continental_16c import HardContinental16CenturyGamutGraph
from delasol.evaluate import EvalResult
from delasol.utils import as_stream

CUR_DIR = os.path.dirname(__file__)


class TestContinental16cSolmizer(unittest.TestCase):
    def test_init_empty(self):
        with self.assertRaises(TypeError):
            Continental16cSolmizer()

    def test_input_strings(self):
        input = "G3 A3 C4 B3 G3"
        sol = Continental16cSolmizer(input, key=0)
        targets = as_pitch_list(input)
        self.assertEqual(sol.pitches, targets)
        self.assertIsInstance(sol.gamut, HardContinental16CenturyGamutGraph)

    def test_input_pitches(self):
        input = as_pitch_list("G3 A3 C4 B3 G3")
        sol = Continental16cSolmizer(input, key=0)
        self.assertEqual(sol.pitches, input)

    def test_input_notes(self):
        pitches = as_pitch_list("G3 A3 C4 B3 G3")
        input = [Note(p) for p in pitches]
        sol = Continental16cSolmizer(input, key=0)
        self.assertEqual(sol.pitches, pitches)

    def test_solmize(self):
        sol = Continental16cSolmizer("G3 A3 C4 B3 G3", key=0)
        self.assertEqual(sol.solmize(), ["ut", "re", "fa", "mi", "ut"])

    def test_output(self):
        sol = Continental16cSolmizer("G3 A3 B3 C4 D4 E4 F4", key=0)

        self.assertListEqual(sol.solmize(), "ut re mi fa sol la fa".split(" "))

        self.assertListEqual(
            sol.solmize(format="name"),
            "ut_G3 re_G3 mi_G3 fa_G3 sol_G3 la_G3 fi_G3".split(" "),
        )

        self.assertListEqual(
            sol.solmize(format="syllable_hexnum", subscript=False),
            "ut4 re4 mi4 fa4 sol4 la4 fa4".split(" "),
        )

        self.assertListEqual(
            sol.solmize(format="syllable_hexnum", subscript=True),
            ["ut₄", "re₄", "mi₄", "fa₄", "sol₄", "la₄", "fa₄"],
        )

        self.assertListEqual(
            sol.solmize(format="modern_syllable"), "do re mi fa sol la ti".split(" ")
        )

    def test_stream(self):
        stream = as_stream("G3 A3 B-3 A3 C4")

        # No KeySignature: raises NO error
        sol = Continental16cSolmizer(stream)

        # Specify b_flats
        sol = Continental16cSolmizer(stream, key=-1)
        self.assertEqual(sol.opts.get("key"), -1)

        # Insert a key signature
        key = KeySignature(-1)
        stream.insert(0, key)
        sol = Continental16cSolmizer(stream)
        self.assertEqual(sol.opts.get("key"), -1)

        # Test solmization
        self.assertListEqual(sol.solmize(), "re mi fa mi sol".split(" "))

    # TODO fix!
    def test_annotation(self):
        pitches = "G3 A3 B3 A3 C4".split(" ")
        syllables = "ut re mi re fa".split(" ")
        stream = Stream()
        for pitch, syll in zip(pitches, syllables):
            note = Note(pitch)
            stream.append(note)
            note.lyric = syll
        solmization = Continental16cSolmizer(stream)
        solmization.annotate(
            target_lyric_number=1, format="syllable_hexnum", subscript=True
        )  # , show_weights=False)
        targets = [f"{syll}₄" for syll in syllables]
        for note, target in zip(stream.flat.notes, targets):
            lyrics = {lyric.number: lyric for lyric in note.lyrics}
            self.assertEqual(lyrics[2].text, target)

    def test_annotation_no_targets(self):
        stream = as_stream("G3 A3 B3 A3 C4")
        solmization = Continental16cSolmizer(stream)
        solmization.annotate(offset=1)
        syllables = "ut re mi re fa".split(" ")
        for note, target in zip(stream.flat.notes, syllables):
            lyrics = {lyric.number: lyric for lyric in note.lyrics}
            self.assertEqual(lyrics[2].text, target)


class TestIssues(unittest.TestCase):

    def test_ties(self):
        score = converter.parse(f"{CUR_DIR}/scores/issue-ties.mxl")
        solmization = solmize(score, style="continental_16c")
        targets = ["la", "sol", "fa", "mi", "re"]
        self.assertListEqual(solmization.solmize(), targets)
        results = solmization.evaluate(target_lyric_number=2, return_counts=True)
        self.assertEqual(results[EvalResult.CORRECT], 5)

    # TODO this is a problem
    @unittest.skip
    def test_b_flat_in_hard_gamut(self):
        example = as_stream("D4 A3 B-3 A3")
        targets = "sol la fa la".split(" ")
        solmization = solmize(example, style="continental_16c", key=0)

        # TODO this shouldn't happen?
        self.assertListEqual(solmization.solmize(rank=2), targets)

        self.assertListEqual(solmization.solmize(), targets)

    def test_c_sharp_hard_gamut(self):
        example = as_stream("C5 D5 C#5 B4 C#5 D5")
        targets = ["fa", "sol", "fa", "mi", "fa", "sol"]
        solmization = solmize(example, style="continental_16c", key=0)
        self.assertEqual(solmization.solmize(), targets)

    def test_evaluate_MdB001(self):
        score = converter.parse(f"{CUR_DIR}/scores/MdB001.musicxml")
        solmization = solmize(score, style="continental_16c")
        results = solmization.evaluate(target_lyric_number=2, return_counts=True)
        self.assertEqual(results[EvalResult.CORRECT], 62)

    def test_evaluate_MdB004(self):
        score = converter.parse(f"{CUR_DIR}/scores/MdB004.musicxml")
        solmization = solmize(score, style="continental_16c")
        results = solmization.evaluate(target_lyric_number=2, return_counts=True)
        self.assertEqual(results[EvalResult.CORRECT], 86)
