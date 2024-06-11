# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
import os
from music21 import converter
from music21.pitch import Pitch
from music21.note import Note

# Local imports
from solmization.gamut_graph import HardContinentalGamut
from solmization.solmization import solmize, Solmization, StreamSolmization
from solmization.utils import as_stream

CUR_DIR = os.path.dirname(__file__)


class TestSolmization(unittest.TestCase):
    def test_init_empty(self):
        Solmization(gamut="hard-continental")
        self.assertTrue(True)

    def test_input_strings(self):
        sol = Solmization(gamut="hard-continental")
        input = "G3 A3 C4 B3 G3".split()
        pitches = sol._validate_input(input)
        targets = [Pitch(p) for p in input]
        self.assertEqual(pitches, targets)
        self.assertIsInstance(sol.gamut, HardContinentalGamut)

    def test_input_pitches(self):
        sol = Solmization(gamut="hard-continental")
        input = [Pitch(p) for p in "G3 A3 C4 B3 G3".split()]
        pitches = sol._validate_input(input)
        self.assertIsInstance(sol.gamut, HardContinentalGamut)

    def test_input_notes(self):
        sol = Solmization(gamut="hard-continental")
        seq = "G3 A3 C4 B3 G3".split()
        input = [Note(Pitch(p)) for p in seq]
        pitches = sol._validate_input(input)
        targets = [Pitch(p) for p in seq]
        self.assertEqual(pitches, targets)
        self.assertIsInstance(sol.gamut, HardContinentalGamut)

    def test_example_1(self):
        input = "G3 A3 C4 B3 G3".split()
        sol = Solmization(input, gamut="hard-continental")
        syllables = sol.best_syllables()
        self.assertEqual(syllables, ["ut₄", "re₄", "fa₄", "mi₄", "ut₄"])

    def test_iter_solmization(self):
        input = "G3 A3 C4 B3 G3".split()
        best_seq = ["ut₄", "re₄", "fa₄", "mi₄", "ut₄"]
        sol = Solmization(input, gamut="hard-continental")
        for i, data in enumerate(sol.iter_solmizations()):
            self.assertEqual(i, data["pos"])
            self.assertEqual(best_seq[i], data["syllables"][0])

    def test_iter_segments(self):
        input = "G3 A3 C4 B3 G3".split()
        targets = [["ut₄", "re₄"], ["fa₄"], ["mi₄", "ut₄"]]
        sol = Solmization(input, gamut="hard-continental")
        for target, segment in zip(targets, sol.iter_segments()):
            self.assertEqual(target, segment["syllables"][0])

    def test_ties(self):
        score = converter.parse(f"{CUR_DIR}/scores/issue-ties.mxl")
        solmization = StreamSolmization(score)
        evaluation = solmization.evaluate(target_lyrics=2)
        self.assertEqual(evaluation["correct"], 5)


class TestIssues(unittest.TestCase):
    def test_b_flat_in_hard_gamut(self):
        from music21.clef import TenorClef

        example = as_stream("D4 A3 B-3 A3")
        targets = "sol la fa la".split(" ")
        example.insert(0, TenorClef())
        solmization = solmize(example, gamut="hard-continental", prune_parse=False)
        solmization.annotate(targets=targets)
        # example.show()

    def test_c_sharp_hard_gamut(self):
        from music21.clef import TenorClef

        example = as_stream("C5 D5 C#5 B4 C#5 D5")
        solmization = solmize(example, gamut="hard-continental", prune_parse=False)
        # solmization.annotate(targets=targets)
        # example.show()
        print(solmization)
