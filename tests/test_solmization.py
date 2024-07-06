# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
import os
from music21 import converter
from music21.stream import Stream
from music21.pitch import Pitch
from music21.note import Note

# Local imports
from alamire.gamut_graph import HardContinentalGamut
from alamire.solmization import solmize, Solmization, StreamSolmization
from alamire.utils import as_stream

CUR_DIR = os.path.dirname(__file__)


class TestSolmization(unittest.TestCase):
    def test_init_empty(self):
        with self.assertRaises(ValueError):
            Solmization(gamut="hard-continental")

    def test_input_strings(self):
        input = "G3 A3 C4 B3 G3".split()
        sol = Solmization(input, gamut="hard-continental")
        targets = [Pitch(p) for p in input]
        self.assertEqual(sol.pitches, targets)
        self.assertIsInstance(sol.gamut, HardContinentalGamut)

    def test_input_pitches(self):
        input = [Pitch(p) for p in "G3 A3 C4 B3 G3".split()]
        sol = Solmization(input, gamut="hard-continental")
        self.assertEqual(sol.pitches, input)

    def test_input_notes(self):
        seq = "G3 A3 C4 B3 G3".split()
        input = [Note(Pitch(p)) for p in seq]
        sol = Solmization(input, gamut="hard-continental")
        targets = [Pitch(p) for p in seq]
        self.assertEqual(sol.pitches, targets)
        self.assertIsInstance(sol.gamut, HardContinentalGamut)

    def test_example_1(self):
        input = "G3 A3 C4 B3 G3".split()
        sol = Solmization(input, gamut="hard-continental")
        self.assertEqual(
            sol.output(style="state-subscript"), ["ut₄", "re₄", "fa₄", "mi₄", "ut₄"]
        )

    def test_output(self):
        input = "G3 A3 B3 C4 D4 E4 F4".split()
        sol = Solmization(input, gamut="hard-continental")
        self.assertListEqual(
            sol.output(style="syllable"), "ut re mi fa sol la fa".split(" ")
        )
        self.assertListEqual(
            sol.output(style="state"), "ut4 re4 mi4 fa4 sol4 la4 fi4".split(" ")
        )
        self.assertListEqual(
            sol.output(style="state-subscript"),
            ["ut₄", "re₄", "mi₄", "fa₄", "sol₄", "la₄", "fi₄"],
        )
        self.assertListEqual(
            sol.output(style="state-syllable"), "ut re mi fa sol la fi".split(" ")
        )
        self.assertListEqual(
            sol.output(style="do-ti"), "do re mi fa sol la ti".split(" ")
        )

    def test_custom_output(self):
        input = "G3 A3 B3".split()
        sol = Solmization(input, gamut="hard-continental")
        formatter = lambda pitch, **args: pitch.nameWithOctave
        self.assertListEqual(sol.output(style=formatter), "G3 A3 B3".split())

    def test_annotation(self):
        pitches = "G3 A3 B3 A3 C4".split(" ")
        syllables = "ut re mi re fa".split(" ")
        stream = Stream()
        for pitch, syll in zip(pitches, syllables):
            note = Note(pitch)
            stream.append(note)
            note.lyric = syll
        solmization = StreamSolmization(stream, gamut="hard-continental")
        solmization.annotate(target_lyrics=1, show_weights=False)
        targets = [f"{syll}₄" for syll in syllables]
        for note, target in zip(stream.flat.notes, targets):
            lyrics = {lyric.number: lyric for lyric in note.lyrics}
            self.assertEqual(lyrics[2].text, target)

    def test_annotation_no_targets(self):
        pitches = "G3 A3 B3 A3 C4".split(" ")
        stream = Stream([Note(p) for p in pitches])
        solmization = StreamSolmization(stream, gamut="hard-continental")
        solmization.annotate(
            show_weights=False, output_style="syllable", best_only=True, offset=1
        )
        syllables = "ut re mi re fa".split(" ")
        for note, target in zip(stream.flat.notes, syllables):
            lyrics = {lyric.number: lyric for lyric in note.lyrics}
            self.assertEqual(lyrics[2].text, target)


class TestIssues(unittest.TestCase):
    def test_ties(self):
        score = converter.parse(f"{CUR_DIR}/scores/issue-ties.mxl")
        solmization = StreamSolmization(score)
        targets = ["la", "sol", "fa", "mi", "re"]
        self.assertListEqual(solmization.output(), targets)
        evaluation = solmization.evaluate(target_lyrics=2)
        self.assertEqual(evaluation["correct"], 5)

    def test_b_flat_in_hard_gamut(self):
        from music21.clef import TenorClef

        example = as_stream("D4 A3 B-3 A3")
        targets = "sol la fa la".split(" ")
        example.insert(0, TenorClef())
        solmization = solmize(example, gamut="hard-continental")
        self.assertListEqual(solmization.output(), targets)

    def test_c_sharp_hard_gamut(self):
        example = as_stream("C5 D5 C#5 B4 C#5 D5")
        targets = ["fa", "sol", "fa", "mi", "fa", "sol"]
        solmization = solmize(example, gamut="hard-continental")
        self.assertEqual(solmization.output(), targets)

    def test_files(self):
        score_1 = converter.parse(f"{CUR_DIR}/scores/MdB001.musicxml")
        solmization = solmize(score_1, style="continental")
        solmization.annotate(target_lyrics=2)

        score_2 = converter.parse(f"{CUR_DIR}/scores/MdB004.musicxml")
        solmization = solmize(score_2, style="continental")
        solmization.annotate(target_lyrics=2)
        self.assertTrue(True)
