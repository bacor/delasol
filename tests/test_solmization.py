import unittest
from music21.pitch import Pitch
from music21.note import Note
from hexachord.gamut_graph import HardGamutGraph
from hexachord.solmization import Solmization


class TestSolmization(unittest.TestCase):
    def test_init_empty(self):
        Solmization()
        self.assertTrue(True)

    def test_input_strings(self):
        sol = Solmization()
        input = "G3 A3 C4 B3 G3".split()
        pitches, gamut = sol._validate_input(input, gamut="hard")
        targets = [Pitch(p) for p in input]
        self.assertEqual(pitches, targets)
        self.assertIsInstance(gamut, HardGamutGraph)

    def test_input_pitches(self):
        sol = Solmization()
        input = [Pitch(p) for p in "G3 A3 C4 B3 G3".split()]
        pitches, gamut = sol._validate_input(input, gamut="hard")
        self.assertIsInstance(gamut, HardGamutGraph)

    def test_input_notes(self):
        sol = Solmization()
        seq = "G3 A3 C4 B3 G3".split()
        input = [Note(Pitch(p)) for p in seq]
        pitches, gamut = sol._validate_input(input, gamut="hard")
        targets = [Pitch(p) for p in seq]
        self.assertEqual(pitches, targets)
        self.assertIsInstance(gamut, HardGamutGraph)

    def test_example_1(self):
        input = "G3 A3 C4 B3 G3".split()
        sol = Solmization(input, gamut="hard")
        syllables = sol.best_syllables()
        self.assertEqual(syllables, ["ut₄", "re₄", "fa₄", "mi₄", "ut₄"])

    def test_iter_solmization(self):
        input = "G3 A3 C4 B3 G3".split()
        best_seq = ["ut₄", "re₄", "fa₄", "mi₄", "ut₄"]
        sol = Solmization(input, gamut="hard")
        for i, data in enumerate(sol.iter_solmizations()):
            self.assertEqual(i, data["pos"])
            self.assertEqual(best_seq[i], data["syllables"][0])

    def test_iter_segments(self):
        input = "G3 A3 C4 B3 G3".split()
        targets = [["ut₄", "re₄"], ["fa₄"], ["mi₄", "ut₄"]]
        sol = Solmization(input, gamut="hard")
        for target, segment in zip(targets, sol.iter_segments()):
            self.assertEqual(target, segment["syllables"][0])