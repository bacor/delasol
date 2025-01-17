# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
import matplotlib
import matplotlib.pyplot as plt

# Local imports
from delasol.utils.music import as_pitch_list
from delasol.graphs.hexachord_graph import HexachordGraph
from delasol.graphs.gamut_graph import GamutGraph
from delasol.utils.sequence import segment_deviations

# Set matplotlib backend
matplotlib.use("MacOSX")


class TestSegmentation(unittest.TestCase):
    def test_constant_sequence(self):
        for length in range(1, 10):
            example = [1] * length
            segments = segment_deviations(example, 1)
            self.assertEqual(segments, [(0, length - 1)])

    def test_constant_tail(self):
        example = [1, 2, 1, 1, 1, 1]
        segments = segment_deviations(example, 1)
        self.assertEqual(segments, [(0, 2), (3, 5)])

    def test_one_segment(self):
        example = [1, 2, 1]
        segments = segment_deviations(example, 1)
        self.assertEqual(segments, [(0, 2)])

    def test_no_matches(self):
        example = [2, 2, 2]
        segments = segment_deviations(example, 1)
        self.assertEqual(segments, [(0, 2)])

    def test_deviant_opening_and_end(self):
        example = [2, 2, 1, 1, 2]
        segments = segment_deviations(example, 1)
        self.assertEqual(segments, [(0, 2), (3, 4)])

    def test_issue_1(self):
        issue = [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1]
        segments = segment_deviations(issue, 1)
        self.assertEqual(segments, [(0, 3), (4, 7), (8, 11)])


class TestDrawing(unittest.TestCase):
    @unittest.skip
    def test_draw_hexachord(self):
        H = HexachordGraph("G2")
        H.draw()
        plt.show()

    @unittest.skip
    def test_draw_gamut(self):
        G = GamutGraph(["G2", "C3", "G3", "C4"])
        G.add_edge_by_names("fa_G2", "re_C3")
        G.add_edge_by_names("fa_C3", "la_G2")
        G.draw()
        plt.show()

    @unittest.skip
    def test_draw_solmization_graph(self):
        # TODO no longer needed?
        gamut = GamutGraph(["G2", "C3", "G3"], mutations=CONTINENTAL_MUTATIONS)
        pitches = as_pitch_list("A2 B2 C3 D3")
        sol = SolmizationGraph(gamut, pitches)
        sol.draw()
        plt.show()
