# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
from music21.pitch import Pitch

# Local imports
from delasol.utils import as_pitch_list
from delasol.styles import get_gamut
from delasol.styles import HardContinental16CenturyGamut as HardContinentalGamut
from delasol.solmization_graph import SolmizationGraph


class TestSolmizationGraph(unittest.TestCase):
    def test_search_musica_ficta(self):
        gamut = get_gamut("hard-continental")
        pg = SolmizationGraph(gamut)
        matches = pg.search(Pitch("C#4"))
        self.assertEqual(len(matches), 2)

    def test_shortest_paths_c_sharp(self):
        gamut = get_gamut("hard-continental")
        pg = SolmizationGraph(gamut)
        paths = pg.shortest_paths(Pitch("D4"), Pitch("C#4"))
        self.assertEqual(len(paths), 2)

    # def test_input_segments(self):
    #     seq = [Pitch(p) for p in "C3 G3 C4 F3 A3".split(" ")]
    #     gamut = get_gamut("hard-continental")
    #     pg = SolmizationGraph(gamut, seq)
    #     for step in pg.iter_best_path():
    #         print(step)
    #     print(pg)

    # TODO should work, but the order nof nodes may be randomized?
    # def test_b_flat_in_hard_hex(self):
    #     example = [Pitch(p) for p in "D4 A3 B-3 A3 G3 F3".split(" ")]
    #     gamut = get_gamut("hard-continental")
    #     pg = SolmizationGraph(gamut, example, mismatch_penalty=10)
    #     node_a = pg.positions[4][1]
    #     node_b_flat = pg.positions[5][0]
    #     self.assertGreater(pg[node_a][node_b_flat]["weight"], 10)

    # def test_mismatch_penalty(self):
    #     gamut = HardContinentalGamut()
    #     seq = as_pitch_list("A3 B-3 A3")
    #     sg = SolmizationGraph(gamut, seq, mismatch_penalty=2)
    #     node1 = (1, gamut.names["la_C3"])
    #     node2 = (2, gamut.names["fi_C3"])
    #     node3 = (3, gamut.names["la_C3"])
    #     self.assertEqual(sg[node1][node2]["weight"], 1.5)
    #     self.assertEqual(sg[node2][node3]["weight"], 1)
