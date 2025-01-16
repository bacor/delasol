# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest

# Library imports
from music21.pitch import Pitch
import matplotlib.pyplot as plt
import networkx as nx

# Local imports
from delasol.utils.music import as_pitch_list
from delasol.styles import get_gamut
from delasol.graphs.hexachord_graph import HexachordGraph
from delasol.graphs.rollout_graph import RolloutGraph
from delasol.solmizers.continental_16c import Continental16cSolmizer


def hexachord_match_fn(node, target):
    return node == target


class TestRolloutGraph(unittest.TestCase):
    def test_hexachord(self):
        graph = HexachordGraph("C3")
        seq = "re mi fa mi re ut re".split(" ")
        match_fn = lambda node, target: graph.nodes[node]["syllable"] == target
        rollout = RolloutGraph(graph, seq, match_fn=match_fn)
        self.assertEqual(len(rollout), len(seq) + 2)

    def test_hexachord_width(self):
        graph = HexachordGraph("C3")
        match_fn = lambda node, target: graph.nodes[node]["syllable"] == target
        rollout = RolloutGraph(graph, ["re", "mi"], match_fn=match_fn)
        self.assertListEqual(rollout.width, [1, 1, 1, 1])

    def test_general_example(self):
        graph = nx.complete_graph(3)
        rollout = RolloutGraph(graph, [0, 2, 1, 0])
        self.assertEqual(len(rollout), 6)

    def test_circulant_graph(self):
        graph = nx.circulant_graph(6, [1])
        rollout = RolloutGraph(graph, [0, 3, 0])
        self.assertEqual(len(rollout), 9)
        self.assertListEqual(rollout.width, [1, 1, 2, 2, 1, 2, 2, 1, 1])

    def test_gamut_example(self):
        pass


class TestSolmizationGraph(unittest.TestCase):
    def test_search_musica_ficta(self):
        seq = as_pitch_list("C#4")
        sol = Continental16cSolmizer(seq, key=0)
        matches = sol.rollout.search_base(Pitch("C#4"))
        self.assertEqual(len(matches), 2)

    def test_shortest_paths_c_sharp(self):
        seq = as_pitch_list("C#4")
        sol = Continental16cSolmizer(seq, key=0)
        paths = sol.rollout.shortest_paths_base(Pitch("D4"), Pitch("C#4"))
        self.assertEqual(len(paths), 2)

    def test_basics(self):
        seq = as_pitch_list("C3 G3")
        sol = Continental16cSolmizer(seq, key=0)
        self.assertEqual(len(sol.rollout), 7)

    def test_search(self):
        seq = as_pitch_list("C3 G3")
        sol = Continental16cSolmizer(seq, key=0)
        matches = sol.rollout.search_base(Pitch("C3"))
        self.assertListEqual(
            matches, [(Pitch("G2"), Pitch("C3")), (Pitch("C3"), Pitch("C3"))]
        )

    def test_repetition(self):
        seq = as_pitch_list("C3 C3 C3")
        sol = Continental16cSolmizer(seq, key=0)
        self.assertEqual(len(sol.rollout), 5)

    @unittest.skip
    def test_timesteps(self):
        # TODO what's this?
        seq = as_pitch_list("C3 D3 E3 D3 C3")
        sol = Continental16cSolmizer(seq, key=0)

        self.assertEqual(sol.rollout.timesteps[0], [sol.rollout.start])
        self.assertEqual(sol.rollout.timesteps[len(sol.rollout) - 1], [sol.rollout.end])

    def test_node_attributes(self):
        H = HexachordGraph("C3")
        seq = as_pitch_list("C3 D3 E3 D3 C3")
        rollout = RolloutGraph(H, seq, match_fn=hexachord_match_fn)

        node1attrs = rollout.nodes[rollout.timesteps[1][0]]
        self.assertEqual(node1attrs["name"], "ut_C3")

        node2attrs = rollout.nodes[rollout.timesteps[2][0]]
        self.assertEqual(node2attrs["name"], "re_C3")

    def test_widths(self):
        seq = as_pitch_list("A2 B2 C3 D3 E3 F3 E3 D3 C3 B2 A2")
        sol = Continental16cSolmizer(seq, key=0)
        expected = [1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1]
        self.assertListEqual(sol.rollout.width, expected)

    def test_jumping(self):
        seq = as_pitch_list("F3 B-3 F3 B-3 F3")
        sol = Continental16cSolmizer(seq, key=-1)
        self.assertTrue(True)

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
