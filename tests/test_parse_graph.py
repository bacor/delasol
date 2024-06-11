# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
from music21.pitch import Pitch

# Local imports
from solmization.gamut_graph import HardContinentalGamut, HexachordGraph, get_gamut
from solmization.parse_graph import ParseGraph, GamutParseGraph


def hexachord_match_fn(node, target):
    return node == target


class TestParseGraph(unittest.TestCase):
    def test_basics(self):
        gamut = HardContinentalGamut()
        seq = [Pitch(p) for p in "C3 G3".split(" ")]
        pg = ParseGraph(gamut, seq)
        self.assertEqual(len(pg), 7)

    def test_repetition(self):
        gamut = HardContinentalGamut()
        seq = [Pitch(p) for p in "C3 C3 C3".split(" ")]
        pg = ParseGraph(gamut, seq)
        self.assertEqual(len(pg), 5)

    def test_positions(self):
        H = HexachordGraph("C3")
        seq = [Pitch(p) for p in "C3 D3 E3 D3 C3".split(" ")]
        parse = ParseGraph(H, seq, match_fn=hexachord_match_fn)

        self.assertEqual(parse.positions[0], [parse.start])
        self.assertEqual(parse.positions[len(parse) - 1], [parse.end])
        for i in range(len(seq)):
            self.assertEqual(parse.positions[i + 1], [(i + 1, seq[i])])

    def test_node_attributes(self):
        H = HexachordGraph("C3")
        seq = [Pitch(p) for p in "C3 D3 E3 D3 C3".split(" ")]
        parse = ParseGraph(H, seq, match_fn=hexachord_match_fn)

        node1attrs = parse.nodes[parse.positions[1][0]]
        self.assertEqual(node1attrs["position"], (1, 0))
        self.assertEqual(node1attrs["name"], "ut2")

        node2attrs = parse.nodes[parse.positions[2][0]]
        self.assertEqual(node2attrs["position"], (2, 0))
        self.assertEqual(node2attrs["name"], "re2")

    def test_widths(self):
        gamut = HardContinentalGamut()
        seq = [Pitch(p) for p in "A2 B2 C3 D3 E3 F3 E3 D3 C3 B2 A2".split(" ")]
        parse = ParseGraph(gamut, seq)
        expected = [1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1]
        self.assertListEqual(parse.width.tolist(), expected)

    def test_jumping(self):
        gamut = get_gamut("soft-continental")
        seq = [Pitch(p) for p in "F3 B-3 F3 B-3 F3".split(" ")]
        pg = ParseGraph(gamut, seq)
        self.assertTrue(True)


class TestGamutParseGraph(unittest.TestCase):
    def test_search_musica_ficta(self):
        gamut = get_gamut("hard-continental")
        pg = GamutParseGraph(gamut)
        matches = pg.search(Pitch("C#4"))
        self.assertEqual(len(matches), 2)

    def test_shortest_paths_c_sharp(self):
        gamut = get_gamut("hard-continental")
        pg = GamutParseGraph(gamut)
        paths = pg.shortest_paths(Pitch("D4"), Pitch("C#4"))
        self.assertEqual(len(paths), 2)

    # TODO should work, but the order nof nodes may be randomized?
    # def test_b_flat_in_hard_hex(self):
    #     example = [Pitch(p) for p in "D4 A3 B-3 A3 G3 F3".split(" ")]
    #     gamut = get_gamut("hard-continental")
    #     pg = GamutParseGraph(gamut, example, mismatch_penalty=10)
    #     node_a = pg.positions[4][1]
    #     node_b_flat = pg.positions[5][0]
    #     self.assertGreater(pg[node_a][node_b_flat]["weight"], 10)


#     def test_mismatch_penalty(self):
#         gamut = HardContinentalGamut()
#         seq = [Pitch(p) for p in "A3 B-3 A3".split(" ")]
#         pg = GamutParseGraph(gamut, seq, mismatch_penalty=2)
#         node1 = (1, gamut.names["la2"])
#         node2 = (2, gamut.names["fi2"])
#         node3 = (3, gamut.names["la2"])
#         self.assertEqual(pg[node1][node2]["weight"], 1.5)
#         self.assertEqual(pg[node2][node3]["weight"], 1)
