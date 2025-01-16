# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
from music21.pitch import Pitch
import matplotlib.pyplot as plt

# Local imports
from delasol.utils import as_pitch_list
from delasol.graphs.hexachord_graph import HexachordGraph
from delasol.graphs.gamut_graph import GamutGraph


class TestGamutGraph(unittest.TestCase):
    def test_init(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        gamut = GamutGraph(hexachords=[H1, H2])
        self.assertEqual(len(gamut), 14)
        self.assertDictEqual(gamut.hexachords, {Pitch("G2"): H1, Pitch("C3"): H2})

    def test_init_with_bases(self):
        gamut = GamutGraph(["G2", "C3"])
        self.assertIsInstance(gamut, GamutGraph)

        gamut = GamutGraph(as_pitch_list("G2 C3"))
        self.assertIsInstance(gamut, GamutGraph)

    def test_extrema(self):
        gamut = GamutGraph(["G2", "C3"])
        self.assertEqual(gamut.lowest_node, (Pitch("G2"), Pitch("G2")))
        self.assertEqual(gamut.highest_node, (Pitch("C3"), Pitch("B-3")))

    def test_overlapping_hexachords(self):
        [H1, H2, H3, H4] = [HexachordGraph(p) for p in "G2 C3 F3 G3".split(" ")]
        gamut = GamutGraph(hexachords=[H1, H2, H3, H4])
        neighbours = gamut.overlapping_hexachords
        self.assertListEqual(neighbours[H1], [H2, H3])
        self.assertListEqual(neighbours[H2], [H1, H3, H4])
        self.assertListEqual(neighbours[H3], [H1, H2, H4])
        self.assertListEqual(neighbours[H4], [H2, H3])

    def test_names(self):
        G = GamutGraph(["G2", "C3"])
        self.assertEqual(G.name_to_node["ut_G2"], (Pitch("G2"), Pitch("G2")))

    def test_add_edges_by_name(self):
        G = GamutGraph(["G2", "C3"])
        G.add_edge_by_names("fa_G2", "re_C3")
        G.add_edge_by_names("fa_C3", "la_G2")
        self.assertTrue((G.name_to_node["fa_G2"], G.name_to_node["re_C3"]) in G.edges)
        self.assertTrue((G.name_to_node["fa_C3"], G.name_to_node["la_G2"]) in G.edges)

    def test_pitch_to_node(self):
        G = GamutGraph(["G2", "C3"])
        targets = [(Pitch("G2"), Pitch("C3")), (Pitch("C3"), Pitch("C3"))]
        self.assertListEqual(G.pitch_to_node[Pitch("C3")], targets)

    def test_node_positons(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = GamutGraph(hexachords=[H1, H2])

        pos = G.node_positions(pos_x="order", pos_y="order")
        self.assertEqual(pos[(H1.base, Pitch("G2"))], (0, 0))
        self.assertEqual(pos[(H2.base, Pitch("G3"))], (7, 1))

        # Default
        pos = G.node_positions(pos_x="diatonic", pos_y="order")
        self.assertEqual(pos[(H1.base, Pitch("G2"))], (19, 0))
        self.assertEqual(pos[(H2.base, Pitch("G3"))], (26, 1))

        pos = G.node_positions(pos_x="ps", pos_y="order")
        self.assertEqual(pos[(H1.base, Pitch("G2"))], (43, 0))
        self.assertEqual(pos[(H2.base, Pitch("G3"))], (55, 1))

        pos = G.node_positions(pos_x="order", pos_y="diatonic")
        self.assertEqual(pos[(H1.base, Pitch("G2"))], (0, 19))
        self.assertEqual(pos[(H2.base, Pitch("G3"))], (7, 22))

        pos = G.node_positions(pos_x="order", pos_y="ps")
        self.assertEqual(pos[(H1.base, Pitch("G2"))], (0, 43))
        self.assertEqual(pos[(H2.base, Pitch("G3"))], (7, 48))

        with self.assertRaises(ValueError):
            G.node_positions(pos_x="foo", pos_y="bar")

    def test_add_mutations(self):
        G = GamutGraph(["G2", "C3", "G3"])
        mutations = [
            dict(source="natural", dir="up", target="hard", moves=[("sol", "re")]),
            dict(source="natural", dir="down", target="hard", moves=[("fa", "la")]),
            dict(source="hard", dir="up", target="natural", moves=[("fa", "re", 4)]),
        ]
        G.add_mutations(mutations, default_weight=3)

        # Natural up to the hard hexachord
        sol_C3 = G.get_node(name="sol_C3")
        re_G3 = G.get_node(name="re_G3")
        self.assertTrue(re_G3 in G[sol_C3])
        self.assertEqual(G[sol_C3][re_G3]["weight"], 3)

        # Natural down to the hard hexachord
        fa_C3 = G.get_node(name="fa_C3")
        la_G2 = G.get_node(name="la_G2")
        self.assertTrue(la_G2 in G[fa_C3])
        self.assertEqual(G[fa_C3][la_G2]["weight"], 3)

        # Hard up to the natural hexachord, different weight
        fa_G2 = G.get_node(name="fa_G2")
        re_C3 = G.get_node(name="re_C3")
        self.assertTrue(re_C3 in G[fa_G2])
        self.assertEqual(G[fa_G2][re_C3]["weight"], 4)
