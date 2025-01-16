# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
import networkx as nx
import numpy as np
from music21.pitch import Pitch

# Local imports
from delasol.utils.music import as_pitch_list
from delasol.graphs.hexachord_graph import HexachordGraph as Hexachord


class TestHexachord(unittest.TestCase):

    def test_init(self):
        hg = Hexachord(base="G2", fa_super_la=True, fa_super_la_weight=0.75)
        weights = np.array(
            [
                [0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.75],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.5],
            ]
        )
        adj = nx.adjacency_matrix(hg).todense()
        self.assertListEqual(adj.tolist(), weights.tolist())
        self.assertEqual(hg.number, 1)
        self.assertEqual(hg.base, Pitch("G2"))
        self.assertEqual(hg.quality, "hard")
        attrs = hg.nodes[Pitch("G2")]
        self.assertEqual(attrs["index"], 0)
        self.assertEqual(attrs["name"], "ut_G2")

    def test_qualities(self):
        H1 = Hexachord("G2")
        self.assertEqual(H1.quality, "hard")
        H2 = Hexachord("C3")
        self.assertEqual(H2.quality, "natural")
        H3 = Hexachord("F3")
        self.assertEqual(H3.quality, "soft")

    def test_name_to_node(self):
        H1 = Hexachord("G2")
        names = "ut_G2 re_G2 mi_G2 fa_G2 sol_G2 la_G2 fi_G2".split(" ")
        nodes = as_pitch_list("G2 A2 B2 C3 D3 E3 F3")
        for name, node in zip(names, nodes):
            self.assertEqual(H1.name_to_node[name], node)

        H2 = Hexachord("C3")
        names = "ut_C3 re_C3 mi_C3 fa_C3 sol_C3 la_C3 fi_C3".split(" ")
        nodes = as_pitch_list("C3 D3 E3 F3 G3 A3 B-3")
        for name, node in zip(names, nodes):
            self.assertEqual(H2.name_to_node[name], node)

    def test_syllable_to_node(self):
        syllables = "ut re mi fa sol la fi".split(" ")

        H1 = Hexachord("G2")
        nodes = as_pitch_list("G2 A2 B2 C3 D3 E3 F3")
        for syllable, node in zip(syllables, nodes):
            self.assertEqual(H1.syllable_to_node[syllable], node)

        H2 = Hexachord("C3")
        nodes = as_pitch_list("C3 D3 E3 F3 G3 A3 B-3")
        for syllable, node in zip(syllables, nodes):
            self.assertEqual(H2.syllable_to_node[syllable], node)

    def test_index_to_node(self):
        indices = [0, 1, 2, 3, 4, 5, 6]

        H1 = Hexachord("G2")
        nodes = as_pitch_list("G2 A2 B2 C3 D3 E3 F3")
        for index, node in zip(indices, nodes):
            self.assertEqual(H1.index_to_node[index], node)

        H2 = Hexachord("C3")
        nodes = as_pitch_list("C3 D3 E3 F3 G3 A3 B-3")
        for index, node in zip(indices, nodes):
            self.assertEqual(H2.index_to_node[index], node)

    def test_get_node(self):
        H = Hexachord("F3")
        self.assertEqual(H.get_node(name="ut_F3"), Pitch("F3"))
        self.assertEqual(H.get_node(syllable="mi"), Pitch("A3"))
        self.assertEqual(H.get_node(index=6), Pitch("E-4"))

        with self.assertRaises(ValueError):
            H.get_node(name="ut_F3", syllable="ut")

        with self.assertRaises(ValueError):
            H.get_node(syllable="ut", index=2)

        with self.assertRaises(ValueError):
            H.get_node(index=2, name="bla")

        with self.assertRaises(ValueError):
            H.get_node()
