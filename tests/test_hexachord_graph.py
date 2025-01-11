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
from delasol.hexachord_graph import HexachordGraph as Hexachord


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

    def test_names(self):
        H1 = Hexachord("G2")
        self.assertEqual(H1.names["ut_G2"], Pitch("G2"))
        self.assertEqual(H1.names["re_G2"], Pitch("A2"))
        self.assertEqual(H1.names["mi_G2"], Pitch("B2"))
        self.assertEqual(H1.names["fi_G2"], Pitch("F3"))

        H2 = Hexachord("C3")
        self.assertEqual(H2.names["ut_C3"], Pitch("C3"))
        self.assertEqual(H2.names["re_C3"], Pitch("D3"))
        self.assertEqual(H2.names["mi_C3"], Pitch("E3"))
        self.assertEqual(H2.names["fi_C3"], Pitch("B-3"))
