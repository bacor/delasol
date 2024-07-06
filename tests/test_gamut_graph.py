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
from alamire.gamut_graph import HexachordGraph
from alamire.gamut_graph import GamutGraph
from alamire.gamut_graph import HardContinentalGamut
from alamire.gamut_graph import SoftContinentalGamut
from alamire.gamut_graph import CONTINENTAL_MUTATIONS


class TestHexachordGraph(unittest.TestCase):

    def test_init(self):
        hg = HexachordGraph(tonic="G2", fa_super_la=True, fa_super_la_weight=0.75)
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
        self.assertEqual(hg.tonic, Pitch("G2"))
        self.assertEqual(hg.type, "hard")
        attrs = hg.nodes[Pitch("G2")]
        self.assertEqual(attrs["degree"], 1)
        self.assertEqual(attrs["name"], "ut1")

    def test_types(self):
        H1 = HexachordGraph("G2")
        self.assertEqual(H1.type, "hard")
        H2 = HexachordGraph("C3")
        self.assertEqual(H2.type, "natural")
        H3 = HexachordGraph("F3")
        self.assertEqual(H3.type, "soft")

    def test_names(self):
        H1 = HexachordGraph("G2")
        self.assertEqual(H1.names["ut1"], Pitch("G2"))
        self.assertEqual(H1.names["re1"], Pitch("A2"))
        self.assertEqual(H1.names["mi1"], Pitch("B2"))
        self.assertEqual(H1.names["fi1"], Pitch("F3"))

        H2 = HexachordGraph("C3")
        self.assertEqual(H2.names["ut2"], Pitch("C3"))
        self.assertEqual(H2.names["re2"], Pitch("D3"))
        self.assertEqual(H2.names["mi2"], Pitch("E3"))
        self.assertEqual(H2.names["fi2"], Pitch("B-3"))


class TestGamutGraph(unittest.TestCase):
    def test_init(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        gamut = GamutGraph(hexachords=[H1, H2])
        self.assertEqual(len(gamut), 14)
        self.assertDictEqual(gamut.hexachords, {1: H1, 2: H2})

    def test_extrema(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        gamut = GamutGraph(hexachords=[H1, H2])
        self.assertEqual(gamut.lowest, (1, Pitch("G2")))
        self.assertEqual(gamut.highest, (2, Pitch("B-3")))

    def test_overlapping_hexachords(self):
        [H1, H2, H3, H4] = [HexachordGraph(p) for p in "G2 C3 F3 G3".split(" ")]
        gamut = GamutGraph(hexachords=[H1, H2, H3, H4])
        neighbours = gamut.overlapping_hexachords
        self.assertListEqual(neighbours[H1], [H2, H3])
        self.assertListEqual(neighbours[H2], [H1, H3, H4])
        self.assertListEqual(neighbours[H3], [H1, H2, H4])
        self.assertListEqual(neighbours[H4], [H2, H3])

    def test_CONTINENTAL_MUTATIONS(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = GamutGraph(hexachords=[H1, H2], mutations=CONTINENTAL_MUTATIONS)
        self.assertTrue((G.names["fa1"], G.names["re2"]) in G.edges)
        self.assertTrue((G.names["fa2"], G.names["la1"]) in G.edges)

    def test_add_edges_by_name(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = GamutGraph(hexachords=[H1, H2])
        G.add_edges_by_names([("fa1", "re2"), ("fa2", "la1")])
        self.assertTrue((G.names["fa1"], G.names["re2"]) in G.edges)
        self.assertTrue((G.names["fa2"], G.names["la1"]) in G.edges)

    def test_pitches(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = GamutGraph(hexachords=[H1, H2])
        targets = [(1, Pitch("C3")), (2, Pitch("C3"))]
        self.assertListEqual(G.pitches[Pitch("C3")], targets)


class TestHardContinentalGamut(unittest.TestCase):
    def test_hard_gamut(self):
        gamut = HardContinentalGamut()
        self.assertEqual(gamut.hexachords[1].tonic, Pitch("G2"))
        self.assertEqual(gamut.hexachords[2].tonic, Pitch("C3"))
        self.assertEqual(gamut.hexachords[4].tonic, Pitch("G3"))
        self.assertEqual(gamut.hexachords[5].tonic, Pitch("C4"))
        self.assertEqual(gamut.hexachords[7].tonic, Pitch("G4"))


class TestSoftContinentalGamut(unittest.TestCase):
    def test_soft_gamut(self):
        gamut = SoftContinentalGamut()
        self.assertEqual(gamut.hexachords[2].tonic, Pitch("C3"))
        self.assertEqual(gamut.hexachords[3].tonic, Pitch("F3"))
        self.assertEqual(gamut.hexachords[5].tonic, Pitch("C4"))
        self.assertEqual(gamut.hexachords[6].tonic, Pitch("F4"))
