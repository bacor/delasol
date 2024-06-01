import unittest
from music21.pitch import Pitch
from hexachord.gamut_graph import HexachordGraph
from hexachord.gamut_graph import GamutGraph, HardGamutGraph, SoftGamutGraph
from hexachord.gamut_graph import TINCTORIS_MUTATIONS
import networkx as nx
import numpy as np


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
        self.assertEqual(attrs["syllable"], "ut")
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

    def test_tinctoris_mutations(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = GamutGraph(hexachords=[H1, H2], mutations=TINCTORIS_MUTATIONS)
        self.assertTrue((G.names["fa1"], G.names["re2"]) in G.edges)
        self.assertTrue((G.names["fa2"], G.names["la1"]) in G.edges)

    def test_add_edges_by_name(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = GamutGraph(hexachords=[H1, H2])
        G.add_edges_by_names([("fa1", "re2"), ("fa2", "la1")])
        self.assertTrue((G.names["fa1"], G.names["re2"]) in G.edges)
        self.assertTrue((G.names["fa2"], G.names["la1"]) in G.edges)

    def test_spine(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = GamutGraph(hexachords=[H1, H2], mutations=TINCTORIS_MUTATIONS)
        spine = "ut1 re1 mi1 fa1 re2 mi2 fa2 sol2 la2 fi2".split(" ")
        spine = [G.names[name] for name in spine]
        self.assertListEqual(G.spine, spine)

    def test_spine_disconnected_graph(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = GamutGraph(hexachords=[H1, H2])
        self.assertRaises(Exception, lambda: G.spine)

    def test_pitches(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = GamutGraph(hexachords=[H1, H2])
        targets = [(1, Pitch("C3")), (2, Pitch("C3"))]
        self.assertListEqual(G.pitches[Pitch("C3")], targets)

    def test_fill_gap(self):
        [H1, H2, H4] = [HexachordGraph(p) for p in "G2 C3 G3".split(" ")]
        G = GamutGraph(hexachords=[H1, H2, H4], mutations=TINCTORIS_MUTATIONS)
        filled = G.fill_gap(Pitch("C3"), Pitch("G3"))
        expected = [Pitch("C3"), Pitch("D3"), Pitch("E3"), Pitch("F3")]
        self.assertListEqual(filled, expected)

    def test_fill_gap_down(self):
        [H1, H2, H4] = [HexachordGraph(p) for p in "G2 C3 G3".split(" ")]
        G = GamutGraph(hexachords=[H1, H2, H4], mutations=TINCTORIS_MUTATIONS)
        filled = G.fill_gap(Pitch("G3"), Pitch("E3"))
        expected = [Pitch("G3"), Pitch("F3")]
        self.assertListEqual(filled, expected)

    def test_fill_gap_flats(self):
        [H2, H4] = [HexachordGraph(p) for p in "C3 G3".split(" ")]
        G = GamutGraph(hexachords=[H2, H4], mutations=TINCTORIS_MUTATIONS)
        filled = G.fill_gap(Pitch("G3"), Pitch("B-3"))
        expected = [Pitch("G3"), Pitch("A3")]
        self.assertListEqual(filled, expected)

    def test_fill_gaps(self):
        [H2, H4] = [HexachordGraph(p) for p in "C3 G3".split(" ")]
        G = GamutGraph(hexachords=[H2, H4], mutations=TINCTORIS_MUTATIONS)
        melody = [Pitch(p) for p in "C3 G3 D4".split(" ")]
        filled, is_original = G.fill_gaps(melody)
        expected = [Pitch(p) for p in "C3 D3 E3 F3 G3 A3 B3 C4 D4".split(" ")]
        self.assertListEqual(expected, filled)

        reconstruction = [filled[i] for i, is_orig in enumerate(is_original) if is_orig]
        self.assertListEqual(reconstruction, melody)

    def test_fill_gaps_repetitions(self):
        [H2, H4] = [HexachordGraph(p) for p in "C3 G3".split(" ")]
        G = GamutGraph(hexachords=[H2, H4], mutations=TINCTORIS_MUTATIONS)
        melody = [Pitch(p) for p in "C3 E3 E3 E3 G3".split(" ")]
        filled, is_original = G.fill_gaps(melody)
        expected = [Pitch(p) for p in "C3 D3 E3 E3 E3 F3 G3".split(" ")]
        self.assertListEqual(filled, expected)
        reconstruction = [filled[i] for i, is_orig in enumerate(is_original) if is_orig]
        self.assertListEqual(reconstruction, melody)

    def test_fill_gaps_jumpy(self):
        [H2, H4] = [HexachordGraph(p) for p in "C3 G3".split(" ")]
        G = GamutGraph(hexachords=[H2, H4], mutations=TINCTORIS_MUTATIONS)
        melody = [Pitch(p) for p in "C3 G3 E3 C4 B3 B3".split(" ")]
        filled, is_original = G.fill_gaps(melody)
        expected = [
            Pitch(p) for p in "C3 D3 E3 F3 G3 F3 E3 F3 G3 A3 B3 C4 B3 B3".split(" ")
        ]
        self.assertListEqual(expected, filled)


class TestHardGamutGraph(unittest.TestCase):
    def test_hard_gamut(self):
        gamut = HardGamutGraph()
        self.assertEqual(gamut.hexachords[1].tonic, Pitch("G2"))
        self.assertEqual(gamut.hexachords[2].tonic, Pitch("C3"))
        self.assertEqual(gamut.hexachords[4].tonic, Pitch("G3"))
        self.assertEqual(gamut.hexachords[5].tonic, Pitch("C4"))
        self.assertEqual(gamut.hexachords[7].tonic, Pitch("G4"))


class TestSoftGamutGraph(unittest.TestCase):
    def test_soft_gamut(self):
        gamut = SoftGamutGraph()
        self.assertEqual(gamut.hexachords[2].tonic, Pitch("C3"))
        self.assertEqual(gamut.hexachords[3].tonic, Pitch("F3"))
        self.assertEqual(gamut.hexachords[5].tonic, Pitch("C4"))
        self.assertEqual(gamut.hexachords[6].tonic, Pitch("F4"))

    def test_issue1(self):
        melody = ["D3", "F3", "F3", "G3", "A3", "B-3", "B-3", "A3", "F3", "G3"]
        pitches = [Pitch(p) for p in melody]
        gamut = SoftGamutGraph()
        filled, original = gamut.fill_gaps(pitches)
        self.assertTrue(True)
