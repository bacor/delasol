import unittest
from music21.pitch import Pitch
from hexachord.utils import segment_deviations
from hexachord.gamut_graph import HardGamutGraph, HexachordGraph
from hexachord.parse_graph import ParseGraph


def hexachord_match_fn(node, target):
    return node == target


class TestParseGraph(unittest.TestCase):
    def test_exception_jumps(self):
        gamut = HardGamutGraph()
        seq = [Pitch(p) for p in "G3 B3".split(" ")]
        self.assertRaises(ValueError, lambda: ParseGraph(gamut, seq))

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
        gamut = HardGamutGraph()
        seq = [Pitch(p) for p in "A2 B2 C3 D3 E3 F3 E3 D3 C3 B2 A2".split(" ")]
        parse = ParseGraph(gamut, seq)
        expected = [1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1]
        self.assertListEqual(parse.width.tolist(), expected)
