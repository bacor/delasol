# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
from music21.pitch import Pitch

# Local imports
from delasol.graphs.gamut_graph import GamutGraph
from delasol.formatter import get_formatter


class TestFormatter(unittest.TestCase):

    def test_basics(self):
        gamut = GamutGraph(["G2", "C3"])
        fmt = get_formatter("syllable", gamut)
        path = [(Pitch("G2"), Pitch("G2")), (Pitch("C3"), Pitch("D3"))]
        self.assertEqual(fmt.format(path), ["ut", "re"])

    def test_format(self):
        gamut = GamutGraph(["G2", "C3"])
        names = "ut_G2 re_G2 mi_G2 fa_G2 sol_G2 re_C3 mi_C3"
        path = [gamut.get_node(name=name) for name in names.split(" ")]
        fmt = get_formatter("syllable", gamut)
        self.assertListEqual(fmt.format(path), "ut re mi fa sol re mi".split(" "))
        self.assertEqual(fmt.format(path, join=True), "ut re mi fa sol re mi")
        self.assertEqual(fmt.format(path, join=True, sep=","), "ut,re,mi,fa,sol,re,mi")
        self.assertEqual(fmt.format(path[0]), "ut")

    def test_format_names(self):
        gamut = GamutGraph(["G2", "C3"])
        fmt = get_formatter("syllable", gamut)
        self.assertListEqual(fmt.format_names("ut_G2 re_G2"), "ut re".split(" "))
        with self.assertRaises(KeyError):
            fmt.format_names("ut_G2 re_G2", auto_split=False)
        self.assertEqual(fmt.format_names("ut_G2 re_G2", join=True, sep=","), "ut,re")

    def test_syllable_hexnum(self):
        gamut = GamutGraph(["G2", "C3", "B-2"])
        fmt = get_formatter("syllable_hexnum", gamut)
        output = fmt.format_names("ut_G2 mi_G2 re_C3 re_B-2", subscript=False)
        self.assertListEqual(output, ["ut1", "mi1", "re2", "re_B♭2"])

        output = fmt.format_names("re_B-2", unicode=False)
        self.assertEqual(output, "re_B-2")

    def test_name_formatter(self):
        gamut = GamutGraph(["G2", "C3", "B-2"])
        fmt = get_formatter("name", gamut)
        names = "ut_G2 mi_G2 re_C3 re_B-2"
        output = fmt.format_names(names)
        self.assertListEqual(output, names.split(" "))

        output = fmt.format_names("re_C3 fi_B-2", unicode=True)
        self.assertListEqual(output, ["re_C3", "fi_B♭2"])

    def test_davantes(self):
        # TODO implement
        pass

    def test_mutation_formatter(self):
        seq = "ut_G2 re_G2 mi_G2 fa_G2 sol_G2 re_C3 mi_C3"
        gamut = GamutGraph(["G2", "C3"])
        fmt = get_formatter("mutation", gamut)
        output = fmt.format_names(seq)
        self.assertListEqual(output, ["ut", None, None, None, None, "sol/re", None])

        fmt2 = get_formatter("mutation", gamut)
        output2 = fmt2.format_names(
            seq,
            format="syllable_hexnum",
            mutation=" —> ",
            formatter_kws=dict(subscript=False),
        )
        self.assertListEqual(
            output2, ["ut1", None, None, None, None, "sol1 —> re2", None]
        )

        fmt3 = get_formatter("mutation", gamut)
        output3 = fmt3.format_names("fa_C3 mi_C3 sol_G2 fa_G2 mi_G2")
        self.assertListEqual(output3, ["fa", None, "re/sol", None, None])

        fmt4 = get_formatter("mutation", gamut)
        output3 = fmt3.format_names("fa_C3 mi_C3 re_G2 mi_G2")
        self.assertListEqual(output3, ["fa", None, "?/re", None])
