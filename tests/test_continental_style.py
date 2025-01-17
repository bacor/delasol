# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
from music21.pitch import Pitch

# Local imports
from delasol.graphs.hexachord_graph import HexachordGraph
from delasol.graphs.gamut_graph import GamutGraph as Gamut
from delasol.solmizers.continental_16c import CONTINENTAL_MUTATIONS
from delasol.solmizers.continental_16c import (
    HardContinental16CenturyGamutGraph as HardGamut,
)
from delasol.solmizers.continental_16c import (
    SoftContinental16CenturyGamutGraph as SoftGamut,
)


class TestHardContinentalGamut(unittest.TestCase):
    def test_CONTINENTAL_MUTATIONS(self):
        H1 = HexachordGraph("G2")
        H2 = HexachordGraph("C3")
        G = Gamut(hexachords=[H1, H2], mutations=CONTINENTAL_MUTATIONS)
        self.assertTrue((G.get_node(name="fa_G2"), G.get_node(name="re_C3")) in G.edges)
        self.assertTrue(
            (G.get_node(name="fa_C3"), G.names.get_node(name="la_G2")) in G.edges
        )

    def test_hard_gamut(self):
        gamut = HardGamut()
        self.assertEqual(gamut.hexachords[1].base, Pitch("G2"))
        self.assertEqual(gamut.hexachords[2].base, Pitch("C3"))
        self.assertEqual(gamut.hexachords[4].base, Pitch("G3"))
        self.assertEqual(gamut.hexachords[5].base, Pitch("C4"))
        self.assertEqual(gamut.hexachords[7].base, Pitch("G4"))


class TestSoftContinentalGamut(unittest.TestCase):
    def test_soft_gamut(self):
        gamut = SoftGamut()
        self.assertEqual(gamut.hexachords[2].base, Pitch("C3"))
        self.assertEqual(gamut.hexachords[3].base, Pitch("F3"))
        self.assertEqual(gamut.hexachords[5].base, Pitch("C4"))
        self.assertEqual(gamut.hexachords[6].base, Pitch("F4"))
