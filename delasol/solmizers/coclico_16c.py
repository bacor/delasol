import typing as t

from delasol.graphs.gamut_graph import GamutGraph, register_gamut
from delasol.graphs.hexachord_graph import HexachordGraph
from delasol.solmizers.solmizer import register_solmizer

from tinctoris_15c import Tinctoris15cSolmizer
from continental_16c import CONTINENTAL_MUTATIONS

class Coclico16CenturyGamutGraph(GamutGraph):
    name = "coclico_16c"

    def __init__(
        self,
        hexachords: t.Optional[list[HexachordGraph]] = None,
        mutations: t.Optional[dict] = CONTINENTAL_MUTATIONS,
        hexachord_kws: t.Optional[dict] = {},
        **kwargs,
    ):
        if hexachords is None:
            bases = ["C2", "F2", "G2", "C3", "F3", "G3", "C4", "F4", "G4", "C5", "F5"]
            hexachords = [HexachordGraph(base, **hexachord_kws) for base in bases]
        super().__init__(hexachords=hexachords, mutations=mutations, **kwargs)

register_gamut(Coclico16CenturyGamutGraph)


class Coclico16cSolmizer(Tinctoris15cSolmizer):
    name = "coclico_16c"

    def get_gamut_graph(self):

        return Coclico16CenturyGamutGraph(hexachord_kws=dict(fa_super_la=False))

register_solmizer(Coclico16cSolmizer)
