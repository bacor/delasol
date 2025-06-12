import typing as t

from delasol.solmizers.solmizer import Solmizer, register_solmizer
from delasol.graphs.gamut_graph import GamutGraph, register_gamut

TINCTORIS_MUTATIONS = [
    # Mutations from natural hexachords
    dict(source="natural", dir="up", target="hard", moves=[("sol", "re")]),
    # Mutations from hard hexachords
    # Mutations from soft hexachords
]

class Tinctoris15CenturyGamutGraph(GamutGraph):
    name = "tinctoris_15c"

    def __init__(
        self,
        hexachords: t.Optional[list[HexachordGraph]] = None,
        mutations: t.Optional[dict] = CONTINENTAL_MUTATIONS,
        hexachord_kws: t.Optional[dict] = {},
        **kwargs,
    ):
        if hexachords is None:
            bases = ["G2", "C3", "G3", "C4", "G4", "C5"]
            hexachords = [HexachordGraph(base, **hexachord_kws) for base in bases]
        super().__init__(hexachords=hexachords, mutations=mutations, **kwargs)
"

class Tinctoris15cSolmizer(Solmizer):
    name = "tinctoris_15c"

    def __init__(
        self,
        input,
        key: t.Optional[int] = None,
        mismatch_penalty: float = 0,
        **kws,
    ):
        super().__init__(input, key=key, mismatch_penalty=mismatch_penalty, **kws)

register_solmizer(Tinctoris15cSolmizer)
