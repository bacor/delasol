import typing as t

from delasol.solmizers.solmizer import Solmizer, register_solmizer
from delasol.graphs.gamut_graph import GamutGraph, register_gamut

TINCTORIS_MUTATIONS = [ # Expositio Manus, capitulum VII: de mutationibus
    # ut -> re (always soft -> hard)
    dict(source="hard", dir="up", target="soft", moves=[("ut", "fa")]),
    dict(source="hard", dir="down", target="soft", moves=[("ut", "ut")]),
    # ut -> fa
    dict(source="natural", dir="down", target="hard", moves=[("ut", "mi")]),
    dict(source="natural", dir="down", target="hard", moves=[("ut", "re")]),
    dict(source="natural", dir="down", target="hard", moves=[("ut", "ut")]),
    dict(source="soft", dir="down", target="natural", moves=[("ut", "mi")]),
    dict(source="soft", dir="down", target="natural", moves=[("ut", "re")]),
    dict(source="soft", dir="down", target="natural", moves=[("ut", "ut")]),
    # ut -> sol
    dict(source="hard", dir="down", target="natural", moves=[("ut", "fa")]),
    dict(source="hard", dir="down", target="natural", moves=[("ut", "mi")]),
    dict(source="hard", dir="down", target="natural", moves=[("ut", "re")]),
    dict(source="hard", dir="down", target="natural", moves=[("ut", "ut")]),
    dict(source="natural", dir="down", target="soft", moves=[("ut", "fa")]),
    dict(source="natural", dir="down", target="soft", moves=[("ut", "mi")]),
    dict(source="natural", dir="down", target="soft", moves=[("ut", "re")]),
    dict(source="natural", dir="down", target="soft", moves=[("ut", "ut")]),
    # re -> ut (always soft -> hard)
    dict(source="soft", dir="up", target="hard", moves=[("re", "mi")]),
    dict(source="soft", dir="up", target="hard", moves=[("re", "la")]),
    # re -> mi (always hard -> soft)
    dict(source="hard", dir="up", target="soft", moves=[("re", "fa")]),
    dict(source="hard", dir="down", target="soft", moves=[("re", "ut")]),
    # re -> sol
    dict(source="natural", dir="down", target="hard", moves=[("re", "mi")]),
    dict(source="natural", dir="down", target="hard", moves=[("re", "re")]),
    dict(source="natural", dir="down", target="hard", moves=[("re", "ut")]),
    dict(source="soft", dir="down", target="natural", moves=[("re", "mi")]),
    dict(source="soft", dir="down", target="natural", moves=[("re", "re")]),
    dict(source="soft", dir="down", target="natural", moves=[("re", "ut")]),
    # re -> la
    dict(source="hard", dir="down", target="natural", moves=[("re", "fa")]),
    dict(source="hard", dir="down", target="natural", moves=[("re", "mi")]),
    dict(source="hard", dir="down", target="natural", moves=[("re", "re")]),
    dict(source="hard", dir="down", target="natural", moves=[("re", "ut")]),
    dict(source="natural", dir="down", target="soft", moves=[("re", "fa")]),
    dict(source="natural", dir="down", target="soft", moves=[("re", "mi")]),
    dict(source="natural", dir="down", target="soft", moves=[("re", "re")]),
    dict(source="natural", dir="down", target="soft", moves=[("re", "ut")]),
    # mi -> re (always soft -> hard)
    dict(source="soft", dir="up", target="hard", moves=[("mi", "mi")]),
    dict(source="hard", dir="down", target="soft", moves=[("mi", "la")]),
    # mi -> la
    dict(source="natural", dir="down", target="hard", moves=[("mi", "mi")]),
    dict(source="natural", dir="down", target="hard", moves=[("mi", "re")]),
    dict(source="natural", dir="down", target="hard", moves=[("mi", "ut")]),
    dict(source="soft", dir="down", target="natural", moves=[("mi", "mi")]),
    dict(source="soft", dir="down", target="natural", moves=[("mi", "re")]),
    dict(source="soft", dir="down", target="natural", moves=[("mi", "ut")]),
    # fa -> ut
    dict(source="hard", dir="up", target="natural", moves=[("fa", "fa")]),
    dict(source="hard", dir="up", target="natural", moves=[("fa", "sol")]),
    dict(source="hard", dir="up", target="natural", moves=[("fa", "la")]),
    dict(source="natural", dir="up", target="soft", moves=[("fa", "fa")]),
    dict(source="natural", dir="up", target="soft", moves=[("fa", "sol")]),
    dict(source="natural", dir="up", target="soft", moves=[("fa", "la")]),
    # fa -> sol (always hard -> soft)
    dict(source="hard", dir="down", target="soft", moves=[("fa", "fa")]),
    dict(source="hard", dir="down", target="soft", moves=[("fa", "ut")]),
    # sol -> ut
    dict(source="natural", dir="up", target="hard", moves=[("sol", "mi")]),
    dict(source="natural", dir="up", target="hard", moves=[("sol", "fa")]),
    dict(source="natural", dir="up", target="hard", moves=[("sol", "sol")]),
    dict(source="natural", dir="up", target="hard", moves=[("sol", "la")]),
    dict(source="soft", dir="up", target="natural", moves=[("sol", "mi")]),
    dict(source="soft", dir="up", target="natural", moves=[("sol", "fa")]),
    dict(source="soft", dir="up", target="natural", moves=[("sol", "sol")]),
    dict(source="soft", dir="up", target="natural", moves=[("sol", "la")]),
    # sol -> re
    dict(source="hard", dir="up", target="natural", moves=[("sol", "fa")]),
    dict(source="hard", dir="up", target="natural", moves=[("sol", "sol")]),
    dict(source="hard", dir="up", target="natural", moves=[("sol", "la")]),
    dict(source="natural", dir="up", target="soft", moves=[("sol", "fa")]),
    dict(source="natural", dir="up", target="soft", moves=[("sol", "sol")]),
    dict(source="natural", dir="up", target="soft", moves=[("sol", "la")]),
    # sol -> fa (always soft -> hard)
    dict(source="soft", dir="up", target="hard", moves=[("sol", "la")]),
    dict(source="soft", dir="down", target="hard", moves=[("sol", "mi")]),
    # sol -> la (always hard -> soft)
    dict(source="hard", dir="down", target="soft", moves=[("sol", "fa")]),
    dict(source="hard", dir="down", target="soft", moves=[("sol", "ut")]),
    # la -> re
    dict(source="natural", dir="up", target="hard", moves=[("la", "mi")]),
    dict(source="natural", dir="up", target="hard", moves=[("la", "fa")]),
    dict(source="natural", dir="up", target="hard", moves=[("la", "sol")]),
    dict(source="natural", dir="up", target="hard", moves=[("la", "la")]),
    dict(source="soft", dir="up", target="natural", moves=[("la", "mi")]),
    dict(source="soft", dir="up", target="natural", moves=[("la", "fa")]),
    dict(source="soft", dir="up", target="natural", moves=[("la", "sol")]),
    dict(source="soft", dir="up", target="natural", moves=[("la", "la")]),
    # la -> mi
    dict(source="hard", dir="up", target="natural", moves=[("la", "fa")]),
    dict(source="hard", dir="up", target="natural", moves=[("la", "sol")]),
    dict(source="hard", dir="up", target="natural", moves=[("la", "la")]),
    dict(source="natural", dir="up", target="soft", moves=[("la", "fa")]),
    dict(source="natural", dir="up", target="soft", moves=[("la", "sol")]),
    dict(source="natural", dir="up", target="soft", moves=[("la", "la")]),
    # la -> sol (always soft -> hard)
    dict(source="soft", dir="up", target="hard", moves=[("la", "la")]),
    dict(source="soft", dir="down", target="hard", moves=[("la", "mi")]),
]

'''
Note.
Previously we though that in Tinctoris mutation can happen "from any to any syllable".
Practically however this is not true. Mutation happens only when the next note
lies outside the current hexachord. There is no need for mutation when it stays in the same hexachord.
'''

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
