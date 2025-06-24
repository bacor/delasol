import typing as t

from delasol.graphs.gamut_graph import GamutGraph, register_gamut
from delasol.graphs.hexachord_graph import HexachordGraph
from delasol.graphs.rollout_graph import RolloutGraph
from delasol.pathfinders.simple_pathfinder import SimplePathfinder
from delasol.pathfinders.segmented_pathfinder import SegmentedPathfinder
from delasol.solmizers.solmizer import Solmizer, register_solmizer

from delasol.solmizers.continental_16c import match_diatonically

from music21.pitch import Pitch
from delasol.custom_types import GamutGraphNode

import matplotlib.pyplot as plt

# To do: better descriptions as comments.

TINCTORIS_MUTATIONS = [
    # Reinterpretation: only one mutation at the very last of the previous hexachord node
    # before the next out-of-reach hexachord node occurs allowed.
    # So: we are staying in the previous hexachord as long as possible.
    #
    # Important insight: on the gamut graph, the arrow points
    # _towards the place of mutation_.
    # In the 'dict' notation below, the last syllable is one that mutates.
    # Initially listed all the mutations from Expositio Manus, capitulum VII: de mutationibus.

    # hard -> natural up
    dict(source="hard", dir="up", target="natural", moves=[("sol", "mi")]),
    # natural -> soft up
    dict(source="natural", dir="up", target="soft", moves=[("sol", "mi")]),
    # natural -> hard up
    dict(source="natural", dir="up", target="hard", moves=[("sol", "re")]),
    # soft -> hard up
    dict(source="soft", dir="up", target="hard", moves=[("re", "re")]),
    dict(source="soft", dir="up", target="hard", moves=[("sol", "sol")]),
    # hard -> soft up
    dict(source="hard", dir="up", target="soft", moves=[("re", "fa")]), # ???
    # soft -> natural up
    dict(source="soft", dir="up", target="natural", moves=[("sol", "re")]),

    # natural -> hard down
    dict(source="natural", dir="down", target="hard", moves=[("re", "fa")]),
    # natural -> soft down
    dict(source="natural", dir="down", target="soft", moves=[("re", "sol")]),
    # hard -> soft down
    dict(source="hard", dir="down", target="soft", moves=[("fa", "fa")]),
    dict(source="hard", dir="down", target="soft", moves=[("ut", "ut")]), # ?
    # soft -> hard down
    dict(source="soft", dir="down", target="hard", moves=[("la", "fa")]),
    # hard -> natural down
    dict(source="hard", dir="down", target="natural", moves=[("re", "sol")]),
    # soft -> natural down
    dict(source="soft", dir="down", target="natural", moves=[("re", "fa")]),

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
        mutations: t.Optional[dict] = TINCTORIS_MUTATIONS,
        hexachord_kws: t.Optional[dict] = {},
        **kwargs,
    ):
        if hexachords is None:
            bases = ["G2", "C3", "F3", "G3", "C4", "F4", "G4"]
            hexachords = [HexachordGraph(base, **hexachord_kws) for base in bases]
        super().__init__(hexachords=hexachords, mutations=mutations, **kwargs)

register_gamut(Tinctoris15CenturyGamutGraph)


def match_musica_vera(node: GamutGraphNode, target: Pitch) -> bool:
    """Check if a node matches a target pitch using musica vera criterion.

    Musica vera is a collection of pitches from "G gamma ut" up to "e la".
    It contains all diatonic pitches and "B molle" (b-flat).

    Parameters
    ----------
    node : GamutGraphNode
        A node in the graph.
    target : Pitch
        The target pitch to match against.

    Returns
    -------
    bool
        Comparison on musica verat criterions.
    """

    diatonic_match = node[1].diatonicNoteNum == target.diatonicNoteNum
    if not diatonic_match:
        return False
    elif node[1].step == 'B' and target.step == 'B':
       return node[1].name == target.name
    else:
        return True


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


    def preprocess_input_and_opts(self, input, **kws):
        input, opts = super().preprocess_input_and_opts(input, **kws)

        key = opts.get("key")

        if key and key not in [0, -1]:
            raise ValueError(f"Unsupported key signature ({opts['key']}).")

        # perhaps check if F below gamma is present and warn if so...?
        # ...or change to a version with one additional hexachord from F in gamut
        # but then reindex hexachords to have the same number as the vanilla tinctoris gamut?

        return input, opts


    def get_gamut_graph(self):

        return Tinctoris15CenturyGamutGraph(hexachord_kws=dict(fa_super_la=False))


    def get_rollout_graph(self, gamut, pitches, **kws):

        # Copied from continental_16c solmizer.
        rollout = RolloutGraph(gamut, pitches, match_fn=match_musica_vera, **kws)

        # Adjust the rollout
        for time, target in zip(rollout.input_timesteps, rollout.sequence):
            for node in rollout.timesteps[time]:
                _, (_, pitch) = node
                if pitch != target:
                    for predecessor in rollout.predecessors(node):
                        rollout[predecessor][node]["weight"] += self.opts.get(
                            "mismatch_penalty"
                        )

        return rollout


    def get_pathfinder(self, rollout, **kws):
        return SegmentedPathfinder(rollout, **kws)

register_solmizer(Tinctoris15cSolmizer)
