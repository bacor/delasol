# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from abc import ABC, abstractmethod
from collections.abc import Iterable

from music21.pitch import Pitch
from music21.note import Note
from music21.stream import Stream
from music21.tie import Tie
from copy import deepcopy

# Local imports
from delasol.utils.music import as_pitch, as_pitch_list, extract_lyrics
from delasol.graphs.gamut_graph import GamutGraph
from delasol.graphs.rollout_graph import RolloutGraph
from delasol.pathfinders.pathfinder import Pathfinder
from delasol.custom_types import GamutGraphPath
from delasol.formatter import get_formatter
from delasol.annotator import get_annotator
from delasol.evaluator import Evaluator


class Solmizer(ABC):
    """
    Abstract class for a solmizer that can produce solmizations for a given
    melody in a particular solmization style.

    More precisely, a solmizer produces a solmization path for a given melody.
    Besides the simpler hard vs soft gamuts, it should be able to handle complex
    cases which for example temporarily mutate to a soft hexachord in a hard
    hexachord context.

    We stick to the idea that solmization happens within one gamut, even
    for more complex changes. This means that such more complicated cases will
    require a richer gamut graph. The idea is then that for a given melody
    the solmization style will _not_ change the gamut graph, but the _rollout_
    instead: by pruning mutations to unneeded gamuts, or make those mutations
    prohibitively expensive, it can dynamically adjust which hexachords are
    available in which part of the melody. How this is decided, is up to the
    particular implementation, and one imagine simple rule-based approaches,
    to indeed more statistical approaches. The point of this class is that it
    should provide the framework to accomodate all such solmization models.
    """

    def __init__(self, input, rollout_kws={}, pathfinder_kws={}, **opts):
        super().__init__()

        self.raw_input = input
        self.input, self.opts = self.preprocess_input_and_opts(input, **opts)
        self.pitches = self.get_pitches(self.input)
        if len(self.pitches) == 0:
            raise ValueError("No pitches were found in the input.")

        self.gamut = self.get_gamut_graph()
        self.rollout = self.get_rollout_graph(self.gamut, self.pitches, **rollout_kws)
        self.pathfinder = self.get_pathfinder(self.rollout, **pathfinder_kws)

    def __repr__(self):
        return f"<Solmizer {self.__class__.__name__}>"

    @property
    def stream(self) -> Stream | bool:
        if isinstance(self.raw_input, Stream):
            return self.raw_input
        else:
            False

    def preproces_stream(self, stream, in_place: bool = True, **kws):
        # Possibly copy the stream
        stream = stream if in_place else deepcopy(stream)

        # Check if the stream has parts
        if stream.hasPartLikeStreams():
            if len(stream.parts) >= 2:
                print(
                    f"Warning: found {len(stream.parts)} parts, but only the first part of the stream will be used"
                )
                stream = stream.parts[0]

        return stream

    def preprocess_input_and_opts(self, input, **kws):
        # Automatically inferred options override keywords
        opts = dict(**kws)

        if isinstance(input, Stream):
            # Possibly copy the stream
            input = self.preproces_stream(input, **kws)

            # Store the clef (only used for Davantes formatting)
            opts["clef"] = input.flatten().clef

            # Determine the key signature (default: 0 flats/sharps)
            if opts.get("key", None) is None:
                key = input.flatten().keySignature
                if key is None:
                    opts["key"] = 0
                else:
                    opts["key"] = key.sharps

            # Only process the notes afterwards
            input = [note for note in input.flatten().notes if note.tie != Tie("stop")]

        return input, opts

    def get_pitches(self, input) -> t.Iterable[Pitch]:
        """Return the melody"""
        if isinstance(input, str):
            pitches = as_pitch_list(input)
        elif isinstance(input, Iterable) and isinstance(input[0], Pitch):
            pitches = input
        elif isinstance(input, Iterable) and isinstance(input[0], str):
            pitches = [as_pitch(p) for p in input]
        elif isinstance(input, Iterable) and isinstance(input[0], Note):
            pitches = [Pitch(n.pitch) for n in input]
        else:
            raise ValueError(
                "Unsupported input type: you can pass an string of pitches or an iterable of pitches, notes or pitch strings"
            )
        return pitches

    @abstractmethod
    def get_gamut_graph(self) -> GamutGraph:
        """Build a gamut graph"""
        raise NotImplemented

    @abstractmethod
    def get_rollout_graph(self, gamut, pitches, **kws) -> RolloutGraph:
        """Set the edge weights of the rollout graph."""
        raise NotImplemented

    @abstractmethod
    def get_pathfinder(self, rollout: RolloutGraph, **kws) -> "Pathfinder":
        raise NotImplemented

    def solmization_path(self, rank: int = 0) -> GamutGraphPath:
        """Compute the solmization path for a given rank.

        Parameters
        ----------
        rank : int, optional
            The rank for which to compute the solmization path. Default is 0.

        Returns
        -------
        GamutGraphPath
            The computed solmization path based on the specified rank.
        """
        return self.pathfinder.get_base_path(rank, inputs_only=True)

    def solmize(self, rank=0, format="syllable", **kws) -> t.Iterable[str]:
        """Solmization with a given rank and format.

        Parameters
        ----------
        rank : int, optional
            The rank of the solmization. Default is 0 (the best path).
        format : str, optional
            The format in which to return the solmization. Default is "syllable".
        **kws : keyword arguments
            Additional keyword arguments to be passed to the formatter.

        Returns
        -------
        Iterable[str]
            A sequence of for example syllables, depending on the format.
        """
        path = self.solmization_path(rank)
        formatter = get_formatter(format, self.gamut, **kws)
        return formatter.format(path)

    def annotate(self, annotator_name: str = None, **kws) -> t.Type["Annotator"]:
        """Annotate the stream using a particular annotator.

        Note that this method only works when the solmizer has been
        initialized using a stream. Sequences of pitches, for example,
        cannot be annotated.

        Parameters
        ----------
        name : str
            The name of the annotator to be used for annotation. Default
            to "solmization", unless either 'target' or 'target_lyric_num'
            is specified; then the evaluation annotator is used.
        **kws : keyword arguments
            Additional keyword arguments to be passed to the annotator.

        Returns
        -------
        Annotator
            An instance of the specified annotator
        """
        # Default annotators
        if annotator_name is None and "targets" in kws or "target_lyric_number" in kws:
            annotator_name = "evaluation"
        elif annotator_name is None:
            annotator_name = "solmization"

        annotator = get_annotator(annotator_name, self)
        annotator.annotate(**kws)
        return annotator

    def evaluate(
        self,
        rank=0,
        format="syllable",
        targets: t.Iterable[str] = None,
        target_notes: t.Iterable[Note] = None,
        target_lyric_number: int = None,
        return_predictions: bool = False,
        return_counts: bool = False,
        use_editorial_suggestion: bool = False,
        skip_uncertain: bool = False,
        detect_insertions_deletions: bool = True,
        **format_kws,
    ):
        # Read out targets from lyrics if needdd
        if targets is None and target_lyric_number is not None:
            if target_notes is None:
                target_notes = self.input
            targets = extract_lyrics(target_notes, number=target_lyric_number)

        # Get evaluation
        predictions = self.solmize(rank=rank, format=format, **format_kws)
        results = Evaluator.evaluate(
            predictions=predictions,
            targets=targets,
            return_counts=return_counts,
            use_editorial_suggestion=use_editorial_suggestion,
            skip_uncertain=skip_uncertain,
            detect_insertions_deletions=detect_insertions_deletions,
        )

        if return_predictions:
            return results, predictions
        else:
            return results


###############################################################################

# Solmizer registry

SOLMIZERS = {}


# TODO documentation
def register_solmizer(solmizer: Solmizer):
    if not issubclass(solmizer, Solmizer):
        raise ValueError("The solmizer must be an instance of the Solmizer class")
    if not hasattr(solmizer, "name"):
        raise ValueError("Solmizers must have a 'name' attribute.")

    SOLMIZERS[solmizer.name] = solmizer


def get_solmizer(name, input, **kws) -> Solmizer:
    solmizer_class = SOLMIZERS.get(name)
    if not solmizer_class:
        raise ValueError(f"Solmizer '{name}' not found.")
    return solmizer_class(input, **kws)


###############################################################################


def solmize(input, style: str = None, **kws) -> Solmizer:
    """Get a solmizer instance based on the provided name and input.

    Parameters
    ----------
    input : Any
        The input data to be processed by the solmizer.
    solmizer_name : str, optional
        The name of the solmizer to be used. If None, a default solmizer
        will be selected.
    **kws : keyword arguments
        Additional keyword arguments to be passed to the solmizer.

    Returns
    -------
    Solmizer
        An instance of the specified solmizer configured with the input
        data and any additional parameters.
    """
    return get_solmizer(style, input, **kws)
