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
from delasol.evaluator import Evaluator, EvaluationResult


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
                "Unsupported input type: you can pass a string of pitches or an iterable of pitches, notes or pitch strings."
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

    def solmize(self, rank=0, format="syllable", **formatter_kws) -> t.Iterable[str]:
        """Solmization with a given rank and format.

        Parameters
        ----------
        rank : int, optional
            The rank of the solmization. Default is 0 (the best path).
        format : str, optional
            The format in which to return the solmization. Default is "syllable".
        **formatter_kws : keyword arguments
            Additional keyword arguments to be passed to the formatter.

        Returns
        -------
        Iterable[str]
            A sequence of for example syllables, depending on the format.
        """
        path = self.solmization_path(rank)
        formatter = get_formatter(format, self.gamut, **formatter_kws)
        return formatter.format(path)

    def evaluate(
        self,
        rank=0,
        format="syllable",
        targets: t.Iterable[str] = None,
        target_notes: t.Iterable[Note] = None,
        target_lyric_number: int = None,
        formatter_kws: dict = {},
        evaluator_kws: dict = {},
    ) -> EvaluationResult:
        """Evaluate the predicted syllables against the specified targets: the annotated syllables.

        Parameters
        ----------
        rank
            The rank to use for evaluation. Default is 0.
        format
            The format of the evaluation, default is "syllable".
        targets
            The target lyrics to evaluate against. If None and
            target_lyric_number is provided, targets will be extracted
            from the specified target notes.
        target_notes
            The notes corresponding to the target lyrics. If None, all
            input notes will be used.
        target_lyric_number
            The specific lyric number to extract targets from if
            targets is None.
        formatter_kws
            Additional keyword arguments for the formatter.
        evaluator_kws
            Additional keyword arguments for the evaluator: see :meth:`Evaluator.evaluate`

        Returns
        -------
        results : EvaluationResult
            The results of the evaluation.
        """
        if targets is None and target_lyric_number is None:
            raise ValueError("Either targets or target_lyric_number must be provided.")

        # Read out targets from lyrics if needdd
        if targets is None and target_lyric_number is not None:
            if target_notes is None:
                target_notes = self.input
            targets = extract_lyrics(target_notes, number=target_lyric_number)

        # Raise an error if all targets are None
        if all([t is None for t in targets]):
            raise ValueError(
                "No targets were found: all targets are None. Did you specify targets or targets_lyric_number correctly?"
            )

        # Get predictions
        predictions = self.solmize(rank=rank, format=format, **formatter_kws)

        # Evaluate
        results = Evaluator.evaluate(
            predictions=predictions, targets=targets, **evaluator_kws
        )

        return results

    def annotate(
        self, annotator_name: str = "solmization", **annotator_kws
    ) -> "Annotator":
        """Annotate the stream using a particular annotator.

        Note that this method only works when the solmizer has been
        initialized using a stream. Sequences of pitches, for example,
        cannot be annotated.

        Parameters
        ----------
        name
            The name of the annotator to be used for annotation
        **annotator_kws
            Additional keyword arguments to be passed to `annotator.annotate`.
        """
        annotator = get_annotator(annotator_name, self)
        annotator.annotate(**annotator_kws)
        return annotator


###############################################################################

# Solmizer registry

SOLMIZERS = {}


def register_solmizer(solmizer: Solmizer):
    """Register a solmizer class.

    Parameters
    ----------
    solmizer : Solmizer
        A subclass of the Solmizer class that must have a 'name' attribute.

    Raises
    ------
    ValueError
        If the provided solmizer is not a subclass of Solmizer or if it does
        not have a 'name' attribute.
    """
    if not issubclass(solmizer, Solmizer):
        raise ValueError("The solmizer must be an instance of the Solmizer class")
    if not hasattr(solmizer, "name"):
        raise ValueError("Solmizers must have a 'name' attribute.")

    SOLMIZERS[solmizer.name] = solmizer


def get_solmizer(name, input, **kws) -> Solmizer:
    """Get an instance of a Solmizer class based on the provided name.

    Parameters
    ----------
    name : str
        The name of the Solmizer class to instantiate.
    input : Any
        The input data to be processed by the Solmizer.
    **kws : keyword arguments
        Additional keyword arguments to be passed to the Solmizer class.

    Returns
    -------
    Solmizer
        An instance of the specified Solmizer class.

    Raises
    ------
    ValueError
        If the specified Solmizer class name is not found.
    """
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
