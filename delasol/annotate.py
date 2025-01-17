# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from abc import ABC, abstractmethod

# Library imports
from music21.note import Note
from music21.spanner import Line

# Local imports
from delasol.pathfinders.segmented_pathfinder import SegmentedPathfinder
from delasol.evaluate import EvalResult


EvalColors = {
    EvalResult.CORRECT: "green",
    EvalResult.INCORRECT: "red",
    EvalResult.MISSING: "blue",
    EvalResult.DELETION: "red",
    EvalResult.INSERTION: "red",
    EvalResult.SKIP: "grey",
}
"""Dict[EvalResult, str]: A dictionary mapping evaluation results to their corresponding colors."""

# TODO rename solmizer -> solmization?


class Annotator(ABC):
    """Abstract annotator class.

    An annotator adds, well, annotations to a stream solmized by a solmizer.
    That means that its only input is a solmizer, and that solmizer is required
    to have been initialized with a stream. If you solmize a sequence of pitches
    (instead of a stream), you cannot annotate those. A solmization can have
    multiple annotators working on it. One may for example annotate predicted
    syllables, another may show the segments (when using a segmented pathfinder).

    Subclasses only need to implement the `get_annotations` method, which takes
    a sequence of notes in the stream, and should return a list of annotations
    of the following form:

    .. code-block:: python

        annotations = [
            dict(text='ut', color='red', lyric_num=2),
            dict(text='re', color='green', lyric_num=2),
            # ...


    Basically, an annotation contains keywords passed to the `annotate_note`
    method: possibly text, possibly a color, possibly the lyrics line number.

    Parameters
    ----------
    solmizer : object
        A solmizer initialized with a stream input. Annotators only work on
        music21 streams, so if you initialize a solmizer with a sequence of
        pitches, you cannot annotate them.

    **kws : keyword arguments
        Additional keyword arguments that are eventually passed to the method
        `get_annotations`, which is implemented by a subclass.

    Raises
    ------
    ValueError
        If the provided `solmizer` does not have a valid 'stream' attribute.
    """

    def __init__(self, solmizer, **kws):
        if not solmizer.stream:
            raise ValueError("Solmizer must have a stream")
        self.solmizer = solmizer
        self.stream = self.solmizer.stream

        # Keywords are passed on to get_annotations
        self.__init_kws = kws

    # Static methods

    @staticmethod
    def num_lyrics(notes: t.Iterable[Note]) -> int:
        """Calculate the maximum number of lyrics lines in an iterable of notes.

        Parameters
        ----------
        notes : Iterable[Note]
            An iterable of notes that may have multipel lyrics

        Returns
        -------
        int
            The maximum number of lyric lines.
        """
        num_lyrics = 0
        for note in notes:
            numbers = [lyric.number for lyric in note.lyrics]
            if len(numbers) > 0:
                num_lyrics = max(max(numbers), num_lyrics)
        return num_lyrics

    @staticmethod
    def annotate_note(
        note: Note, text: str = None, color: str = None, lyric_number: int = 1
    ) -> None:
        """Annotate a musical note with optional text and color.

        Parameters
        ----------
        note : Note
            The note to be annotated.
        text : str, optional
            The text to add as a lyric to the note. If None, no text is added.
        color : str, optional
            The color to apply to the lyric. If None, no color is applied.
        lyric_number : int, optional
            The lyric line number; default is 1.

        Returns
        -------
        None
            This function modifies the note in place and does not return a value.
        """
        if text is not None:
            note.addLyric(text, lyricNumber=lyric_number)
        lyrics = {lyric.number: lyric for lyric in note.lyrics}
        if color is not None:
            if lyric_number in lyrics:
                lyrics[lyric_number].style.color = color

    @staticmethod
    def set_lyrics_color(
        notes: t.Iterable[Note], lyric_number: int, color: str = "#000000"
    ) -> None:
        """Set the color of the lyrics for a given set of musical notes.

        Parameters
        ----------
        notes : Iterable[Note]
            An iterable collection of Note objects to which the color will be
            applied.
        lyric_number : int
            The lyrics line number
        color : str, optional
            A string representing the color in hexadecimal format (default is
            "#000000").

        Returns
        -------
        None
            This function does not return a value. It modifies the notes in
            place by annotating them with the specified color.
        """
        for note in notes:
            Annotator.annotate_note(note, color=color, lyric_number=lyric_number)

    # Annotations

    @abstractmethod
    def get_annotations(self, notes, **kws) -> dict:
        raise NotImplemented

    def annotate(self, notes=None, offset: int = None, **kws) -> dict:
        """Annotate notes with specified annotations.

        Parameters
        ----------
        notes : iterable, optional
            A collection of notes to annotate. If None, all notes in the stream
            are used.
        offset : int, optional
            An integer value to offset the annotation numbers. If None, the
            offset is determined by the number of lyrics associated with the
            notes.
        **kws : keyword arguments
            Additional keyword arguments passed to the annotation retrieval
            function.

        Returns
        -------
        dict
            A list of annotations corresponding to the provided notes.

        Raises
        ------
        ValueError
            If the number of notes does not match the number of annotations, or
            if the annotations are not an iterable of dictionaries.
        """
        # By default use all notes in the stream
        if notes is None:
            notes = self.stream.flatten().notes

        # Get and validate all annotations
        kwargs = dict(**self.__init_kws)
        kwargs.update(**kws)
        annotations = self.get_annotations(notes, **kwargs)
        if not len(notes) == len(annotations):
            raise ValueError(
                "The number of notes should match the number of annotations."
            )
        if not isinstance(annotations[0], dict):
            raise ValueError("Annotations should be an iterable of dictionaries")

        # Update the offset for all annotations
        if offset is None:
            offset = self.num_lyrics(notes)
        for annot in annotations:
            annot["lyric_number"] = annot.get("lyric_number", 1) + offset

        # Go!
        for note, annotation in zip(notes, annotations):
            self.annotate_note(note, **annotation)

        return annotations


########################## Registry ##########################

ANNOTATORS = {}
"""dict[str, Annotator]: The registry of annotators"""


def register_annotator(annotator: Annotator):
    """Register an new annotator class.

    Parameters
    ----------
    annotator : Annotator
        A subclass of the `Annotator` class that must have a `name` attribute.

    Raises
    ------
    ValueError
        If `annotator` is not a subclass of `Annotator` or if it does not have
        a `name` attribute.
    """
    if not issubclass(annotator, Annotator):
        raise ValueError("An annotator must be an subclas of the Annotator class")
    if not hasattr(annotator, "name"):
        raise ValueError("An Annotator must have a 'name' attribute.")

    ANNOTATORS[annotator.name] = annotator


def get_annotator(name: str, solmizer: "Solmizer", **kws) -> Annotator:
    """Get an annotator instance by name.

    Parameters
    ----------
    name : str
        The name of the annotator to retrieve.
    solmizer : Solmizer
        An instance of the Solmizer class to be passed to the annotator.
    **kws :
        Additional keyword arguments to be passed to the annotator's
        constructor.

    Returns
    -------
    Annotator
        An instance of the specified annotator class.

    Raises
    ------
    ValueError
        If the specified annotator name is not found in the ANNOTATORS
        registry.
    """
    annotator_class = ANNOTATORS.get(name)
    if not annotator_class:
        raise ValueError(f"Annotator '{name}' not found.")
    return annotator_class(solmizer, **kws)


########################## Annotators ##########################


class SolmizationAnnotator(Annotator):
    name = "solmization"

    def get_annotations(self, notes, **kws):
        predictions = self.solmizer.solmize(**kws)
        return [dict(text=pred) for pred in predictions]


register_annotator(SolmizationAnnotator)


class EvaluationAnnotator(Annotator):
    name = "evaluation"

    def get_annotations(
        self,
        notes,
        use_color: bool = True,
        **eval_kws,
    ):
        eval_kws.update(return_counts=False)
        results, predictions = self.solmizer.evaluate(
            return_predictions=True, **eval_kws
        )

        # Transform into annotations
        annotations = []
        for pred, result in zip(predictions, results):
            annot = dict(text=pred)
            if use_color:
                annot["color"] = EvalColors[result]
            annotations.append(annot)
        return annotations


register_annotator(EvaluationAnnotator)


class TextAnnotator(Annotator):
    name = "text"

    def get_annotations(self, notes, text: t.Iterable[str], **kws):
        return [dict(text=t, **kws) for t in text]


register_annotator(TextAnnotator)


class SegmentAnnotator(Annotator):
    # TODO TEST

    name = "segment"

    def __init__(self, solmizer):
        if not isinstance(solmizer.pathfinder, SegmentedPathfinder):
            raise ValueError(
                "Segment annotator can only annotate solmizers using a segmented pathfinder."
            )
        super().__init__(solmizer)

    def annotate_segment(self, notes):
        line = Line(notes)
        line.lineType = "dotted"
        self.stream.insert(0, line)

    def annotate_segments(self, segments):
        raise NotImplemented


register_annotator(SegmentAnnotator)
