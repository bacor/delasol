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
from delasol.evaluator import EvalStatus, EvaluationResult


EvalColors = {
    EvalStatus.CORRECT: "green",
    EvalStatus.INCORRECT: "red",
    EvalStatus.MISSING: "blue",
    EvalStatus.DELETION: "red",
    EvalStatus.INSERTION: "red",
    EvalStatus.SKIP: "grey",
}
"""Dict[EvalStatus, str]: A dictionary mapping evaluation results to their corresponding colors."""


class Annotator(ABC):
    """Abstract annotator class.

    An annotator adds, well, annotations to a stream solmized by a solmizer.
    That means that its only input is a solmizer, and that solmizer is required
    to have been initialized with a stream. If you solmize a sequence of pitches
    (instead of a stream), you cannot annotate those. A solmization can have
    multiple annotators working on it. One may for example annotate predicted
    syllables, another may show the segments (when using a segmented pathfinder).

    Subclasses only need to implement `get_lyric_annotations`, `get_editorial_annotations`,
    or both. Both methods should have the same signature and take a sequence of notes
    plus possible keyword arguments. They should return a list of dictionaries
    describing how the notes are annotated. For lyric annotations, that for example
    should have the following form:

    .. code-block:: python

        annotations = [
            dict(text='ut', color='red', lyric_num=2),
            dict(text='re', color='green', lyric_num=2),
            # ...


    Basically, these annotations contain keywords passed to the `annotate_note`
    method: possibly text, possibly a color, possibly the lyrics line number.
    For the editorial annotations, the dictionary can contain whatever keys
    you would like to store. You can additionally specify a `key_prefix` that will
    be prefixed to the key, which may be useful when using multiple annotators.

    Parameters
    ----------
    solmizer : object
        A solmizer initialized with a stream input. Annotators only work on
        music21 streams, so if you initialize a solmizer with a sequence of
        pitches, you cannot annotate them.

    Raises
    ------
    ValueError
        If the provided `solmizer` does not have a valid 'stream' attribute.
    """

    def __init__(self, solmizer):
        if not solmizer.stream:
            raise ValueError("Solmizer must have a stream")
        self.solmizer = solmizer
        self.notes = self.solmizer.input
        self.stream = self.solmizer.stream

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
    def annotate_note_lyrics(
        note: Note,
        text: str = None,
        color: str = None,
        lyric_number: int = 1,
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
    def annotate_note_editorial(
        note: Note, key_prefix: str = "", **annotations
    ) -> None:
        for key, value in annotations.items():
            note.editorial.__setitem__(f"{key_prefix}{key}", value)

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

    def get_lyric_annotations(self, notes, **kws) -> t.Iterable[dict] | bool:
        """Get lyric annotations for the given notes.

        These annotations are stored in the lyrics object of the note. The
        function should be implemented by an inheriting class and return a list
        of dictionaries; see the main class for detials.

        Parameters
        ----------
        notes : iterable
            A collection of notes for which to retrieve lyric annotations.
        **kws : keyword arguments
            Additional parameters to customize the behavior of the function.

        Returns
        -------
        iterable of dict or bool
            A generator yielding dictionaries containing lyric annotations for each
            note, or False if the lyrics should not be annotated.
        """
        return False

    def get_editorial_annotations(self, notes, **kws) -> t.Iterable[dict] | bool:
        """Get editorial annotations based on provided notes.

        This method should be implemented by an inheriting class; see main class
        for details.

        Parameters
        ----------
        notes : iterable
            A collection of notes to be processed for annotations.
        **kws : keyword arguments
            Additional parameters that may influence the annotation process.

        Returns
        -------
        iterable of dict or bool
            Returns an iterable of dictionaries containing editorial annotations
            if successful, or False if no editorial annotations should be added.
        """
        return False

    def _annotate_lyrics(
        self,
        notes,
        annotations,
        offset: int = None,
    ) -> dict:

        # Validate all annotations
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

        # Annotate
        for note, annotation in zip(notes, annotations):
            self.annotate_note_lyrics(note, **annotation)

    def _annotate_editorial(
        self,
        notes,
        annotations,
        key_prefix: str = "",
    ) -> dict:
        if not len(notes) == len(annotations):
            raise ValueError(
                "The number of notes should match the number of annotations."
            )
        if not isinstance(annotations[0], dict):
            raise ValueError("Annotations should be an iterable of dictionaries")

        # Update the offset for all annotations
        for annot in annotations:
            annot["key_prefix"] = annot.get("key_prefix", key_prefix)

        for note, annotation in zip(notes, annotations):
            self.annotate_note_editorial(note, **annotation)

        return annotations

    def annotate(
        self,
        notes=None,
        offset: int = None,
        key_prefix: str = "",
        **annotation_kws,
    ) -> dict:
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
        **annotation_kws : keyword arguments
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
            notes = self.notes

        # Annotate the lyrics
        lyric_annot = self.get_lyric_annotations(notes, **annotation_kws)
        if lyric_annot is not False:
            self._annotate_lyrics(notes, lyric_annot, offset=offset)

        # Annotate editorial information
        editorial_annot = self.get_editorial_annotations(notes, **annotation_kws)
        if editorial_annot is not False:
            self._annotate_editorial(notes, editorial_annot, key_prefix=key_prefix)


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

    def get_lyric_annotations(self, notes, predictions=None, **kws):
        predictions = self.solmizer.solmize(**kws)
        return [dict(text=pred) for pred in predictions]

    def get_editorial_annotations(self, notes, **kws):
        # Note that this does not annotate the editorial data
        return False


register_annotator(SolmizationAnnotator)


class EvaluationAnnotator(Annotator):
    name = "evaluation"

    def get_lyric_annotations(
        self,
        notes,
        evaluation: EvaluationResult = None,
        use_color: bool = True,
    ):
        if evaluation is None:
            raise ValueError(
                "EvaluationAnnotator requires a results argument to annotate."
            )
        if not isinstance(evaluation, EvaluationResult):
            raise ValueError("Results should be an EvaluationResult object.")

        annotations = []
        for pred, result in zip(evaluation.predictions, evaluation):
            annot = dict(text=pred)
            if use_color:
                annot["color"] = EvalColors[result]
            annotations.append(annot)
        return annotations

    def get_editorial_annotations(
        self, notes, evaluation: EvaluationResult = None, **kws
    ):
        if evaluation is None:
            raise ValueError(
                "EvaluationAnnotator requires a results argument to annotate."
            )
        if not isinstance(evaluation, EvaluationResult):
            raise ValueError("Results should be an EvaluationResult object.")

        annotations = []
        for pred, result in zip(evaluation.predictions, evaluation):
            annot = dict(solmization=pred, status=result)
            annotations.append(annot)
        return annotations


register_annotator(EvaluationAnnotator)


class TextAnnotator(Annotator):
    name = "text"

    def get_lyric_annotations(self, notes, text: t.Iterable[str], **kws):
        return [dict(text=t, **kws) for t in text]


register_annotator(TextAnnotator)


class SegmentsAnnotator(Annotator):
    """Annotates the segments when using a SegmentedPathfinder
    by adding a dashed line over all notes in a segment. Only works
    if the solmizer uses a SegmentedPathfinder.
    """

    name = "segments"

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

    def annotate(self, notes=None, **kws):
        if notes is None:
            notes = self.notes

        pathfinder = self.solmizer.pathfinder
        input_times = self.solmizer.rollout.input_timesteps
        for segment in pathfinder.segments:
            segment_times = range(segment.start, segment.end + 1)
            intersect = [t for t in segment_times if t in input_times]
            if len(intersect) > 0:
                self.annotate_segment([notes[input_times.index(t)] for t in intersect])


register_annotator(SegmentsAnnotator)
