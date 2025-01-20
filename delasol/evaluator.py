# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from functools import cached_property
from abc import ABC
from enum import Enum
from collections import Counter

# Library imports
from music21.note import Note

# Local imports
from delasol.constants import SYLLABLES
from delasol.utils.music import extract_lyrics


class EvalStatus(Enum):
    """Possible evaluation results.

    This enumeration defines the possible outcomes of evaluating a predicted
    syllable against a target annotation. The possible values are:

    - CORRECT: The prediction matches the target.
    - INCORRECT: The prediction does not match the target.
    - MISSING: The target syllable is missing.
    - DELETION: The prediction is missing.
    - INSERTION: An extra syllable is predicted.
    - SKIP: The evaluation is skipped due to uncertainty.
    """

    CORRECT = "correct"
    INCORRECT = "incorrect"
    MISSING = "missing"
    DELETION = "deletion"
    INSERTION = "insertion"
    SKIP = "skip"


class EvaluationResult(list):
    """
    Results of an evaluation.

    An evaluation result is essentially a list with the EvalStatus
    for every prediction, and behaves like a list. The predictions,
    targets and options are stored as attributes, and there is a
    method that returns a dictionary with the counts of each EvalStatus.

    Parameters
    ----------
    *args : tuple
        Individual evaluation results; pased to the list initializer.
    predictions : array-like
        The predicted values. Must not be None.
    targets : array-like
        The true values. Must not be None.
    options : dict, optional
        Additional options for configuration. Default is an empty dictionary.

    Attributes
    ----------
    predictions : array-like
        The predicted values.
    targets : array-like
        The true values.
    options : dict
        Options passed to the evaluator; stored for reference.

    Raises
    ------
    ValueError
        If predictions or targets are None.
    """

    def __init__(self, *args, predictions=None, targets=None, options={}):
        if predictions is None:
            raise ValueError("Predictions cannot be None")
        self.predictions = predictions
        if targets is None:
            raise ValueError("Targets cannot be None")
        self.targets = targets
        self.options = options
        super().__init__(*args)

    @property
    def counts(self) -> dict[EvalStatus, int]:
        """Get the counts of each evaluation status.

        This method counts the occurrences of each status in the current instance
        and returns a dictionary mapping each `EvalStatus` to its corresponding
        count.

        Returns
        -------
        dict[EvalStatus, int]
            A dictionary where the keys are instances of `EvalStatus` and the
            values are the counts of each status.
        """
        counter = Counter(self)
        return {status: counter[status] for status in EvalStatus}

    def count(self, what: str) -> int:
        """Count the occurrences of a specified status.

        Parameters
        ----------
        what : str
            The status to count, specified as a string. This will be converted to
            uppercase to match the keys in the EvalStatus enumeration.

        Returns
        -------
        int
            The count of occurrences for the specified status.

        Raises
        ------
        KeyError
            If the provided status does not exist in the EvalStatus enumeration.
        """
        status = EvalStatus[what.upper()]
        return self.counts[status]

    def export(self):
        counts_export = dict()
        counts = self.counts
        for name, const in EvalStatus.__members__.items():
            counts_export[name.lower()] = counts[const]

        return dict(
            predictions=self.predictions,
            targets=self.targets,
            counts=counts_export,
            options=self.options,
        )


class Evaluator(ABC):

    @staticmethod
    def parse_annotation(annotation: str) -> dict:
        """Parse an annotated syllable.

        - `?`: No syllable in the source, no editorial suggestion.
        - `ut`: No uncertainty about 'ut' in the source, no editorial correction
        - `?ut`: Uncertainty about 'ut' in the source, no editorial correction.
        - `?[ut]`: Syllable missing in the source, editor suggests 'ut'.
        - `ut[re]`: No uncertainty about 'ut' in the source; editor suggests 're' instead.
        - `?ut[re]`: Some uncertainty about 'ut' in the source, editor suggests 're'

        Parameters
        ----------
        syllable : str
            A string representing the syllable annotation to be parsed. The
            expected format includes an optional ambiguity indicator ('?'),
            a syllable source (one of 'ut', 're', 'mi', 'fa', 'sol', 'la'),
            and an optional editorial suggestion.

        Returns
        -------
        dict
            A dictionary containing the parsed components:
            - 'uncertain' (bool): Indicates if there is uncertainty about the
            syllable annotated in the source.
            - 'syllable_source' (str or None): The source syllable, or None if
            not present.
            - 'syllable_editor' (str or None): The editor syllable, or None if
            not present.

        Raises
        ------
        ValueError
            If the input syllable cannot be parsed according to the expected
            format.
        """
        output = dict(uncertain=None, syllable_source=None, syllable_editor=None)

        # Extract and verify the editorial suggestion
        if "[" in annotation:
            source, editor = annotation.split("[")
            if not editor.endswith("]"):
                raise ValueError(
                    "Invalid annotation: editorial syllable should end with square bracket"
                )
            else:
                editor = editor[:-1]

            if editor == "":
                raise ValueError(
                    "Invalid annotation: the square brackets cannot be empty"
                )
            elif editor not in SYLLABLES:
                raise ValueError(
                    f"Invalid annotation: unknown editorial syllable: '{editor}'"
                )
            else:
                output["syllable_editor"] = editor
        else:
            output["syllable_editor"] = None
            source = annotation

        # Check whether the source syllable is uncertain
        if source == "":
            raise ValueError("Invalid annotation: the source syllable cannot be empty")
        elif source.startswith("?"):
            output["uncertain"] = True
            source = source[1:]
        else:
            output["uncertain"] = False

        # Process the source syllable
        if len(source) == 0:
            output["syllable_source"] = None
        elif source not in SYLLABLES:
            raise ValueError(f"Invalid annotation: unknown source syllable {source}")
        else:
            output["syllable_source"] = source

        return output

    @staticmethod
    def get_targets(notes: t.Iterable[Note], lyric_number: int) -> list[str]:
        """Get target lyrics or strings from the provided input.

        Parameters
        ----------
        input : Iterable[Note] | Iterable[str]
            An iterable containing either Note objects or strings. If it contains
            Note objects, lyrics will be extracted from the specified target lyrics
            line. If it contains strings, they will be returned as is.
        target_lyric_number : int, optional
            The line number of the lyrics to extract. If provided, the function
            will attempt to extract targets from the lyrics of the input.

        Returns
        -------
        list[str]
            A list of strings representing either the extracted lyrics or the
            input strings.

        Raises
        ------
        ValueError
            If the input is not a valid type (neither Note nor str) or if
            attempting to extract lyrics from a non-Note iterable.
        """
        raise DeprecationWarning

        if not isinstance(input[0], Note):
            raise ValueError(
                "Can only extract lyrics when the input is an iterable of notes"
            )
        return extract_lyrics(input, number=lyric_number)

    @staticmethod
    def evaluate_prediction(
        prediction: str,
        target: str,
        use_editorial_suggestion: bool = True,
        skip_uncertain: bool = True,
        detect_insertions_deletions: bool = True,
    ) -> EvalStatus:
        """Evaluate the prediction against the target and return the evaluation result.

        Parameters
        ----------
        prediction : str
            The predicted syllable or string to evaluate.
        target : str
            The target syllable or string to compare against.
        use_editorial_suggestion : bool, optional
            If True, use the editorial suggestion from the target annotation.
            Default is False.
        skip_uncertain : bool, optional
            If True, skip evaluation for uncertain annotations. Default is False.
        detect_insertions_deletions : bool, optional
            If True, detect insertions and deletions in the evaluation. Default is True.

        Returns
        -------
        Literal[EvalStatus.CORRECT, EvalStatus.INCORRECT, EvalStatus.MISSING,
                EvalStatus.DELETION, EvalStatus.INSERTION, EvalStatus.SKIP]
            The result of the evaluation, indicating whether the prediction is
            correct, incorrect, missing, a deletion, an insertion, or should be skipped.
        """

        # If no target_annotation is given: an insertion
        if target is None and prediction is not None:
            return (
                EvalStatus.INSERTION
                if detect_insertions_deletions
                else EvalStatus.INCORRECT
            )

        # Parse the annotated target syllable
        annot = Evaluator.parse_annotation(target)
        target_syllable = (
            annot["syllable_editor"]
            if use_editorial_suggestion and annot["syllable_editor"]
            else annot["syllable_source"]
        )

        # Evaluate the type of error
        if annot["uncertain"] and skip_uncertain:
            return EvalStatus.SKIP
        elif annot["uncertain"] and target_syllable is None:
            return EvalStatus.MISSING
        elif prediction == target_syllable:
            return EvalStatus.CORRECT
        elif prediction == "" or prediction is None:
            return (
                EvalStatus.DELETION
                if detect_insertions_deletions
                else EvalStatus.INCORRECT
            )
        else:
            return EvalStatus.INCORRECT

    @staticmethod
    def evaluate(
        predictions: t.Iterable[str],
        targets: t.Iterable[str] = None,
        **evaluate_prediction_kws,
    ) -> EvaluationResult:
        """Evaluate predictions against targets and return results.

        Parameters
        ----------
        predictions : Iterable[str]
            A collection of predicted values as strings.
        targets : Iterable[str], optional
            A collection of true values as strings. If provided, must have the
            same length as `predictions`. Default is None.
        evaluate_prediction_kws : dict
            Optional keyword arguments passed to `evaluate_prediction`. These
            include `use_editorial_suggestion`, `skip_uncertain`, and `detect_insertions_deletions`.

        Returns
        -------
        EvaluationResult
            A list of evaluation results for each prediction. Besides, the
            EvaluationResult object contains the predictions, targets and
            keyword arguments.

        Raises
        ------
        ValueError
            If the length of `predictions` and `targets` do not match.
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets should have the same length")

        results = EvaluationResult(
            predictions=predictions, targets=targets, options=evaluate_prediction_kws
        )
        for prediction, target in zip(predictions, targets):
            result = Evaluator.evaluate_prediction(
                prediction, target, **evaluate_prediction_kws
            )
            results.append(result)
        return results


########################## Registry ##########################

# TODO this whole registry might be pointless?

EVALUATORS = {}


def register_evaluator(evaluator: Evaluator):
    if not issubclass(evaluator, Evaluator):
        raise ValueError("An evaluator must be an subclas of the Evaluator class")
    if not hasattr(evaluator, "name"):
        raise ValueError("An Evaluator must have a 'name' attribute.")

    EVALUATORS[evaluator.name] = evaluator


def get_evaluator(name: str) -> Evaluator:
    evaluator_class = EVALUATORS.get(name)
    if not evaluator_class:
        raise ValueError(f"Evaluator '{name}' not found.")
    return evaluator_class()


########################## Evaluators ##########################


class DefaultEvaluator(Evaluator):
    name = "default"


register_evaluator(DefaultEvaluator)
