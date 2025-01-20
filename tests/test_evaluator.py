# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 202 Bas Cornelissen
# -------------------------------------------------------------------
import unittest

# Local imports
from delasol.evaluator import Evaluator, EvalStatus, EvaluationResult


class TestEvaluationResults(unittest.TestCase):
    def test_counts(self):
        predictions = ["re", "mi", "fa", "sol"]
        targets = ["re", "la", "?fa", "sol"]
        comparisons = [
            EvalStatus.CORRECT,
            EvalStatus.INCORRECT,
            EvalStatus.SKIP,
            EvalStatus.CORRECT,
        ]
        results = EvaluationResult(
            comparisons,
            predictions=predictions,
            targets=targets,
        )

        expected = {
            EvalStatus.CORRECT: 2,
            EvalStatus.INCORRECT: 1,
            EvalStatus.MISSING: 0,
            EvalStatus.DELETION: 0,
            EvalStatus.INSERTION: 0,
            EvalStatus.SKIP: 1,
        }
        self.assertDictEqual(results.counts, expected)
        self.assertEqual(results.count("skip"), 1)


class TestEvaluator(unittest.TestCase):
    def test_parse_annotation(self):
        parse_annotation = Evaluator.parse_annotation

        with self.assertRaises(ValueError):
            parse_annotation("")

        with self.assertRaises(ValueError):
            parse_annotation("fi")

        with self.assertRaises(ValueError):
            parse_annotation("[fi]")

        self.assertDictEqual(
            parse_annotation("?"),
            dict(uncertain=True, syllable_source=None, syllable_editor=None),
        )

        self.assertDictEqual(
            parse_annotation("?[la]"),
            dict(uncertain=True, syllable_source=None, syllable_editor="la"),
        )

        self.assertDictEqual(
            parse_annotation("re"),
            dict(uncertain=False, syllable_source="re", syllable_editor=None),
        )

        self.assertDictEqual(
            parse_annotation("mi[fa]"),
            dict(uncertain=False, syllable_source="mi", syllable_editor="fa"),
        )

        self.assertDictEqual(
            parse_annotation("?ut"),
            dict(uncertain=True, syllable_source="ut", syllable_editor=None),
        )

        self.assertDictEqual(
            parse_annotation("?fa[sol]"),
            dict(uncertain=True, syllable_source="fa", syllable_editor="sol"),
        )

    def test_evaluate_prediction(self):
        eval_pred = Evaluator.evaluate_prediction

        self.assertEqual(eval_pred("la", "la"), EvalStatus.CORRECT)
        self.assertEqual(eval_pred("la", "fa"), EvalStatus.INCORRECT)

        self.assertEqual(eval_pred("la", "?fa", skip_uncertain=True), EvalStatus.SKIP)
        self.assertEqual(
            eval_pred("la", "?fa", skip_uncertain=False), EvalStatus.INCORRECT
        )

        self.assertEqual(
            eval_pred(
                "la", "?[la]", use_editorial_suggestion=False, skip_uncertain=False
            ),
            EvalStatus.MISSING,
        )
        self.assertEqual(
            eval_pred(
                "la", "?[la]", use_editorial_suggestion=True, skip_uncertain=False
            ),
            EvalStatus.CORRECT,
        )
        self.assertEqual(
            eval_pred(
                "la", "?[mi]", use_editorial_suggestion=True, skip_uncertain=False
            ),
            EvalStatus.INCORRECT,
        )

        self.assertEqual(
            eval_pred(
                "la", "?la[mi]", use_editorial_suggestion=False, skip_uncertain=False
            ),
            EvalStatus.CORRECT,
        )
        self.assertEqual(
            eval_pred(
                "la", "?la[mi]", use_editorial_suggestion=True, skip_uncertain=False
            ),
            EvalStatus.INCORRECT,
        )

        self.assertEqual(eval_pred("", "la"), EvalStatus.DELETION)
        self.assertEqual(eval_pred(None, "la"), EvalStatus.DELETION)
        self.assertEqual(eval_pred("la", None), EvalStatus.INSERTION)
