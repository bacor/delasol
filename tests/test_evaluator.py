# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 202 Bas Cornelissen
# -------------------------------------------------------------------
import unittest

# Local imports
from delasol.evaluator import Evaluator, EvalResult


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

        self.assertEqual(eval_pred("la", "la"), EvalResult.CORRECT)
        self.assertEqual(eval_pred("la", "fa"), EvalResult.INCORRECT)

        self.assertEqual(eval_pred("la", "?fa", skip_uncertain=True), EvalResult.SKIP)
        self.assertEqual(
            eval_pred("la", "?fa", skip_uncertain=False), EvalResult.INCORRECT
        )

        self.assertEqual(
            eval_pred("la", "?[la]", use_editorial_suggestion=False), EvalResult.MISSING
        )
        self.assertEqual(
            eval_pred("la", "?[la]", use_editorial_suggestion=True), EvalResult.CORRECT
        )
        self.assertEqual(
            eval_pred("la", "?[mi]", use_editorial_suggestion=True),
            EvalResult.INCORRECT,
        )

        self.assertEqual(
            eval_pred("la", "?la[mi]", use_editorial_suggestion=False),
            EvalResult.CORRECT,
        )
        self.assertEqual(
            eval_pred("la", "?la[mi]", use_editorial_suggestion=True),
            EvalResult.INCORRECT,
        )

        self.assertEqual(eval_pred("", "la"), EvalResult.DELETION)
        self.assertEqual(eval_pred(None, "la"), EvalResult.DELETION)
        self.assertEqual(eval_pred("la", None), EvalResult.INSERTION)
