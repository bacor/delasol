# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest

# Local imports
from delasol.utils.music import as_pitch_list, as_stream
from delasol.annotator import get_annotator
from delasol.solmizers.solmizer import get_solmizer
from delasol.evaluator import EvalStatus


class TestAnnotator(unittest.TestCase):
    def test_require_stream(self):
        input = as_pitch_list("G3 A3 C4 B3 G3")
        solmizer = get_solmizer("continental_16c", input, key=0)
        with self.assertRaises(ValueError):
            annotator = get_annotator("best_path", solmizer)

    def test_init(self):
        input = as_stream("G3 A3 C4 B3 G3")
        targets = ["ut", "re", "fa", "mi", "ut"]
        solmizer = get_solmizer("continental_16c", input, key=0)
        solmizer.annotate("text", text=targets)

        results = solmizer.evaluate(targets=targets)
        solmizer.annotate("evaluation", evaluation=results)

        results = solmizer.evaluate(targets=targets, rank=1)
        solmizer.annotate("evaluation", evaluation=results)

        results = solmizer.evaluate(targets=targets, rank=2)
        solmizer.annotate("evaluation", evaluation=results)

        self.assertTrue(True)

    def test_editorial(self):
        input = as_stream("G3 A3 C4 B3 G3")
        targets = ["ut", "re", "fa", "mi", "ut"]
        solmizer = get_solmizer("continental_16c", input, key=0)
        results = solmizer.evaluate(targets=targets)
        solmizer.annotate("evaluation", evaluation=results, key_prefix="foo_")
        self.assertDictEqual(
            solmizer.input[0].editorial,
            {"foo_solmization": "ut", "foo_status": EvalStatus.CORRECT},
        )
