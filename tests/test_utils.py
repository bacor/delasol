# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest

# Local imports
from alamire.utils import segment_deviations


class TestSegmentation(unittest.TestCase):
    def test_constant_sequence(self):
        for length in range(1, 10):
            example = [1] * length
            segments = segment_deviations(example, 1)
            self.assertEqual(segments, [(0, length - 1)])

    def test_constant_tail(self):
        example = [1, 2, 1, 1, 1, 1]
        segments = segment_deviations(example, 1)
        self.assertEqual(segments, [(0, 2), (3, 5)])

    def test_one_segment(self):
        example = [1, 2, 1]
        segments = segment_deviations(example, 1)
        self.assertEqual(segments, [(0, 2)])

    def test_no_matches(self):
        example = [2, 2, 2]
        segments = segment_deviations(example, 1)
        self.assertEqual(segments, [(0, 2)])

    def test_deviant_opening_and_end(self):
        example = [2, 2, 1, 1, 2]
        segments = segment_deviations(example, 1)
        self.assertEqual(segments, [(0, 2), (3, 4)])

    def test_issue_1(self):
        issue = [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1]
        segments = segment_deviations(issue, 1)
        self.assertEqual(segments, [(0, 3), (4, 7), (8, 11)])
