# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
import os
import pandas as pd
from music21.stream import Stream

# Local imports
from solmization.corpus import Corpus

TEST_CORPUS = "marot-de-beze"
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))


class TestCorpus(unittest.TestCase):
    def test_init_with_name(self):
        corpus = Corpus(TEST_CORPUS)
        self.assertIsInstance(corpus, Corpus)
        self.assertEqual(corpus.name, TEST_CORPUS)

    def test_init_with_dir(self):
        directory = os.path.join(ROOT_DIR, "corpora", TEST_CORPUS)
        corpus = Corpus(directory=directory)
        self.assertIsInstance(corpus, Corpus)
        self.assertEqual(corpus.name, TEST_CORPUS)

    def test_metadata(self):
        corpus = Corpus(TEST_CORPUS)
        meta = corpus.metadata
        self.assertEqual(meta["name"], TEST_CORPUS)

    def test_files(self):
        corpus = Corpus(TEST_CORPUS)
        files = corpus.files("musescore")
        self.assertEqual(len(files), len(corpus))
        self.assertTrue(all([os.path.exists(f) for f in files.values()]))

    @unittest.skip
    def test_musicxml_conversion(self):
        corpus = Corpus(TEST_CORPUS)
        id = corpus.ids[0]
        corpus.convert_musescore_files(to="musicxml", ids=[id], refresh=True)
        self.assertTrue(os.path.exists(corpus.works[id]["_musicxml"]))

    @unittest.skip
    def test_musescore_conversion(self):
        corpus = Corpus(TEST_CORPUS)
        id = corpus.ids[0]
        corpus.convert_musescore_to_other_formats(ids=[id], refresh=True)
        for format in corpus.formats.keys():
            self.assertTrue(os.path.exists(corpus.works[id][f"_{format}"]))

    def test_lyric_number(self):
        corpus = Corpus(TEST_CORPUS)
        self.assertEqual(corpus.lyric_number["text"], 1)
        self.assertEqual(corpus.lyric_number["syllables"], 2)

    @unittest.skip
    def test_evaluate_work(self):
        corpus = Corpus(TEST_CORPUS)
        id = corpus.ids[0]
        score, evaluation = corpus.evaluate_work(id=id, target_lyrics="syllables")
        self.assertIsInstance(score, Stream)

    @unittest.skip
    def test_evaluate(self):
        corpus = Corpus(TEST_CORPUS)
        id = corpus.ids[0]
        df = corpus.evaluate(ids=[id])
        self.assertEqual(len(df), 1)
        self.assertTrue("perc_missing" in df)

    @unittest.skip
    def test_load_evaluations(self):
        corpus = Corpus(TEST_CORPUS)
        df = corpus.load_evaluation()
        self.assertIsInstance(df, pd.DataFrame)

    def test_evaluate_without_output(self):
        corpus = Corpus(TEST_CORPUS)
        ids = corpus.ids[:5]
        df = corpus.evaluate(ids=ids, write_output=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(ids))

    def test_davantes(self):
        corpus = Corpus("davantes")
        ids = corpus.ids[:1]
        df = corpus.evaluate(
            ids=ids,
            target_lyrics="davantes_numbering",
            style="davantes",
            output_style="davantes",
        )
        self.assertTrue(all(df["correct"] > 50))

    def test_smith(self):
        corpus = Corpus("smith")
        df = corpus.evaluate(ids=corpus.ids[:2])
        self.assertEqual(len(df), 2)
