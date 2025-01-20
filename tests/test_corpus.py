# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
import os, shutil
import pandas as pd

# Local imports
from delasol.corpus import Corpus, Collection, Work
from delasol.evaluator import EvalStatus

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))

# An actual corpus
DELASOL_CORPUS_DIR = os.environ["DELASOL_CORPUS"]
MDB_COLLECTION_DIR = os.path.join(
    DELASOL_CORPUS_DIR, "collections", "marot-de-beze-1562"
)

# Test collection directory: this is used for testing the Collection class
# Note that all files in that directory may be removed/overwritten by
# the tests
TEST_COLLECTION_DIR = os.path.join(CUR_DIR, "test_collection")


class TestCorpus(unittest.TestCase):

    def test_init(self):
        corpus = Corpus(DELASOL_CORPUS_DIR)
        self.assertIn("marot-de-beze-1562", corpus.collections)
        self.assertIsInstance(corpus.collections["marot-de-beze-1562"], Collection)

    def test_init_empty(self):
        self.assertIn("DELASOL_CORPUS", os.environ)
        corpus = Corpus()
        self.assertIsInstance(corpus, Corpus)


class TestCollection(unittest.TestCase):
    def test_environ_variable(self):
        # These tests assume the DELASOL_CORPUS directory is set
        self.assertIn("DELASOL_CORPUS", os.environ)

    def test_init_with_name(self):
        collection = Collection(name="marot-de-beze-1562")
        self.assertIsInstance(collection, Collection)
        self.assertEqual(collection.name, "marot-de-beze-1562")

    def test_metadata(self):
        collection = Collection(TEST_COLLECTION_DIR)
        meta = collection.metadata
        self.assertEqual(meta["name"], "test_collection")

    def test_musicxml_conversion(self):
        corpus = Collection(TEST_COLLECTION_DIR)
        id = corpus.ids[0]
        xml_path = os.path.join(TEST_COLLECTION_DIR, "musicxml", f"{id}.musicxml")
        if os.path.exists(xml_path):
            os.remove(xml_path)
        success, _ = corpus.convert(to="musicxml", ids=[id], refresh=True)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(xml_path))

    def test_lyric_number(self):
        corpus = Collection(TEST_COLLECTION_DIR)
        self.assertEqual(corpus.lyric_number["original_text"], 1)
        self.assertEqual(corpus.lyric_number["modernized_text"], 2)
        self.assertEqual(corpus.lyric_number["syllables"], 3)

    def test_evaluate(self):
        corpus = Collection(TEST_COLLECTION_DIR)
        corpus.convert(to="musicxml", refresh=False)
        df, _ = corpus.evaluate(style="continental_16c", target_lyric_number=3)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0, 0], 62)
        self.assertEqual(df.iloc[1, 0], 85)

    @unittest.skip
    def test_load_evaluations(self):
        corpus = Collection(TEST_COLLECTION_DIR)
        df = corpus.load_evaluation()
        self.assertIsInstance(df, pd.DataFrame)

    @unittest.skip
    def test_evaluate_without_output(self):
        corpus = Collection(TEST_COLLECTION_DIR)
        ids = corpus.ids[:5]
        df = corpus.evaluate(ids=ids, write_output=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(ids))

    @unittest.skip
    def test_davantes(self):
        corpus = Collection("davantes")
        ids = corpus.ids[:1]
        df = corpus.evaluate(
            ids=ids,
            target_lyrics="davantes_numbering",
            style="davantes",
            output_style="davantes",
        )
        self.assertTrue(all(df["correct"] > 50))

    @unittest.skip
    def test_smith(self):
        corpus = Collection("smith")
        df = corpus.evaluate(ids=corpus.ids[:2])
        self.assertEqual(len(df), 2)


class TestWork(unittest.TestCase):
    def test_init(self):
        work = Work("MdB001", TEST_COLLECTION_DIR)
        self.assertEqual(work.id, "MdB001")

        ms_path = os.path.join(TEST_COLLECTION_DIR, "musescore", "MdB001.mscz")
        self.assertEqual(work.path("musescore", verify_exists=False), ms_path)

        xml_path = os.path.join(TEST_COLLECTION_DIR, "musicxml", "MdB001.musicxml")
        self.assertEqual(work.path("musicxml", verify_exists=False), xml_path)

    def test_convert_xml(self):
        xml_path = os.path.join(TEST_COLLECTION_DIR, "musicxml", "MdB001.musicxml")
        if os.path.exists(xml_path):
            os.remove(xml_path)

        work = Work("MdB001", TEST_COLLECTION_DIR)
        work.convert(to="musicxml")
        self.assertTrue(os.path.exists(xml_path))

    def test_convert_pdf(self):
        pdf_path = os.path.join(TEST_COLLECTION_DIR, "pdf", "MdB001.pdf")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        work = Work("MdB001", TEST_COLLECTION_DIR)
        work.convert(to="pdf")
        self.assertTrue(os.path.exists(pdf_path))

    def test_missing_output_dir(self):
        pdf_dir = os.path.join(TEST_COLLECTION_DIR, "pdf")
        shutil.rmtree(pdf_dir)
        self.test_convert_pdf()

    def test_evaluate(self):
        work = Work("MdB001", TEST_COLLECTION_DIR)
        result, _ = work.evaluate(style="continental_16c", target_lyric_number=3)
        self.assertEqual(result.counts[EvalStatus.CORRECT], 62)

    def test_evaluate_no_musicxml(self):
        work = Work("MdB001", TEST_COLLECTION_DIR)
        if os.path.exists(work.musicxml_path):
            os.remove(work.musicxml_path)
        with self.assertRaises(FileNotFoundError):
            result, _ = work.evaluate(style="continental_16c", target_lyric_number=3)


class TestActualCorpus(unittest.TestCase):

    def test_conversion(self):
        corpus = Corpus()
        collection = corpus.get_collection("marot-de-beze-1562")
        ids = collection.ids[:3]
        success, results = collection.convert(to="musicxml", ids=ids)
        self.assertTrue(success)
        for id in ids:
            xml_path = collection.get_work(id).musicxml_path
            self.assertTrue(os.path.exists(xml_path))

    def test_evaluation(self):
        corpus = Corpus()
        collection = corpus.get_collection("marot-de-beze-1562")
        ids = collection.ids
        df, log = collection.evaluate(
            style="continental_16c", ids=ids, target_lyric_number=3
        )
        self.assertEqual(len(log["errors"]), 0)

    def test_evaluation_output(self):
        corpus = Corpus()
        collection = corpus.get_collection("marot-de-beze-1562")
        ids = collection.ids[5:10]
        df, log = collection.evaluate(
            style="continental_16c",
            ids=ids,
            target_lyric_number=3,
            write_output=True,
            write_log=True,
            refresh=True,
        )
        self.assertEqual(len(log["errors"]), 0)
