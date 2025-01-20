# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from functools import cached_property

import os
import yaml
from datetime import datetime
import pandas as pd
import music21
from music21.metadata import Metadata
from tqdm.auto import tqdm

from delasol import solmize
from delasol.utils.musescore import run_mscore
from delasol.solmizers.solmizer import get_solmizer
from delasol.evaluator import EvalStatus, EvaluationResult

# MSCORE_EXECUTABLE = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))


class Corpus:
    """
    A corpus class.

    A corpus consists of a collection of works. The corpus class is mostly
    used to automaticallly retrieve a list of available collections.

    Parameters
    ----------
    corpus_dir : str, optional
        The path to the directory containing the corpus. If not provided,
        the `get_corpus_dir` methods will attempt to locate a corpus using
        environment variables.

    Attributes
    ----------
    dir : str
        The directory path for the corpus.
    collections_dir : str
        The path to the collections directory within the corpus directory.
    """

    def __init__(self, corpus_dir: str = None):
        self.dir = self.get_corpus_dir(corpus_dir)
        self.collections_dir = os.path.join(self.dir, "collections")

    def __repr__(self) -> str:
        dirname = os.path.split(self.dir)[1]
        return f"<Corpus '{dirname}' with {len(self)} collections>"

    def __len__(self) -> int:
        """Returns the number of collections in the corpus"""
        return len(self.collections)

    @cached_property
    def collections(self) -> t.Dict[str, t.Type["Collection"]]:
        """A dictionary of collections found in the corpus.

        This method walks through the `collections` subdirectory and tries to
        instantiate a `Collection` object for each subdirectory.  If successful,
        the collection is added, using the name as a key.

        Returns
        -------
        dict
            A dictionary where the keys are collection names (str) and the values
            are instances of the `Collection` class.
        """
        collections = {}
        for subdir, _, _ in os.walk(self.collections_dir):
            try:
                collection = Collection(subdir)
                collections[collection.name] = collection
            except:
                # Not a collection
                pass
        return collections

    @staticmethod
    def get_corpus_dir(directory: str = None):
        """Get the absolute path of the corpus directory.

        This function retrieves the corpus directory path. If the `directory`
        parameter is not provided, it checks the environment variable
        `DELASOL_CORPUS`. If neither is available, it raises a ValueError.
        The function also verifies the existence of the specified directory and
        the presence of a 'collections' subdirectory, raising a FileNotFoundError
        if either check fails.

        Parameters
        ----------
        directory : str, optional
            The path to the corpus directory. If not provided, the function
            will look for the `DELASOL_CORPUS` environment variable.

        Returns
        -------
        str
            The absolute path of the corpus directory.

        Raises
        ------
        ValueError
            If the directory is not specified and the environment variable
            `DELASOL_CORPUS` is not set.

        FileNotFoundError
            If the specified directory does not exist or does not contain a
            'collections' directory.
        """
        if directory is None and "DELASOL_CORPUS" in os.environ:
            dir = os.environ["DELASOL_CORPUS"]
            if dir.startswith(".."):
                dir = os.path.abspath(os.path.join(ROOT_DIR, dir))
            else:
                dir = os.path.abspath(dir)
            directory = dir
        elif directory is None:
            raise ValueError(
                "Please specify the delasol corpus directory, either directly or by setting the DELASOL_CORPUS environment variable"
            )
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Corpus directory not found: {directory}")
        if not os.path.exists(os.path.join(directory, "collections")):
            raise FileNotFoundError(
                f"Corpus directory does not contain a 'collections' directory: {directory}"
            )
        return os.path.abspath(directory)

    def get_collection(self, name: str):
        """Get a collection from the specified name.

        Parameters
        ----------
        name : str
            The name of the collection to be created. This name is used to
            construct the directory path where the collection is stored.

        Returns
        -------
        Collection
            An instance of the Collection class initialized with the path
            to the specified collection directory.

        Raises
        ------
        FileNotFoundError
            If the directory for the specified collection does not exist.
        """
        if not name in self.collections:
            raise FileNotFoundError(f"Collection not found: {name}")
        return self.collections[name]


class Work:
    """
    A work class

    Parameters
    ----------
    id : str
        A unique identifier for the instance.
    collection_dir : str
        The path to the collection directory. This directory must exist.
    metadata : dict, optional
        A dictionary containing metadata associated with the instance.
        Defaults to an empty dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified collection directory does not exist.
    """

    supported_formats = {
        "musescore": "mscz",
        "musicxml": "musicxml",
        "pdf": "pdf",
    }

    def __init__(self, id: str, collection_dir: str, metadata: dict = {}):
        if not os.path.exists(collection_dir):
            raise FileNotFoundError(f"Collection directory not found: {collection_dir}")
        self.collection_dir = collection_dir
        self.id = id
        self.metadata = metadata

    def __repr__(self):
        return f"<Work {self.id}>"

    def path(self, format: str, verify_exists: bool = False) -> str:
        """Generate the file path for a specified format of the music file.

        Parameters
        ----------
        format : str
            The format of the music file. Supported formats are "musescore",
            "musicxml", and "pdf".

        verify_exists : bool, optional
            If True, the function will check if the file exists at the generated
            path. Default is False.

        Returns
        -------
        str
            The full path to the music file in the specified format.

        Raises
        ------
        ValueError
            If the provided format is not supported.

        FileNotFoundError
            If verify_exists is True and the file does not exist at the generated
            path.
        """
        if format not in ["musescore", "musicxml", "pdf"]:
            raise ValueError(f"Unsupported format: {format}")
        ext = self.supported_formats[format]
        path = os.path.join(self.collection_dir, format, f"{self.id}.{ext}")
        if verify_exists and not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found (id={self.id}, format={format}): {path}"
            )
        return path

    @cached_property
    def musescore_path(self) -> str:
        """Path to the MuseScore file."""
        return self.path("musescore")

    @cached_property
    def musicxml_path(self) -> str:
        """Path to the musicxml file."""
        return self.path("musicxml")

    @cached_property
    def pdf_path(self) -> str:
        """Path to the PDF file."""
        return self.path("pdf")

    def convert(
        self, to: str, refresh: bool = False
    ) -> t.Type["subprocess.CompletedProcess"] | bool:
        """Convert a music score to a specified format using MuseScore.

        Parameters
        ----------
        to : str
            The target format to convert the music score to (e.g., 'musicxml').
        refresh : bool, optional
            If True, forces a refresh of the conversion even if the target file
            already exists. Default is False.

        Returns
        -------
        subprocess.CompletedProcess or bool
            Returns a CompletedProcess object if the conversion is successful,
            or False if the target file already exists and refresh is not requested.

        Raises
        ------
        Exception
            If the conversion fails, an exception is raised with an error message
            indicating the failure reason.
        """
        source = self.path("musescore", verify_exists=True)
        target = self.path(to, verify_exists=False)
        if not os.path.exists(target) or refresh:

            # Create output directory if it does not exist
            output_dir = os.path.dirname(target)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Convert
            result = run_mscore("-o", target, source)

            if result.returncode != 0:
                raise Exception(
                    f"Error converting work {self.id} to {to}: {result.stderr}"
                )
            else:
                return result
        else:
            return False

    def load(self, **kwargs) -> music21.stream.Score:
        """Load a score from a MusicXML file.

        Parameters
        ----------
        **kwargs : keyword arguments
            Additional parameters to be passed to the music21 converter.

        Returns
        -------
        music21.stream.Score
            The parsed score from the MusicXML file.
        """
        return music21.converter.parse(self.musicxml_path, **kwargs)

    def evaluate(
        self,
        style: str = None,
        force_source: bool = False,
        annotate: bool = True,
        solmizer_kws: dict = {},
        annotator_kws: dict = {},
        **evaluator_kws,
    ):
        """Evaluate a score using the specified style and options.

        Parameters
        ----------
        style : str, optional
            The style to use for evaluation. If None, a default style will be
            applied.
        force_source : bool, optional
            If True, forces the loading of the source score even if it is already
            cached. Default is False.
        annotate : bool, optional
            If True, adds annotations to the evaluation results. Default is True.
        solmizer_kws : dict, optional
            Additional keyword arguments to pass to the solmizer during evaluation.
        annotator_kws : dict, optional
            Additional keyword arguments to pass to the annotator if annotations
            are enabled.
        **evaluator_kws : keyword arguments
            Additional keyword arguments to pass to the evaluator.

        Returns
        -------
        tuple
            A tuple containing the evaluation results and the solmizer instance
            used for the evaluation.

        Notes
        -----
        This function loads a score, evaluates it using the specified style, and
        optionally annotates the results. The evaluation results and the solmizer
        instance are returned for further processing.
        """
        score = self.load(forceSource=force_source)
        solmizer = solmize(score, style=style, **solmizer_kws)
        results = solmizer.evaluate(**evaluator_kws)
        if annotate:
            solmizer.annotate(
                "evaluation",
                evaluation=results,
                **annotator_kws,
            )

            # Annotate the evaluated score
            now = datetime.now().strftime("%d-%m-%Y")
            metadata = dict(title=id, composer=f"Generated on {now}")
            score.metadata = Metadata(**metadata)

        return results, solmizer


class Collection:

    supported_formats = Work.supported_formats

    def __init__(
        self,
        name: str = None,
        directory: str = None,
        formats: t.Iterable[t.Literal["musescore", "musicxml", "pdf"]] = [
            "musescore",
            "musicxml",
        ],
        metafile: str = "collection.yaml",
    ):

        if name is not None:
            # Directory, if set, must be corpus_dir:
            corpus_dir = Corpus.get_corpus_dir(directory)
            directory = os.path.join(corpus_dir, "collections", name)
        elif directory is None:
            raise ValueError(
                "Please specify the directory or the name of the collection"
            )
        # Else: directory is the collection directory; infer name later.

        # Check and save directory and name of the collection
        self.dir = os.path.abspath(directory)
        self.name = os.path.basename(directory)
        if not os.path.exists(self.dir):
            raise FileExistsError(
                f"Collection directory not found (corpus={self.name}): {self.dir}"
            )

        # Load metadata file
        self.metafile = os.path.join(self.dir, metafile)
        if not os.path.exists(self.metafile):
            raise Exception("Metadata file not found: {self.metafile}")

        # Initialize some attributes
        self.formats = {
            fmt: ext for fmt, ext in self.supported_formats.items() if fmt in formats
        }

        # Save directory structure
        self.dirs = dict(output=os.path.join(self.dir, "output"))
        for format in self.formats.keys():
            dir = os.path.join(self.dir, format)
            self.dirs[format] = dir
        for dir in self.dirs.values():
            os.makedirs(dir, exist_ok=True)

    def __len__(self):
        return len(self.works)

    def __repr__(self):
        return f"<Collection {self.name}>"

    # Properties

    @cached_property
    def metadata(self) -> dict:
        """Retrieve metadata from the collections.yaml metafile.

        This function reads the YAML metadata and returns its contents as a
        dictionary. If the 'name' key is not present in the metadata, it
        assigns the instance's name to this key.

        Returns
        -------
        dict
            A dictionary with collection metadata.
        """
        meta = {}
        with open(self.metafile, "r") as file:
            meta = yaml.safe_load(file)

        if not "name" in meta:
            meta["name"] = self.name

        # TODO: solmization style?
        return meta

    @cached_property
    def _works(self) -> t.Dict[str, Work]:
        """Check and retrieve works from the metadata.

        This private method iterates through the works defined in the metadata and
        creates a dictionary mapping work IDs to their corresponding Work objects.

        Returns
        -------
        dict
            A dictionary where the keys are work IDs (str) and the values are
            Work objects associated with those IDs.
        """
        works = {}
        for work in self.metadata["works"]:
            id = work["id"]
            works[id] = Work(collection_dir=self.dir, id=id, metadata=work)
        return works

    @property
    def works(self) -> t.List[Work]:
        """Returns a list works.

        Returns
        -------
        Iterable[Work]
            An iterable containing the Work objects.
        """
        return list(self._works.values())

    @property
    def ids(self) -> list[str]:
        """Returns a sorted list of work ids.

        Returns
        -------
        list
            A sorted list of keys from the `works` dictionary.
        """
        return sorted(list(self._works.keys()))

    @cached_property
    def lyric_number(self) -> t.Dict[str, int]:
        """Generate a dictionary mapping lyric types to their corresponding line
        numbers.

        Returns
        -------
        dict of {str: int}
            A dictionary where the keys are lyric types and the values are the
            corresponding line numbers (1-indexed) from the metadata's lyric lines.
        """
        lyrics = {
            type: line + 1 for line, type in enumerate(self.metadata["lyric_lines"])
        }
        return lyrics

    # Methods

    def get_work(self, id: str) -> Work:
        """Retrieve a Work object by its identifier.

        Parameters
        ----------
        id : str
            The identifier of the Work to retrieve.

        Returns
        -------
        Work
            The Work object associated with the given identifier.

        Raises
        ------
        KeyError
            If the specified id does not exist in the works collection.
        """
        return self._works[id]

    def convert(
        self,
        to: t.Iterable[str] | str = "musicxml",
        refresh: bool = False,
        ids: list[str] = None,
    ) -> t.Tuple[bool, t.Dict[str, bool | str]]:
        """Convert multiple works to the specified formats.

        Parameters
        ----------
        to : iterable of str or str, optional
            The format(s) to convert the works to. Default is 'musicxml'. You can
            either specify a single string or a list of strings.

        refresh : bool, optional
            If True, refresh the work data before conversion. Default is False.

        ids : list of str, optional
            A list of work IDs to convert. If None, all works will be converted.

        Returns
        -------
        (success, results) : tuple
            A tuple containing:
            - success : bool
                True if all conversions succeeded, False otherwise.
            - results : dict
                A dictionary mapping each work ID to a boolean indicating success
                or a string with the error message if the conversion failed.

        Raises
        ------
        ValueError
            If a specified format is not supported or not available for the
            current collection.
        """

        if ids is None:
            ids = self.ids
        if isinstance(to, str):
            to = [to]
        results = {}
        for id in tqdm(ids):
            for format in to:
                if format not in self.supported_formats:
                    raise ValueError(f"Format {format} is not supported.")
                elif format not in self.formats:
                    raise ValueError(
                        f"Format {format} is supported, but not available to this collection. To make it available, specify the format when initializing the class using the 'format' argument."
                    )
                work = self.get_work(id)
                try:
                    work.convert(to=format, refresh=refresh)
                    results[id] = True
                except Exception as e:
                    results[id] = str(e)

        # Check if all conversions succeeded
        success = all([res == True for res in results.values()])
        return success, results

    def report_evaluation(self, evaluation: EvaluationResult):
        counts = evaluation.counts

        # Don't count missing syllables as a mistake
        evaluation["num_notes"] = len(evaluation)
        evaluation["num_syllables"] = (
            evaluation["num_notes"] - counts[EvalStatus.MISSING]
        )
        evaluation["accuracy"] = evaluation["correct"] / evaluation["num_syllables"]
        report = f"Accuracy best solmization: {evaluation['accuracy']:.0%}"
        errors = [
            f"{evaluation[k]} {k}"
            for k in [
                EvalStatus.INCORRECT,
                EvalStatus.MISSING,
                EvalStatus.INSERTION,
                EvalStatus.DELETION,
            ]
            if evaluation[k] > 0
        ]
        if len(errors) > 0:
            report += f' ({", ".join(errors)})'
        return report

    def evaluate(
        self,
        ids: t.Iterable[str] = None,
        write_output: bool = False,
        output_dir: str = None,
        refresh: bool = False,
        remove_musicxml: bool = True,
        write_log: bool = True,
        **evaluator_kws,
    ):
        """Evaluate a set of works and log the results.

        Parameters
        ----------
        ids : Iterable[str], optional
            An iterable of work identifiers to evaluate. If None, all works
            will be evaluated. Default is None.
        write_output : bool, optional
            If True, output files will be written to the specified directory.
            Default is False.
        output_dir : str, optional
            The directory where output files will be saved. If None, a default
            directory will be created based on the current timestamp. Default is None.
        refresh : bool, optional
            If True, existing output files will be overwritten. Default is False.
        remove_musicxml : bool, optional
            If True, the intermediate MusicXML files will be removed after
            generating the PDF. Default is True.
        write_log : bool, optional
            If True, a log of the evaluation process will be written to a YAML
            file. Default is True.
        **evaluator_kws : keyword arguments
            Additional keyword arguments to be passed to the evaluation function
            of each work.

        Returns
        -------
        tuple
            A tuple containing:
            - A DataFrame representation of the log.
            - A dictionary with detailed log information about the evaluation
              process.
        """
        start = datetime.now()
        if ids is None:
            ids = self.ids
        if write_output is False:
            write_log = False
        if write_output:
            if output_dir is None:
                output_dir = f"output-{start.strftime('%Y_%m_%d-%H_%M_%s')}"
            output_dir = os.path.join(self.dirs["output"], output_dir)
            log_fn = os.path.join(output_dir, "log.yaml")
            if os.path.exists(output_dir) and refresh:
                os.remove(log_fn)
            else:
                os.makedirs(output_dir, exist_ok=True)

        # A basic log
        log = {}
        log["time_start"] = start
        log["num_works"] = len(ids)
        log["works"] = {}
        log["errors"] = {}

        iterator = tqdm(ids) if write_output else ids
        for id in iterator:
            log["works"][id] = dict(status=None, errors=[])
            work = self.get_work(id=id)
            if write_output:
                xml_fn = os.path.join(output_dir, f"{id}.musicxml")
                pdf_fn = os.path.join(output_dir, f"{id}.pdf")
            if refresh or not write_output or not os.path.exists(pdf_fn):
                try:
                    evaluation, solmizer = work.evaluate(**evaluator_kws)
                    if write_output:
                        solmizer.stream.write("musicxml.pdf", pdf_fn)
                        if remove_musicxml:
                            os.remove(xml_fn)
                    log["works"][id]["status"] = "success"
                    log["works"][id]["evaluation"] = evaluation.export()
                except Exception as e:
                    log["works"][id]["status"] = "error"
                    log["works"][id]["errors"][id] = str(e)
            else:
                log["works"][id]["status"] = "skipped"

        # Finish up logging
        log["time_stop"] = datetime.now()
        log["num_errors"] = len(log["errors"])
        if write_log:
            with open(log_fn, "w") as file:
                yaml.dump(log, file)

        df = self._log_to_df(log)
        return df, log

    def _log_to_df(self, log) -> pd.DataFrame:
        """Convert an evaluation log into a pandas DataFrame.

        Parameters
        ----------
        log : dict
            A dictionary containing information about works and their evaluations.
            The expected structure is that 'log' has a key 'works', which is a
            dictionary where each value contains an 'evaluation' and 'status'.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row corresponds to a work and includes
            evaluation counts, the number of notes, and the status of the work.
            Additional columns for the percentage of each evaluation status
            are also included if applicable.
        """
        data = {}
        for id, work in log["works"].items():
            data[id] = work["evaluation"]["counts"]
            data[id]["num_notes"] = len(work["evaluation"]["predictions"])
            data[id]["status"] = work["status"]
        df = pd.DataFrame(data).T
        if "num_notes" in df:
            for name, const in EvalStatus.__members__.items():
                df[f"perc_{name.lower()}"] = df[name.lower()] / df["num_notes"]
        return df

    def load_evaluation(self, output_dir: str = None) -> pd.DataFrame:
        if output_dir is None:
            subdirs = sorted(os.listdir(self.dirs["output"]))
            output_dir = os.path.join(self.dirs["output"], subdirs[-1])

        log_fn = os.path.join(output_dir, "log.yaml")
        with open(log_fn, "r") as file:
            log = yaml.safe_load(file)

        return self._log_to_df(log)
