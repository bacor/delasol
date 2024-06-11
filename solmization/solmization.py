# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
from typing import Union, Callable
from collections.abc import Iterable
from collections import Counter
from copy import deepcopy
from music21.spanner import Line
from music21.stream import Stream
from music21.pitch import Pitch
from music21.note import Note

# Local imports
from .parse_graph import GamutParseGraph
from .gamut_graph import GamutGraph, get_gamut
from .utils import set_lyrics_color, SUBSCRIPTS


# TODO should be replaced
def fmt_syllable(syllable, hex):
    return f"{syllable}{SUBSCRIPTS[hex]}"


def _extract_from_path_segments(segments, isOriginal):
    newSegments = []
    origPos = 0
    for segment in segments:
        isOrig = isOriginal[segment["start"] : segment["end"] + 1]
        paths = []
        weights = []
        for i, path in enumerate(segment["paths"]):
            origPath = []
            for j in range(len(path)):
                if isOrig[j]:
                    origPath.append(path[j])

            if len(origPath) > 0:
                paths.append(origPath)
                weights.append(segment["weights"][i])

        if len(paths) > 0:
            newSegments.append(
                dict(
                    start=origPos,
                    end=origPos + len(paths[0]) - 1,
                    paths=paths,
                    weights=weights,
                )
            )
            origPos += len(paths[0])

    return newSegments


def extract_lyrics(notes: Iterable[Note], lyric_num: int) -> list[str]:
    extracted = []
    for note in notes:
        lyrics = {lyric.number: lyric for lyric in note.lyrics}
        if lyric_num in lyrics:
            extracted.append(lyrics[lyric_num].text)
        else:
            extracted.append(None)
    return extracted


def evaluator(syllable: str, target: str) -> str:
    """
    Evaluate a note based on the target lyrics.
    """
    if target is not None:
        if target.startswith("["):
            target = target.replace("[", "").replace("]", "")
        elif target.startswith("("):
            target = target.replace("(", "").replace(")", "")
        if "*" in target:
            target = target.split("*")[1]

    if target == syllable:
        return "correct"
    elif target == "?":
        return "missing"
    elif target == "" and syllable != "":
        return "insertion"
    elif target != "" and syllable == "":
        return "deletion"
    else:
        return "incorrect"


SolmizationInput = Union[Iterable[Pitch], Iterable[Note], Iterable[str]]
GamutInput = Union[GamutGraph, str]


class Solmization:

    def __init__(
        self,
        input: SolmizationInput = None,
        gamut: GamutInput = None,
        gamut_kws: dict = {},
        **kwargs,
    ):
        if isinstance(gamut, str):
            gamut = get_gamut(gamut, **gamut_kws)
        if not isinstance(gamut, GamutGraph):
            raise ValueError("No gamut was specified.")
        self.gamut = gamut
        if input:
            self.solmize(input, **kwargs)

    def _validate_input(
        self,
        input: SolmizationInput = None,
    ) -> list[Pitch]:
        """Extract a sequence of pitches from the input."""
        if isinstance(input, Stream):
            raise ValueError("Use StreamSolmation for stream inputs")
        elif isinstance(input, Iterable):
            if isinstance(input[0], Pitch):
                pitches = [Pitch(p) for p in input]
            elif isinstance(input[0], Note):
                pitches = [Pitch(n.pitch) for n in input]
            elif isinstance(input[0], str):
                pitches = [Pitch(p) for p in input]
            else:
                raise ValueError(
                    "Unsupported input type: you can pass an iterable of pitches, notes or pitch strings"
                )
        return pitches

    def solmize(
        self,
        input: SolmizationInput,
        mismatch_penalty: float = 2,
        prune_parse: bool = True,
        parse_graph_kws: dict = {},
    ):
        self.pitches = self._validate_input(input)
        self.parse = GamutParseGraph(
            self.gamut,
            self.pitches,
            mismatch_penalty=mismatch_penalty,
            prune=prune_parse,
            **parse_graph_kws,
        )

        # TODO this is a bit hacky; it is essentially superflous
        is_original = [False] * len(self.parse)
        for pos in self.parse.input_positions:
            is_original[pos] = True

        full_segments = self.parse.path_segments()
        self.segments = _extract_from_path_segments(full_segments, is_original)

    def iter_segments(self):
        for segment in self.segments:
            output = dict(**segment)
            output["syllables"] = []
            output["raw_syllables"] = []
            for path in segment["paths"]:
                path_sylls = []
                path_raw_sylls = []
                for _, (hex, pitch) in path:
                    syll = self.gamut.solmize((hex, pitch))
                    path_raw_sylls.append(syll)
                    path_sylls.append(fmt_syllable(syll, hex))
                output["syllables"].append(path_sylls)
                output["raw_syllables"].append(path_raw_sylls)
            yield output

    def iter_solmizations(self):
        for segment in self.iter_segments():
            for i, pos in enumerate(range(segment["start"], segment["end"] + 1)):
                yield dict(
                    pos=pos,
                    nodes=[path[i] for path in segment["paths"]],
                    raw_syllables=[sylls[i] for sylls in segment["raw_syllables"]],
                    syllables=[sylls[i] for sylls in segment["syllables"]],
                    weights=segment["weights"],
                )

    def best_syllables(self, subscript=True):
        key = "syllables" if subscript else "raw_syllables"
        return [data[key][0] for data in self.iter_solmizations()]

    def evaluate(
        self,
        targets: Iterable[str],
        return_counts: bool = True,
        evaluator: Callable[[str, str], str] = evaluator,
    ):
        predictions = self.best_syllables(subscript=False)
        evaluation = [evaluator(pred, targ) for pred, targ in zip(predictions, targets)]
        if return_counts:
            return dict(Counter(evaluation))
        else:
            return evaluation

    def draw_parse(self, **kwargs):
        """Draw the parse graph"""
        return self.parse.draw(**kwargs)


class StreamSolmization(Solmization):
    def __init__(
        self,
        stream: Stream,
        style: str = "continental",
        gamut: GamutInput = None,
        in_place: bool = True,
        **kwargs,
    ):
        self.stream = stream if in_place else deepcopy(stream)
        self.stream.stripTies(inPlace=True)
        if self.stream.hasPartLikeStreams():
            if len(self.stream.parts) >= 2:
                print(
                    f"Warning: found {len(self.stream.parts)} parts, but only the first part of the stream will be used"
                )
                self.stream = self.stream.parts[0]
        self.style = style
        if gamut is None:
            key = self.stream.flat.keySignature
            gamut = get_gamut(style=self.style, key=key)
        super().__init__(self.stream.flat.notes, gamut=gamut, **kwargs)

    def evaluate(
        self, target_lyrics: int = None, targets: Iterable[str] = None, **kwargs
    ):
        if targets is None:
            if target_lyrics is None:
                raise ValueError(
                    "Please specify target_lyrics: the lyric number containing the target syllables"
                )
            targets = extract_lyrics(self.stream.flat.notes, target_lyrics)
        return super().evaluate(targets, **kwargs)

    def annotate_note(
        self,
        note: Note,
        text: str,
        syllable: str,
        number: int = 1,
        target: str = None,
        target_number: int = None,
    ) -> None:
        note.addLyric(text, lyricNumber=number)

        # Update color of text and reference syllable, if a target is given
        if target is not None or target_number is not None:
            lyrics = {lyric.number: lyric for lyric in note.lyrics}
            if target is None and target_number in lyrics:
                target = lyrics[target_number].text
                if target.startswith("["):
                    target = target.replace("[", "").replace("]", "")
                elif target.startswith("("):
                    target = target.replace("(", "").replace(")", "")

                if "*" in target:
                    target = target.split("*")[1]
                    lyrics[target_number].style.color = "red"
                if target == "?":
                    lyrics[target_number].style.color = "red"
                else:
                    lyrics[target_number].style.color = "black"

            # Color the syllable according to correctness
            if syllable != target or target is None:
                lyrics[number].style.color = "red"
            elif target == "?":
                lyrics[number].style.color = "blue"
            else:
                lyrics[number].style.color = "green"

    def annotate(
        self,
        targets: Iterable[str] = None,
        target_lyrics: int = None,
        lyric_offset: int = None,
        best_only: bool = False,
        num_paths: int = 6,
        show_more_paths: bool = True,
        show_weights: bool = True,
        show_segments: bool = True,
        show_all_segments: bool = True,
        grey_lyrics_num: int = None,
    ):
        """Annotate a stream with solmization syllables."""
        if best_only:
            num_paths = 1
            show_segments = False
            show_more_paths = False
            show_weights = False

        notes = self.stream.flat.notes

        if lyric_offset is None:
            try:
                lyric_offset = (
                    max([max([l.number for l in n.lyrics]) for n in notes]) + 1
                )
            except ValueError:
                lyric_offset = 0

        for segment in self.iter_segments():
            start, end = segment["start"], segment["end"]
            segment_notes = notes[start : end + 1]

            for rank in range(min(num_paths, len(segment["paths"]))):
                syllables = segment["syllables"][rank]
                raw_syllables = segment["raw_syllables"][rank]
                weight = segment["weights"][rank]

                for pos, note in enumerate(segment_notes):
                    text = syllables[pos]
                    syllable = raw_syllables[pos]
                    if show_weights and pos == 0:
                        text = f"[w={weight:.1f}] {text}"

                    opts = dict()
                    if targets is not None:
                        opts["target"] = targets[start + pos]
                    elif target_lyrics is not None:
                        opts["target_number"] = target_lyrics

                    self.annotate_note(
                        note, text, syllable, number=rank + lyric_offset, **opts
                    )

            if show_more_paths and len(segment["paths"]) > num_paths:
                segment_notes[0].addLyric(
                    f"({len(segment['paths']) - num_paths} more paths)"
                )

            if (
                show_segments
                and len(segment_notes) > 1
                and (show_all_segments or len(segment["paths"]) > 1)
            ):
                line = Line(segment_notes)
                line.lineType = "dotted"
                self.stream.insert(0, line)

        if grey_lyrics_num is not None:
            set_lyrics_color(notes, grey_lyrics_num, "#999999")

        return self.stream


def solmize(
    input,
    style: str = None,
    gamut: GamutInput = None,
    prune_parse: bool = True,
    mismatch_penalty: float = None,
    mutation_weight: float = None,
    loop_weight: float = None,
    fa_super_la: bool = None,
    fa_super_la_weight: float = None,
    step_weight: float = None,
    hexachord_weights=None,
) -> Solmization:
    """A convenience function that creates a Solmization object depending on the input type."""
    opts = {}

    gamut_kws = {}
    if mutation_weight is not None:
        gamut_kws["mutation_weight"] = mutation_weight
    opts["gamut_kws"] = gamut_kws

    hex_kws = {}
    if loop_weight is not None:
        hex_kws["loop_weight"] = loop_weight
    if fa_super_la is not None:
        hex_kws["fa_super_la"] = fa_super_la
    if fa_super_la_weight is not None:
        hex_kws["fa_super_la_weight"] = fa_super_la_weight
    if step_weight is not None:
        hex_kws["step_weight"] = step_weight
    if hexachord_weights is not None:
        hex_kws["weights"] = hexachord_weights
    opts["gamut_kws"]["hexachord_kws"] = hex_kws

    if mismatch_penalty is not None:
        opts["mismatch_penalty"] = mismatch_penalty
    if prune_parse is not None:
        opts["prune_parse"] = prune_parse

    if isinstance(input, Stream):
        solmization = StreamSolmization(input, style=style, gamut=gamut, **opts)
    else:
        solmization = Solmization(input, gamut=gamut, **opts)
    return solmization
