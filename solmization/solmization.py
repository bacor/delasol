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
from .parse_graph import ParseGraph
from .gamut_graph import GamutGraph, get_gamut
from .utils import set_lyrics_color, SUBSCRIPTS


# TODO should be replaced
def fmt_syllable(syllable, hex):
    return f"{syllable}{SUBSCRIPTS[hex]}"


def _extract_from_path_segments(segments, isOriginal):
    isOriginal = [False] + isOriginal + [False]
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
        **kwargs,
    ):
        if input:
            self.solmize(input, gamut, **kwargs)

    def _validate_input(
        self,
        input: SolmizationInput = None,
        gamut: GamutInput = None,
    ) -> tuple[list[Pitch], GamutGraph]:
        if isinstance(input, Stream):
            raise ValueError("Use StreamSolmation for stream inputs")

        # Case two: an iterable of pitches, notes, or strings
        elif isinstance(input, Iterable):
            if gamut is None:
                raise ValueError("Please provide a gamut")
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

        if isinstance(gamut, str):
            gamut = get_gamut(gamut)

        if not isinstance(gamut, GamutGraph):
            raise ValueError("No gamut was specified.")

        return pitches, gamut

    def solmize(self, input: SolmizationInput, gamut: GamutInput = None, **kwargs):
        # Gap-fill the melody and build a solmization graph (steps=gap filled, pitches=originals)
        self.pitches, self.gamut = self._validate_input(input, gamut)
        self.steps, self.is_original = self.gamut.fill_gaps(self.pitches)
        self.parse = ParseGraph(self.gamut, self.steps, **kwargs)
        self.step_segments = self.parse.path_segments()
        self.pitch_segments = _extract_from_path_segments(
            self.step_segments, self.is_original
        )

    def iter_segments(self):
        for segment in self.pitch_segments:
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

    def show_steps(self, **kwargs):
        """Show a stream of the the gap-filled melody."""
        notes = []
        for pitch, isOrig in zip(self.steps, self.is_original):
            note = Note(pitch)
            if not isOrig:
                note.notehead = "diamond"
                note.style.color = "#999999"
            note.stemDirection = "noStem"
            notes.append(note)
        stream = Stream(notes)
        return stream.show(**kwargs)

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
            lyric_offset = max([max([l.number for l in n.lyrics]) for n in notes]) + 1

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
    to_stream: bool = False,
    **kwargs,
) -> Solmization:
    """A convenience function that creates a Solmization object depending on the input type."""
    if to_stream:
        input = Stream(input)
    if isinstance(input, Stream):
        solmization = StreamSolmization(input, style=style, gamut=gamut, **kwargs)
    else:
        solmization = Solmization(input, gamut=gamut, **kwargs)
    return solmization
