from typing import Union
from collections.abc import Iterable
import music21
from music21.pitch import Pitch
from music21.note import Note
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


SolmizationInput = Union[
    music21.stream.Score,
    music21.stream.Stream,
    Iterable[Pitch],
    Iterable[Note],
]
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
        # Case one: an input stream, such as a score.
        if isinstance(input, music21.stream.Score):
            notes = input.parts[0].flat.notes
            pitches = [Pitch(n.pitch) for n in notes]
            if gamut is None:
                key = input.parts[0].measure(1).keySignature
                if key == music21.key.KeySignature(0):
                    gamut = "hard"
                elif key == music21.key.KeySignature(-1):
                    gamut = "soft"
                else:
                    raise Exception(f"Key signature {key} not supported")

        # Case two: an iterable of pitches, notes, or strings
        elif isinstance(input, Iterable):
            if gamut is None:
                raise ValueError("Please provide a gamut")
            if isinstance(input, music21.stream.Stream):
                pitches = [Pitch(n.pitch) for n in input.flat.notes]
            elif isinstance(input[0], music21.pitch.Pitch):
                pitches = [Pitch(p) for p in input]
            elif isinstance(input[0], music21.note.Note):
                pitches = [Pitch(n.pitch) for n in input]
            elif isinstance(input[0], str):
                pitches = [music21.pitch.Pitch(p) for p in input]
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
        self.pitches, self.gamut = self._validate_input(input, gamut)
        self.input = input

        # Gap-fill the melody and build a solmization graph (steps=gap filled, pitches=originals)
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
                if "*" in target:
                    target = target.split("*")[1]
                    lyrics[target_number].style.color = "red"
                elif target == "?":
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
        stream: music21.stream.Stream,
        num_paths: int = 6,
        targets: Iterable[str] = None,
        target_lyrics: int = None,
        lyric_offset: int = 0,
        show_weights: bool = True,
        show_all_segments: bool = True,
        grey_lyrics_num: int = None,
    ):
        notes = stream.flat.notes
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

            if len(segment["paths"]) > num_paths:
                segment_notes[0].addLyric(
                    f"({len(segment['paths']) - num_paths} more paths)"
                )

            if len(segment_notes) > 1 and (
                show_all_segments or len(segment["paths"]) > 1
            ):
                line = music21.spanner.Line(segment_notes)
                line.lineType = "dotted"
                stream.insert(0, line)

        if grey_lyrics_num is not None:
            set_lyrics_color(notes, grey_lyrics_num, "#999999")

        return stream

    def show_steps(self, **kwargs):
        notes = []
        for pitch, isOrig in zip(self.steps, self.is_original):
            note = music21.note.Note(pitch)
            if not isOrig:
                note.notehead = "diamond"
                note.style.color = "#999999"
            note.stemDirection = "noStem"
            notes.append(note)
        stream = music21.stream.Stream(notes)
        return stream.show(**kwargs)

    def draw(self, **kwargs):
        return self.parse.draw(**kwargs)
