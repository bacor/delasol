# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Libraries
from music21.pitch import Pitch
from music21.stream import Stream
from music21.note import Note
from music21.spanner import Line

# Local imports
from delasol.custom_types import PitchLike


def as_pitch_list(pitch_string: str, sep: str = " ") -> list[Pitch]:
    """Convert a string of pitches into a list of Pitch objects.

    Parameters
    ----------
    pitch_string : str
        A string containing pitch values separated by a specified separator.
    sep : str, optional
        The separator used to split the pitch_string into individual pitches.
        Default is a space (" ").

    Returns
    -------
    list[Pitch]
        A list of Pitch objects created from the input string.

    Examples
    --------
    >>> as_pitch_list("C4 D4 E4 F4")
    [<music21.pitch.Pitch C4>, <music21.pitch.Pitch D4>, <music21.pitch.Pitch E4>, <music21.pitch.Pitch F4>]
    """
    return [Pitch(p) for p in pitch_string.split(sep)]


def as_stream(pitch_string: str, sep: str = " ") -> Stream:
    """Convert a string of pitch representations into a Stream of Note objects.

    Parameters
    ----------
    pitch_string : str
        A string containing pitch representations, separated by the specified
        separator.
    sep : str, optional
        The separator used to split the pitch_string into individual pitches.
        Default is a space (" ").

    Returns
    -------
    Stream
        A Stream object containing Note objects created from the pitches in the
        input string.

    Examples
    --------
    >>> stream = as_stream("C4 D4 E4 F4")
    >>> stream[0]
    <music21.note.Note C>
    >>> stream[1]
    <music21.note.Note D>
    """
    pitches = as_pitch_list(pitch_string, sep=sep)
    notes = [Note(p) for p in pitches]
    return Stream(notes)


def as_pitch(pitch: PitchLike) -> Pitch:
    """Convert a given pitch representation to a Pitch object.

    Parameters
    ----------
    pitch : PitchLike
        The input pitch representation, which can be a string or a Pitch
        object.

    Returns
    -------
    Pitch
        A Pitch object corresponding to the input representation.

    Raises
    ------
    ValueError
        If the input is neither a string nor a Pitch object.

    Examples
    --------
    >>> as_pitch("A4")
    <music21.pitch.Pitch A4>

    >>> as_pitch(Pitch("C#5"))
    <music21.pitch.Pitch C#5>
    >>> as_pitch(1)
    Traceback (most recent call last):
        ...
    ValueError: Expected a Pitch object or string, got <class 'int'>
    """
    if isinstance(pitch, str):
        pitch = Pitch(pitch)
    elif not isinstance(pitch, Pitch):
        raise ValueError(f"Expected a Pitch object or string, got {type(pitch)}")
    return pitch


def extract_lyrics(notes: t.Iterable[Note], number: int) -> list[str]:
    """Extract the lyrics at a given line number from an iterable of notes"""
    extracted = []
    for note in notes:
        lyrics = {lyric.number: lyric for lyric in note.lyrics}
        if number in lyrics:
            extracted.append(lyrics[number].text)
        else:
            extracted.append(None)
    return extracted


def num_lyrics(stream: Stream) -> int:
    """Returns the maximum number of lyrics in a stream"""
    num_lyrics = 0
    for note in stream.flat.notes:
        numbers = [lyric.number for lyric in note.lyrics]
        if len(numbers) > 0:
            num_lyrics = max(max(numbers), num_lyrics)
    return num_lyrics


def annotate_note(
    note: Note, text: str = None, color: str = None, number: int = 1
) -> None:
    """Add lyrics to a note and set its color"""
    if text is not None:
        note.addLyric(text, lyricNumber=number)
    lyrics = {lyric.number: lyric for lyric in note.lyrics}
    if color is not None:
        if number in lyrics:
            lyrics[number].style.color = color


def set_lyrics_color(
    notes: t.Iterable[Note], number: int, color: str = "#000000"
) -> None:
    """Set the color of a lyric in a list of notes."""
    for note in notes:
        annotate_note(note, color=color, number=number)


def overline_notes(notes):
    line = Line(notes)
    line.lineType = "dotted"
    return line
    # self.stream.insert(0, line)
