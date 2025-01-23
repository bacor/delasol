# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from abc import ABC, abstractmethod

# Local imports
from delasol.custom_types import GamutGraphPath, GamutGraphNode
from delasol.graphs.gamut_graph import GamutGraph
from delasol.constants import SYLLABLES, UNICODE_SUBSCRIPTS


###################### Abstract base class ######################


class Formatter(ABC):
    """
    Abstract formatter class used to transform solmization paths into
    for example syllable sequences.

    Usage
    -----

    The formatters are meant to be quite flexible. You can use them to format
    sequences of nodes in the gamut graph: solmization paths. Nodes in the gamut
    graph are (base, pitch) pairs: the base identifies the hexachord, the pitch
    the note within the hexachord. You can pass list of nodes directly:

    >>> from music21.pitch import Pitch
    >>> gamut = GamutGraph(["G2", "C3"])
    >>> fmt = get_formatter('syllable', gamut)
    >>> path = [(Pitch("G2"), Pitch("G2")), (Pitch("C3"), Pitch("D3"))]
    >>> fmt.format(path)
    ['ut', 're']

    Or you can format a single node:

    >>> fmt.format(path[0])
    'ut'

    Besides, all nodes have globally unique names of the form re_C3, which in
    this case identifies the re in the hexachord on C3, that is the D3. You can
    also format a list of names:

    >>> names = "ut_G2 re_G2 mi_G2 fa_G2 sol_G2 re_C3 mi_C3"
    >>> fmt.format_names(names)
    ['ut', 're', 'mi', 'fa', 'sol', 're', 'mi']
    >>> fmt.format_names("re_G2")
    're'

    This form will be more convenient for the examples here. Note that the
    format_names method automatically automatically splits the input string
    if it encounters any spaces.

    The `.format` method takes some additional arguments:

    >>> fmt.format_names("ut_G2 mi_G2 re_C3 fa_C3", join=True, sep="-")
    'ut-mi-re-fa'

    Implementation new formatters
    -----------------------------

    To implement a new formatter, you need to subclass the `Formatter` class and
    implement the `format_path` method. The `format_path` method should take a
    list of nodes (base, pitch) and return a list of formatted strings. The
    `format` method will automatically call the format_path method and take
    care of the rest. The class does need a `name` attribute. That's it:

    .. code-block:: python

        class MyFormatFormatter(Formatter):
            name = "my_format"

            def format_path(self, path: GamutGraphPath) -> list[str]:
                sylls = ["do", "re", "mi", "fa", "sol", "la", "ti"]
                return [sylls[self.gamut.nodes[node]["index"]] for node in path]

        register_formatter(MyFormatFormatter)
    """

    name: str
    """Name of the formatter that can be used to get the formatter using :py:func:`get_formatter`"""

    def __init__(self, gamut: GamutGraph, **kws):
        self.gamut = gamut

        # Store the keywords passed on initialization so they can be used
        # later when formatting as the defaults
        self.__init_kws = kws

    def format_names(self, names: str | list[str], auto_split=True, **kws) -> list[str]:
        """Format solmization paths using node names rather than nodes themselves.
        See :class:`Formatter` for examples. Returns Aa list of formatted names.

        Parameters
        ----------
        names
            A single name as a string or a list of names to be formatted. If a
            string is provided and `auto_split` is True, the string will be split
            into individual names based on spaces.
        auto_split
            A flag indicating whether to automatically split a string of names into
            a list. Default is True.
        **kws
            Additional keyword arguments to be passed to the formatting function.
        """
        if isinstance(names, str) and " " not in names:
            node = self.gamut.get_node(name=names)
            return self.format(node, **kws)

        if isinstance(names, str) and auto_split and " " in names:
            names = names.split(" ")
        path = [self.gamut.get_node(name=name) for name in names]
        return self.format(path, **kws)

    def format(
        self, input: GamutGraphPath | GamutGraphNode, join=False, sep=" ", **kws
    ) -> list[str] | str:
        """Format the input into a string or list of strings based on the provided
        parameters. See :class:`Formatter` for examples.

        Parameters
        ----------
        input
            The input object to be formatted.
        join
            If True, the output will be a single string. If False, a list of
            strings will be returned. Default is False.
        sep
            The separator to use when joining the strings if `join` is True.
            Default is a single space.
        **kws
            Additional keyword arguments to be passed to `format_path`

        Returns
        -------
        list of str or str
            The formatted output as a list of strings or a single string
            depending on the value of `join`.
        """
        kwargs = dict(**self.__init_kws)
        kwargs.update(**kws)
        if input in self.gamut:
            return self.format_path([input], **kwargs)[0]
        elif isinstance(input, list):
            output = self.format_path(input, **kwargs)
            if join:
                return sep.join(output)
            else:
                return output
        else:
            raise ValueError("Input must be a list of nodes or a single node.")

    @abstractmethod
    def format_path(self, path: GamutGraphPath, **kws) -> list[str]:
        """Format a solmization path into a list of strings. Should be implemented
        in the subclass and return a list of strings representing the formatted path.

        Parameters
        ----------
        path
            The path to be formatted.
        kws
            Additional keyword arguments to be passed to the formatting function.
        """
        raise NotImplementedError


########################## Registry ##########################


FORMATTERS = {}
"""Dict[str, Formatter]: A dictionary of registered formatters."""


def register_formatter(formatter: Formatter):
    """Register a new formatter. See the `Formatter` class for an example
    of how to implement a new formatter.

    Parameters
    ----------
    formatter
        The formatter class to be registered.

    Raises
    ------
    ValueError
        If the formatter does not have a 'name' attribute.
    """
    if not issubclass(formatter, Formatter):
        raise ValueError("A formatter must be an subclas of the Formatter class")
    if not hasattr(formatter, "name"):
        raise ValueError("Formatter must have a 'name' attribute.")

    FORMATTERS[formatter.name] = formatter


def get_formatter(name: str, gamut: GamutGraph, **kws) -> Formatter:
    """Get a formatter by name.

    Parameters
    ----------
    name
        The name of the formatter.
    gamut
        The gamut graph.
    **kws
        Additional keyword arguments to be passed to the formatter

    Raises
    ------
    ValueError
        If the formatter is not found.
    """
    formatter_class = FORMATTERS.get(name)
    if not formatter_class:
        raise ValueError(f"Formatter '{name}' not found.")
    return formatter_class(gamut, **kws)


########################## Formatters ##########################


class CustomLabelsFormatter(Formatter):
    """
    Format nodes using custom labels for each index in the hexachord.

    Examples
    --------
    >>> gamut = GamutGraph(["G2", "C3"])
    >>> fmt = get_formatter('custom_labels', gamut)
    >>> labels = ["do", "re", "mi", "fa", "sol", "la", "ti"]
    >>> fmt.format_names("ut_G2 mi_G2 re_C3", labels=labels)
    ['do', 'mi', 're']
    """

    name = "custom_labels"

    default_labels: list[str] | None = None
    """list of str: Default labels to use if no labels are provided."""

    def format_path(self, path: GamutGraphPath, labels=None) -> list[str]:
        if labels is None and self.default_labels is not None:
            labels = self.default_labels

        if labels is None:
            raise ValueError("Labels must be provided.")

        if not len(labels) == 7:
            raise ValueError(
                "You should pass exactly 7 names for the different index in a hexachord"
            )
        return [labels[self.gamut.nodes[node]["index"]] for node in path]


register_formatter(CustomLabelsFormatter)


class SyllableFormatter(CustomLabelsFormatter):
    """
    Formatter that formats a path as a list of syllables.

    Examples
    --------
    >>> gamut = GamutGraph(["G2", "C3"])
    >>> fmt = get_formatter('syllable', gamut)
    >>> fmt.format_names("ut_G2 mi_G2 re_C3")
    ['ut', 'mi', 're']
    """

    name = "syllable"
    default_labels = SYLLABLES


register_formatter(SyllableFormatter)


class ModernSyllableFormatter(CustomLabelsFormatter):
    """
    Formatter that formats a path as a list of modern syllables.

    Examples
    --------
    >>> gamut = GamutGraph(["G2", "C3"])
    >>> fmt = get_formatter('modern_syllable', gamut)
    >>> fmt.format_names("ut_G2 mi_G2 re_C3")
    ['do', 'mi', 're']
    """

    name = "modern_syllable"
    default_labels = ["do", "re", "mi", "fa", "sol", "la", "ti"]


register_formatter(ModernSyllableFormatter)


class NameFormatter(Formatter):
    """
    Formatter a node using their unique name: re_C3, mi_G2, etc.

    Parameters
    ---------
    unicode : bool, optional
        If True, the path will be formatted using Unicode characters.
        Default is False.


    Examples
    --------
    >>> gamut = GamutGraph(["G2", "C3", "B-2"])
    >>> fmt = get_formatter('name', gamut)
    >>> fmt.format_names("ut_G2 mi_G2 re_C3")
    ['ut_G2', 'mi_G2', 're_C3']
    >>> fmt.format_names("re_C3 fi_B-2", unicode=True)
    ['re_C3', 'fi_B♭2']
    """

    name = "name"

    def format_path(self, path: GamutGraphPath, unicode=False) -> list[str]:
        output = []
        for node in path:
            syllable = self.gamut.nodes[node]["syllable"]
            if unicode:
                base_str = node[0].unicodeNameWithOctave
            else:
                base_str = node[0].nameWithOctave
            output.append(f"{syllable}_{base_str}")
        return output


register_formatter(NameFormatter)


class HexachordNumberFormatter(Formatter):
    """
    Format nodes as a syllable followed by a hexachord number: ut1, re2, etc.

    Parameters
    ---------
    unicode : bool, optional
        If True, format the path using Unicode characters. Default is True.
    subscript : bool, optional
        If True, include subscripts in the formatted path. Default is True.

    Examples
    --------

    >>> gamut = GamutGraph(["G2", "C3", "B-2"])
    >>> fmt = get_formatter('syllable_hexnum', gamut)
    >>> fmt.format_names("ut_G2 mi_G2 re_C3 fa_C3")
    ['ut₁', 'mi₁', 're₂', 'fa₂']

    For hexachords without a number, the base name is used:

    >>> fmt.format_names("ut_B-2")
    'ut_B♭2'
    >>> fmt.format_names("ut_B-2", unicode=False)
    'ut_B-2'
    """

    name = "syllable_hexnum"

    def format_path(
        self, path: GamutGraphPath, unicode=True, subscript=True
    ) -> list[str]:
        output = []
        for base, pitch in path:
            hex = self.gamut.hexachords[base]
            number = hex.number
            if number is None:
                if unicode:
                    number = f"_{base.unicodeNameWithOctave}"
                else:
                    number = f"_{base.nameWithOctave}"
            elif unicode and subscript:
                number = UNICODE_SUBSCRIPTS[number]
            index = self.gamut.nodes[(base, pitch)]["index"]
            syllable = SYLLABLES[index]
            output.append(f"{syllable}{number}")
        return output


register_formatter(HexachordNumberFormatter)


class DavantesFormatter(Formatter):
    """
    Formats the solmization path using Davantes' numerical notation.

    The notation is rather peculiar: Stainer (1900) even speaks of the
    "utter absurdity of Davantes' system"). Basically, possible note
    positions on 5 staff lines are numbered. So, assuming a G clef,
    1 stands for a D4, 2 for an E, 3 for an F, and so on. But importantly,
    if the clef changes, the numbering does not. So with a bass clef,
    the 1 now stands for a low F2. Numbering refers to the place on the
    staff line, irrespective of clef.

    Next, he adds a mark to the number that indicates both the length of
    the note and the quality of the hexachord it is in, according to the
    following scheme:

    .. list-table::
        :header-rows: 1

        * - Quality
          - Half note
          - Whole note
        * - Natural hexachord
          - (no mark)
          - :py:`"'"` after number
        * - Soft hexachord
          - :py:`"."` before number
          - :py:`"!"` before number
        * - Hard hexachord
          - :py:`"."` after number
          - :py:`"!"` after number

    So, assuming a alto clef, :py:`"6!"` denotes a half note C
    in a soft hexachord, and :py:`".7"` indicates the next D, etc.
    This is probably not a very useful notation, and indeed it
    attracted little if any following. However, it may be interesting
    that his notation explicitly encodes the hexachord, which is
    one of the reasons for implementing it.

    Finally, note that this formatter requires not just pitches but
    actual notes do determine their duration, and also the clef.

    Examples
    --------

    >>> import music21
    >>> from delasol.solmizers import solmize
    >>> s = music21.converter.parse("tinynotation: 4/2 c1 d c A c B-2 A B-1 G F")
    >>> s.insert(0, music21.key.KeySignature(-1))
    >>> clef = music21.clef.AltoClef()
    >>> s.insert(0, clef)
    >>> solmizer = solmize(s, "continental_16c")
    >>> solmizer.solmize(format="davantes", notes=solmizer.input, clef=clef)
    ['!6', '!7', '!6', '!4', '!6', '.5', '.4', '!5', '!3', '!2']

    Parameters
    ----------
    self : object
        The instance of the class that calls this method.
    hexachord : HexachordGraph
        The hexachord graph to be used for symbol generation.
    pitch : music21.pitch.Pitch
        The pitch object representing the musical pitch.
    note : music21.note.Note
        The note object representing the musical note.
    clef : music21.clef.Clef
        The clef object indicating the musical clef.
    quarterLength : int, optional
        The duration of the note in quarter lengths. If None, defaults to
        a standard value.

    Returns
    -------
    str
        The generated Davantes symbol as a string.

    """

    name = "davantes"

    def mark(self, quality: str, quarterLength: int) -> str:
        davantes_marks = {
            "soft": {2: ".", 4: "!"},
            "hard": {2: ".", 4: "!"},
            "natural": {2: "", 4: "'"},
        }
        return davantes_marks[quality][min(quarterLength, 4)]

    def davantes_symbol(
        self,
        hexachord: "HexachordGraph",
        pitch: "music21.pitch.Pitch",
        note: "music21.note.Note",
        clef: "music21.clef.Clef",
        quarterLength: int = None,
    ) -> str:
        if quarterLength is None and note is None:
            raise ValueError("Either note or quarterLength must be provided.")
        elif quarterLength is None:
            quarterLength = note.duration.quarterLength

        # Determine the number
        number = pitch.diatonicNoteNum - clef.lowestLine + 2

        # Determine the mark
        mark = self.mark(hexachord.quality, quarterLength)

        if hexachord.quality == "soft":
            symbol = f"{mark}{number}"
        else:
            symbol = f"{number}{mark}"
        return symbol

    def format_path(
        self,
        path: GamutGraphPath,
        notes: t.Iterable["music21.note.Note"] = None,
        clef: "music21.clef.Clef" = None,
    ) -> list[str]:
        if notes is None:
            raise ValueError("Notes must be provided.")
        if clef is None:
            raise ValueError("Clef must be provided.")

        output = []
        for note, (base, pitch) in zip(notes, path):
            kws = dict(
                note=note,
                pitch=pitch,
                hexachord=self.gamut.hexachords[base],
                quarterLength=note.duration.quarterLength,
                clef=clef,
            )
            symbol = self.davantes_symbol(**kws)
            output.append(symbol)
        return output


register_formatter(DavantesFormatter)


class MutationFormatter(Formatter):
    """
    Format a solmization path by only solmizing points of mutation,
    in a style similar to Tinctoris.

    At a point of mutation, the formatter will show both the syllable in the
    previous and the new hexachord: 'sol/re' will for example indicate that
    this pitch was 'sol' in the previous hexachord and 're' in the new one.

    Parameters
    ----------
    path : GamutGraphPath
        The path to be formatted, represented as a sequence of nodes.
    format : str, optional
        The formatter to use for labelling the nodes. Default is 'syllable'.
    mutation : str, optional
        The string used to indicate a mutation between nodes. Default is '/'.
    missing : str, optional
        The string to use when a node is missing. Default is '?'.
    formatter_kws : dict, optional
        Additional keyword arguments to pass to the formatter.

    Returns
    -------
    list[str]
        A list of formatted strings representing the path, with mutations
        indicated as specified.

    Examples
    --------

    >>> gamut = GamutGraph(["G2", "C3"])
    >>> fmt = get_formatter('mutation', gamut)
    >>> fmt.format_names("ut_G2 mi_G2 fa_G2 sol_G2 re_C3 mi_C3")
    ['ut', None, None, None, 'sol/re', None]

    You can tweak the mutation string and the format of the nodes:

    >>> fmt.format_names("ut_G2 mi_G2 fa_G2 sol_G2 re_C3 mi_C3", mutation=" -> ", format="syllable_hexnum", formatter_kws=dict(subscript=False))
    ['ut1', None, None, None, 'sol1 -> re2', None]

    Note that when mutating in a large jump, it may not be possible to solmize
    the pitch in the previous hexachord. In that case, the formatter will use
    a special `missing='?'` token to indicate the missing note:

    >>> fmt.format_names("fa_C3 mi_C3 re_G2 mi_G2")
    ['fa', None, '?/re', None]

    """

    name = "mutation"

    def format_path(
        self,
        path: GamutGraphPath,
        format: str = "syllable",
        mutation: str = "/",
        missing: str = "?",
        formatter_kws: dict = {},
    ) -> list[str]:
        # 'Subformatter' used to output node names
        formatter = get_formatter(format, self.gamut)
        fmt = lambda node: formatter.format([node], **formatter_kws)[0]

        # Go through the path, and only output labels whenever there is a
        # mutation between hexachords
        start = fmt(path[0])
        output = [start]
        for prev, cur in zip(path[:-1], path[1:]):
            if prev[0] != cur[0]:

                # If the same pitch can also be found in the previous hexachord,
                # format that note. Otherwise, use ?
                if (prev[0], cur[1]) in self.gamut:
                    old = fmt((prev[0], cur[1]))
                else:
                    old = missing
                new = fmt(cur)
                output.append(f"{old}{mutation}{new}")
            else:
                output.append(None)

        return output


register_formatter(MutationFormatter)
