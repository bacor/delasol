# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t

# Libraries
import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def latexify_subscript(input: str, as_text: bool = True) -> str:
    """Convert a string to a LaTeX formatted subscript representation.

    Parameters
    ----------
    input
        The input string which may contain a subscript indicated by an
        underscore ('_'). If the underscore is present, the part before the
        underscore is treated as the main text, and the part after is treated
        as the subscript.
    as_text
        If True, the function returns a LaTeX formatted string with the text
        and subscript wrapped in a text environment. Default is True.

    Examples
    --------
    >>> latexify_subscript("C4")
    'C4'
    >>> latexify_subscript("C_4")
    '$\\\\text{C}_{\\\\text{4}}$'
    >>> latexify_subscript("C_4", as_text=False)
    '$C_{4}$'
    """
    if "_" in input:
        name, subscript = input.split("_")
        if as_text:
            return f"$\\text{{{name}}}_{{\\text{{{subscript}}}}}$"
        else:
            return f"${name}_{{{subscript}}}$"
    else:
        return input


def get_relative_lims(ticks: t.Iterable[float], margin: float) -> tuple[float, float]:
    """Get the relative limits for a given set of ticks with an added margin.

    Parameters
    ----------
    ticks
        An iterable containing the tick values from which to calculate the
        limits.
    margin
        A margin to be applied to the calculated limits, expressed as a
        fraction of the range between the minimum and maximum tick values.

    Returns
    -------
    tuple of float
        A tuple containing the lower and upper limits, adjusted by the
        specified margin.
    """
    min_val, max_val = min(ticks), max(ticks)
    delta = margin * (max_val - min_val)
    return min_val - delta, max_val + delta


def set_relative_lims(ax: Axes, margin: float = 0.2) -> None:
    """Set relative limits for the x and y axes of a given matplotlib Axes.

    Parameters
    ----------
    ax
        The matplotlib Axes object for which the limits will be set.
    margin
        The margin to apply to the limits, expressed as a fraction of the
        data aspect ratio for the x-axis. Default is 0.2.
    """
    aspect = ax.get_data_ratio()
    xlim = get_relative_lims(ax.get_xticks(), margin * aspect)
    ylim = get_relative_lims(ax.get_yticks(), margin)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def classify_edges(
    graph: nx.Graph,
) -> dict[str, t.Iterable]:
    loops, unidirectional, bidirectional = [], [], []
    for u, v in graph.edges:
        if u == v:
            loops.append((u, v))
        else:
            if (v, u) in graph.edges:
                bidirectional.append(tuple(sorted((u, v))))
            else:
                unidirectional.append((u, v))
    return dict(
        loops=loops,
        unidirectional=unidirectional,
        bidirectional=list(set(bidirectional)),
    )


def draw_edges(
    graph,
    edges,
    pos,
    weights=None,
    width=1,
    min_target_margin=10,
    arrowsize=7,
    node_size=500,
    connectionstyle="arc3,rad=-.2",
    color_mapper: t.Optional[t.Callable[[float], t.Any]] = lambda w: cm.Reds(
        0.9 * w + 0.1
    ),
    ax=None,
    arrows=True,
    **kws,
):
    if weights is None:
        weights = np.array([graph.edges[e]["weight"] for e in edges])
        weights = weights / weights.max()

    if arrows == True:
        kws["arrowsize"] = arrowsize
        kws["min_target_margin"] = min_target_margin
        kws["connectionstyle"] = connectionstyle

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=edges,
        edge_color=[color_mapper(w) for w in weights],
        ax=ax,
        arrows=arrows,
        width=width,
        node_size=node_size,
        **kws,
    )


def draw_graph(
    graph: nx.Graph,
    labels: t.Union[
        t.Literal["latex_name", "name", "syllable", "index"], str, dict[t.Any, str]
    ] = "latex_name",
    pos: dict[t.Any, (float, float)] = None,
    weights: t.Iterable[float] = None,
    show_loops: bool = False,
    ax: Axes = None,
    figsize: tuple[float, float] = (8, 4),
    pos_kws={},
    label_kws={},
    edge_kws={},
    **shared_kws,
) -> Axes:
    """Draw a graph using NetworkX and Matplotlib.

    Parameters
    ----------
    graph
        The graph to be drawn.

    labels
        The labels to use for the nodes:

        - str: The attribute of the nodes to use as labels. The attributes
          "name", "syllable", and "index" are predefined, but you can use
          custom attributes. A ValueError is raised if the the attribute
          specified does not exist.
        - dict: A dictionary mapping nodes to labels.

        Default is "name".

    pos
        A dictionary mapping nodes to their positions in the plot. If None,
        positions will be retrieved from the graph's node attributes.

    weights
        A collection of weights for the edges. If None, weights will be
        extracted from the graph's edges.

    show_loops
        If True, self-loops will be included in the drawing. Default is False.

    ax
        The axes to draw the graph on. If None, a new figure will be created.

    figsize
        The size of the figure in inches; ignored if `ax` is specified.
        Default is (8, 4).

    label_kws
        Additional keyword arguments for node label styling.

    edge_kws
        Additional keyword arguments for edge styling.

    color_mapper
        A function that maps edge weights to colors. Default is a function
        that scales weights to a red color gradient.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the graph was drawn.
    """
    # Create a new figure if no axis is passed
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Determine the node positions. By default try to use a node_positions method
    # if that exists, otherwise look for the positions attribute, and if that does
    # not exist either, raise an error.
    if pos is None:
        if hasattr(graph, "node_positions"):
            pos = graph.node_positions(**pos_kws)
        else:
            pos = nx.get_node_attributes(graph, "position")
            if len(pos) == 0:
                raise ValueError(
                    "The nodes in this graph have no 'position' attribute'"
                )
    if pos is None:
        raise ValueError("No positions found for the nodes")

    # Determine the node labels. By default, use the LaTeX to format subscripts
    # in the node names properly. Otherwise, look for node attributes with the
    # specified label name, and raise an error if those do not exist.
    if labels == "latex_name":
        names = nx.get_node_attributes(graph, "name")
        labels = {node: latexify_subscript(label) for node, label in names.items()}
    elif isinstance(labels, str):
        labels = nx.get_node_attributes(graph, labels)
        if len(labels) == 0:
            raise ValueError(f"No labels found for attribute {labels}")

    # Labels for missing nodes
    for node in graph.nodes:
        if node not in labels:
            labels[node] = str(node)

    # Draw the labels!
    _label_kws = dict(
        font_size=8,
        bbox=dict(
            facecolor="white",
            linewidth=0.5,
            boxstyle="round,pad=.3,rounding_size=.3",
        ),
    )
    _label_kws.update(**label_kws, **shared_kws)
    nx.draw_networkx_labels(graph, labels=labels, pos=pos, ax=ax, **_label_kws)

    # Edges
    _edge_kws = dict(ax=ax, pos=pos, weights=weights)
    _edge_kws.update(**edge_kws, **shared_kws)
    edges = classify_edges(graph)
    if show_loops and len(edges["loops"]) > 0:
        draw_edges(graph, edges["loops"], **_edge_kws)

    if len(edges["bidirectional"]) > 0:
        draw_edges(
            graph,
            edges["bidirectional"],
            connectionstyle="Arc3",
            arrows=False,
            **_edge_kws,
        )

    if len(edges["unidirectional"]) > 0:
        draw_edges(graph, edges["unidirectional"], **_edge_kws)

    return ax


def draw_hexachord_graph(
    graph: "HexachordGraph",
    styling: bool = True,
    width_factor: float = 0.7,
    margin: float = 0.1,
    **kws,
) -> None:
    """Draws a hexachord graph.

    Parameters
    ----------
    graph : HexachordGraph
        The hexachord graph to draw.
    styling : bool, optional
        If True, add styling (e.g. formatted axes). Default is True.
    width_factor : float, optional
        If no axis is passed, a new figure is made. It's width will be the
        number of unique pitches times the width factor. Default is 0.7.
    margin : float, optional
        The margin to add to the axes. Default is 0.1.
    **kws : keyword arguments
        Additional keyword arguments passed to the drawing function.

    Returns
    -------
    None
        This function does not return a value but draws a matplotlib figure.
    """
    ax = draw_graph(graph, figsize=(len(graph) * width_factor, 1), **kws)

    if styling:
        ax.axis("off")
        set_relative_lims(ax, margin=margin)


def draw_gamut_graph(
    graph,
    styling: bool = True,
    width_factor: float = 0.6,
    margin=0.3,
    pos_kws={},
    **kws,
) -> None:
    """Draw a the gamut graph.

    Parameters
    ----------
    styling : bool, optional
        If True, add styling (e.g. formatted axes). Default is True.
    ax : matplotlib.axes.Axes, optional
        The axes on which to draw the graph. If None, a new figure and axes
        will be created.
    width_factor : float, optional
        If no axis is passed, a new figure is made. It's width will be the
        number of unique pitches times the width factor. Default is 0.6.
    margin : float, optional
        The margin to add to the axes. Default is 0.3.
    pos_kws : dict, optional
        Additional keyword arguments for positioning. In particular, the
        `pos_x` and `pos_y` arguments can be set here. These are used
        to determine the scale of the axes and default to pos_x='diatonic'
        and pos_y='order'.
    **kws : keyword arguments
        Additional keyword arguments passed to the drawing function.

    Returns
    -------
    None
        This function does not return a value but draws a matplotlib figure.
    """
    if "figsize" not in kws:
        kws["figsize"] = (len(graph.pitches) * width_factor, len(graph.hexachords))

    ax = draw_graph(graph, pos_kws=pos_kws, **kws)

    # Decorate with nice axes
    if styling:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axis_on()
        ax.xaxis.grid(color=".9")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        # X-ticks
        match pos_kws.get("pos_x", "diatonic"):
            case "diatonic":
                ax.set_xticks([p.diatonicNoteNum for p in graph.pitches])
            case "ps":
                ax.set_xticks([p.ps for p in graph.pitches])
            case "order":
                ax.set_xticks(list(range(len(graph.pitches))))

        xtick_labels = [
            p.unicodeNameWithOctave if p.name in "CEG" else None for p in graph.pitches
        ]
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("pitch")

        # Y-ticks
        bases = graph.hexachords.keys()
        match pos_kws.get("pos_y", "order"):
            case "diatonic":
                ax.set_yticks([p.diatonicNoteNum for p in bases])
            case "ps":
                ax.set_yticks([p.ps for p in bases])
            case _:
                ax.set_yticks(list(range(len(bases))))

        ytick_labels = [base.unicodeNameWithOctave for base in bases]
        ax.set_yticklabels(ytick_labels)
        ax.set_ylabel("base of hexachord")

        set_relative_lims(ax, margin=margin)


def draw_parse_graph(
    graph,
    styling: bool = True,
    show_segments: bool = True,
    width_factor: float = 0.7,
    margin: float = 0.6,
    **kws,
):
    if "figsize" not in kws:
        kws["figsize"] = (width_factor * (len(graph) - 1), graph.width.max())

    ax = draw_graph(graph, **kws)

    if styling:
        if show_segments:
            for segment in graph.segments[1:]:
                ax.axvline(segment.start - 0.5, color="k", lw=0.5, linestyle="--")

        # Show axes and grid
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axis_on()
        ax.xaxis.grid(color=".9")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        # X-ticks
        ax.set_xticks(range(len(graph)))
        xlabels = ["start"] + [f"{i}" for i in range(1, len(graph) - 1)] + ["end"]
        for i, pos in enumerate(graph.input_positions):
            xlabels[pos] += f"\n{graph.sequence[i]}"
        ax.set_xticklabels(xlabels)

        # Y-ticks
        ax.set_yticks(np.arange(0, graph.width.max()))
        ax.set_yticklabels(np.arange(1, graph.width.max() + 1, dtype=int))
        plt.ylabel("width")

        set_relative_lims(ax, margin=margin)
        plt.tight_layout()


def draw_rollout_graph(
    graph,
    styling: bool = True,
    width_factor: float = 0.7,
    margin: float = 0.6,
    **kws,
):
    if "figsize" not in kws:
        kws["figsize"] = (width_factor * (len(graph) - 1), max(graph.width))

    ax = draw_graph(graph, **kws)

    if styling:
        # Show axes and grid
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axis_on()
        ax.xaxis.grid(color=".9")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        # X-ticks
        ax.set_xticks(range(len(graph)))
        xlabels = ["start"] + [f"{i}" for i in range(1, len(graph) - 1)] + ["end"]
        for i, coord in enumerate(graph.input_timesteps):
            xlabels[coord] += f"\n{graph.sequence[i]}"
        ax.set_xticklabels(xlabels)

        # Y-ticks
        ax.set_yticks(np.arange(0, max(graph.width)))
        ax.set_yticklabels(np.arange(1, max(graph.width) + 1, dtype=int))
        plt.ylabel("width")

        set_relative_lims(ax, margin=margin)
        plt.tight_layout()
