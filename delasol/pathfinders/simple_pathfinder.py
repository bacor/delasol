# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import typing as t
from abc import ABC, abstractmethod

# Library imports
import networkx as nx

# Local imports
from delasol.custom_types import BaseGraphPath, RolloutGraphPath
from delasol.graphs.rollout_graph import RolloutGraph
from delasol.custom_types import RolloutGraphNode
from delasol.pathfinders.pathfinder import Pathfinder


class SimplePathfinder(Pathfinder):
    """
    A pathfinder that finds the shortest path in a rollout graph
    simply using nx.shortest_simple_paths.

    Examples
    --------
    Here we use a simple circulant graph with 6 nodes and a single edge
    between neighbouring nodes. Nodes are numberd 0–5 and the edge weight
    from u to v is the maximum of u and v. So a path [0, 1, 2, 3, 2, 1, 0]
    from 0 to 3 and back would have weight 1+2+3+3+2+1 = 12. A path going
    around the circle like [0, 5, 4, 3, 2, 1, 0] or [0, 1, 2, 3, 4, 5, 0]
    would have weight 5+5+4+3+2+1=20, and the most costly path would of
    course be [0, 5, 4, 3, 4, 5, 0] with weight 5+5+4+4+5+5=28.

    >>> import networkx as nx
    >>> graph = nx.circulant_graph(6, [1])
    >>> for u, v in graph.edges: graph[u][v]["weight"] = max(u, v)
    >>> rollout = RolloutGraph(graph, [0, 3, 0])
    >>> pathfinder = Pathfinder(rollout)
    >>> pathfinder.get_path(0)
    [(0, 'START'), (1, 0), (2, 1), (3, 2), (4, 3), (5, 2), (6, 1), (7, 0), (8, 'END')]
    >>> pathfinder.get_base_path(0)
    [0, 1, 2, 3, 2, 1, 0]
    >>> pathfinder.get_base_path(0, inputs_only=True)
    [0, 3, 0]
    >>> pathfinder.get_props(0)
    {'weight': 12}
    >>> pathfinder.get_base_path(1)
    [0, 5, 4, 3, 2, 1, 0]
    >>> pathfinder.get_props(1)
    {'weight': 20}
    >>> pathfinder.get_base_path(2)
    [0, 1, 2, 3, 4, 5, 0]
    >>> pathfinder.get_props(2)
    {'weight': 20}
    >>> pathfinder.get_base_path(3)
    [0, 5, 4, 3, 4, 5, 0]
    >>> pathfinder.get_props(3)
    {'weight': 28}
    """

    def find_paths(self, rollout: RolloutGraph):
        """Find all shortest simple paths in a given rollout graph.

        Parameters
        ----------
        rollout : RolloutGraph
            The graph in which to find the shortest simple paths, containing a
            start and end node.

        Yields
        ------
        tuple
            A tuple containing each shortest path and an empty dictionary.
        """
        paths = nx.shortest_simple_paths(
            rollout, source=rollout.start, target=rollout.end, weight="weight"
        )
        for path in paths:
            yield path, dict()
