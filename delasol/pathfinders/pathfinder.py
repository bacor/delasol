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


class Pathfinder(ABC):
    """
    Abstract base class for pathfinding algorithms.

    A pathfinder should search for the best paths in a rollout graph.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph from which paths will be generated.
    max_paths : int, optional
        The maximum number of paths to generate (default is 1000).
    **kws : keyword arguments
        Additional keyword arguments to be passed to the path finding method.

    Attributes
    ----------
    graph : nx.DiGraph
        The directed graph used for path generation.
    paths : list
        A list of paths that have been generated so far. Paths are assumed to be
        ranked so that the best path is the first item in this list.
    props : list
        A list of property dictionaries for each of the generated paths.
    max_paths : int
        The maximum number of paths that can be generated.

    Raises
    ------
    Exception
        If no paths from the start to end were found in the graph.
    """

    def __init__(self, graph: nx.DiGraph, max_paths=1000, **kws):
        # TODO use graph or rollout?
        self.graph = graph
        self.paths, self.props = [], []
        self.__generator_complete = False
        self.max_paths = max_paths
        self.__generator = self.find_paths(graph, **kws)

        # Generate the first path to check whether any paths exist
        if self.get_path(0) is None:
            raise Exception("No paths froms start to end were found in the graph.")

    @abstractmethod
    def find_paths(
        self, rollout: RolloutGraph, **kws
    ) -> t.Iterator[tuple[RolloutGraphPath, dict]]:
        """Find ranked paths in the rollout and their properties.

        The paths are assumed to be ranked in some way, so that the first
        path is the optimal path, the second path is the second-best path, etc.
        The function should return an iterator of `(path, props)` tuples.
        The `path` should be a list of nodes in the rollout graph of the same
        length as the rollout graph, and the `props` should be a dictionary
        with additional properties of the path. Note that `props` is required,
        although it can of course be empty.

        Parameters
        ----------
        rollout : RolloutGraph
            The rollout graph for which paths are to be generated.
        **kws : keyword arguments
            Additional parameters to customize path generation.

        Returns
        -------
        [(path, props)] : Iterator[RolloutGraphPath]
            An iterator that yields paths from the specified rollout graph.

        Raises
        ------
        NotImplementedError
            This method must be implemented in a subclass.
        """
        raise NotImplementedError

    def validate_path(self, path: RolloutGraphPath) -> None:
        """Validate a path. By default only checks whether the length of the
        path matches the length of the rollout graph.

        Parameters
        ----------
        path : RolloutGraphPath
            The path to be validated.

        Raises
        ------
        Exception
            If the path is invalid.
        """
        if not len(path) == len(self.graph):
            raise Exception(
                f"Invalid path length. Path has length {len(path)}, but should match the length of the rollout graph ({len(self.graph)})"
            )

    def validate_props(self, props: dict) -> None:
        """Validate the path properties. By default only checks that props is
        a dictionary.

        Parameters
        ----------
        props : dict
            The properties to validate.

        Raises
        ------
        Exception
            If the props is invalid.
        """
        if not isinstance(props, dict):
            raise Exception("Properties should be a dictionary")

    def preprocess_path(self, path: RolloutGraphPath, props: dict) -> t.Iterable:
        """Hook for preprocessing paths.

        Note that the preprocessed path should not change the size of the path
        (which should be the same as the length of the rollout graph).

        Parameters
        ----------
        path : RolloutGraphPath
            The path to be preprocessed
        props : dict
            The properties of the path

        Returns
        -------
        Iterable
            A preprocessed version of the input path.
        """
        return path

    def preprocess_props(self, props: dict, path: RolloutGraphPath) -> dict:
        """Hook for preprocessing path properties. By default adds
        the total path weight.

        Parameters
        ----------
        props : dict
            The properties to be preprocessed
        path : RolloutGraphPath
            The path to which the properties belong

        Returns
        -------
        dict
            A preprocessed version of the input properties.
        """
        if not "weight" in props:
            props["weight"] = nx.path_weight(self.graph, path, "weight")

        return props

    def get_path_and_props(
        self, rank: int
    ) -> tuple[RolloutGraphPath | None, dict | None]:
        """Return the path and its properties for a given rank.

        Parameters
        ----------
        rank : int
            The rank of the desired path. Must be a non-negative integer.

        Returns
        -------
        (path, props) : tuple[RolloutGraphPath | None, dict | None]
            A tuple containing the path corresponding to the given rank and its
            associated properties. If the rank exceeds the number of available paths
            or if the generator has completed, returns (None, None).
        """
        if not self.__generator_complete and rank >= len(self.paths):
            for i in range(len(self.paths), rank + 1):
                if i >= self.max_paths:
                    self.__generator_complete = True
                    return None, None

                try:
                    raw_path, raw_props = next(self.__generator)
                    path = self.preprocess_path(raw_path, raw_props)
                    self.validate_path(path)
                    self.paths.append(path)

                    props = self.preprocess_props(raw_props, path)
                    self.validate_props(props)
                    self.props.append(props)
                except StopIteration:
                    self.__generator_complete = True
                    return None, None

        if rank < len(self.paths):
            return self.paths[rank], self.props[rank]
        else:
            return None, None

    def get_path(
        self, rank: int, inputs_only: bool = False
    ) -> list[RolloutGraphNode] | None:
        """Return the path with a given rank.

        Parameters
        ----------
        rank : int
            The rank for which the path is to be retrieved.
        inputs_only : bool, optional
            If True, only the nodes corresponding to inputs are returned

        Returns
        -------
        list[RolloutGraphNode] | None
            A path in the rollout graph if it exists, or None otherwise.
            If inputs_only is True, only nodes corresponding to inputs,
            otherwise the length of the path corresponds to the length
            of the rollout graph.
        """
        path, _ = self.get_path_and_props(rank)
        if path is None:
            return None

        if inputs_only:
            return [path[t] for t in self.graph.input_timesteps]
        else:
            return path

    def get_props(self, rank: int) -> dict:
        """Retrieve properties of the path with a given rank.

        Parameters
        ----------
        rank : int
            The rank of the path for which to retrieve the properties.

        Returns
        -------
        dict
            A dictionary with the path properties
        """
        _, props = self.get_path_and_props(rank)
        return props

    def get_base_path(
        self, rank: int, inputs_only: bool = False
    ) -> BaseGraphPath | None:
        """Return the path of given rank, but transformed into a path in the
        base graph rather than the rollout.

        Nodes in the rollout graph are of the form `(time, node)`, where
        `node` is a node in the base graph. This function returns a list
        of base graph nodes for a given rank, so discards the time indices.
        The START and END nodes are also removed.

        Parameters
        ----------
        rank : int
            The rank for which the path is to be retrieved.
        inputs_only : bool, optional
            If True, only the nodes corresponding to inputs are returned

        Returns
        -------
        BaseGraphPath | None
            A path in the base graph. If inputs_only is True, only nodes
            corresponding to inputs (not the possible gaps in between) are
            returned. Otherwise, the full path is returned, which has the
            same length as the rollout, minus the START and END nodes. If
            no path with the given rank exists, None is returned.
        """
        rollout_path, _ = self.get_path_and_props(rank)
        if rollout_path is None:
            return None

        base_path = [node for _, node in rollout_path]
        if inputs_only:
            return [base_path[t] for t in self.graph.input_timesteps]
        else:
            return base_path[1:-1]
