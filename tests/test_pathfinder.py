# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2024 Bas Cornelissen
# -------------------------------------------------------------------
import unittest
from music21.pitch import Pitch
import networkx as nx


# Local imports
from delasol.utils.music import as_pitch_list
from delasol.graphs.rollout_graph import RolloutGraph
from delasol.pathfinders.simple_pathfinder import SimplePathfinder
from delasol.pathfinders.segmented_pathfinder import SegmentedPathfinder, SegmentsGraph


class TestSimplePathfinder(unittest.TestCase):

    def test_small_circulant_graph(self):
        graph = nx.circulant_graph(3, [1])
        nx.set_edge_attributes(graph, 1, "weight")
        rollout = RolloutGraph(graph, [0, 1, 2])
        pathfinder = SimplePathfinder(rollout)
        path, props = pathfinder.get_path_and_props(0)
        self.assertListEqual(path, [(0, "START"), (1, 0), (2, 1), (3, 2), (4, "END")])
        self.assertEqual(props["weight"], 2)

    def test_base_graph_path(self):
        graph = nx.circulant_graph(6, [1])
        nx.set_edge_attributes(graph, 1, "weight")
        rollout = RolloutGraph(graph, [0, 2])
        pathfinder = SimplePathfinder(rollout)
        path = pathfinder.get_base_path(0)
        self.assertListEqual(path, [0, 1, 2])

    @unittest.skip
    def test_solmization_example(self):
        gamut = get_gamut("hard-continental")
        seq = as_pitch_list("C3 E3 G3")
        sol = SolmizationGraph(gamut, seq)
        pathfinder = SimplePathfinder(sol)
        base_path = pathfinder.get_base_path(0)
        pitches = [pitch for _, pitch in base_path]
        self.assertListEqual(pitches, as_pitch_list("C3 D3 E3 F3 G3"))

    @unittest.skip
    def test_input_position_only(self):
        gamut = get_gamut("hard-continental")
        seq = as_pitch_list("C3 E3 G3")
        sol = SolmizationGraph(gamut, seq)
        pathfinder = SimplePathfinder(sol)
        base_path = pathfinder.get_base_path(0, inputs_only=True)
        pitches = [pitch for _, pitch in base_path]
        self.assertListEqual(pitches, seq)

    def test_weights(self):
        """Circulant graph with weights maximum of nodes. See SimplePathfinder doctest."""
        graph = nx.circulant_graph(6, [1])
        for u, v in graph.edges:
            graph[u][v]["weight"] = max(u, v)
        rollout = RolloutGraph(graph, [0, 3, 0])
        pf = SimplePathfinder(rollout)
        self.assertListEqual(pf.get_base_path(0), [0, 1, 2, 3, 2, 1, 0])
        self.assertEqual(pf.get_props(0)["weight"], 12)
        self.assertListEqual(pf.get_base_path(1), [0, 5, 4, 3, 2, 1, 0])
        self.assertEqual(pf.get_props(1)["weight"], 20)
        self.assertListEqual(pf.get_base_path(2), [0, 1, 2, 3, 4, 5, 0])
        self.assertEqual(pf.get_props(2)["weight"], 20)
        self.assertListEqual(pf.get_base_path(3), [0, 5, 4, 3, 4, 5, 0])
        self.assertEqual(pf.get_props(3)["weight"], 28)


class TestSegmentsGraph(unittest.TestCase):
    def test_init(self):
        graph = nx.circulant_graph(6, [1])
        for u, v in graph.edges:
            graph[u][v]["weight"] = max(u, v)
        rollout = RolloutGraph(graph, [0, 3, 4, 3, 0])
        sg = SegmentsGraph(rollout)
        paths = list(sg.iter_shortest_paths())
        weights = [nx.path_weight(rollout, p, "weight") for p in paths]
        self.assertListEqual(weights, [20, 28, 28, 36])


class TestSegmentedPathfinder(unittest.TestCase):
    def test_init(self):
        graph = nx.circulant_graph(6, [1])
        for u, v in graph.edges:
            graph[u][v]["weight"] = max(u, v)
        rollout = RolloutGraph(graph, [0, 3, 4, 3, 0])

        # First run the exhaustive pathfinder
        pf = SimplePathfinder(rollout)
        weights = [pf.get_props(i)["weight"] for i in range(4)]
        self.assertListEqual(weights, [20, 28, 28, 36])

    def test_init_2(self):
        graph = nx.circulant_graph(6, [1])
        for u, v in graph.edges:
            graph[u][v]["weight"] = max(u, v)
        rollout = RolloutGraph(graph, [0, 3, 4, 3, 0])

        # Then the segmented pathfinder
        segm_pf = SegmentedPathfinder(rollout)
        segm_weights = [segm_pf.get_props(i)["weight"] for i in range(4)]
        self.assertListEqual(segm_weights, [20, 28, 28, 36])
