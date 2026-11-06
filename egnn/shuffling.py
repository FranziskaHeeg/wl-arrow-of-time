"""Timestamp randomisation procedures used to build the classification tasks on empirical data (Appendix E).

* Method A (`shift_randomly`): shift every timestamp by a small random offset; mostly preserves
  time-respecting paths.
* Method B (`shuffle_components`): move whole connected components of the temporal event graph apart
  in time; preserves time-respecting paths exactly but changes the global order of events.
* Method C (`shuffle`): randomly permute all timestamps; destroys time-respecting paths.
"""

import random
from collections import Counter, defaultdict

import numpy as np
import pathpyG as pp

from .event_graphs import temporal_event_graph


def get_slices(g: pp.TemporalGraph, num_edges: int = 500, max_slices: int = 10):
    """Cuts `g` into consecutive windows of `num_edges` timestamped edges (at most `max_slices`)."""
    graphs = []
    edges = g.temporal_edges
    for i in range(0, len(edges), num_edges):
        t = pp.TemporalGraph.from_edge_list(edges[i:i + num_edges])
        if t.m < num_edges:
            continue
        graphs.append(t)
        if len(graphs) == max_slices:
            break
    return graphs


def shift_randomly(g: pp.TemporalGraph, delta: int) -> pp.TemporalGraph:
    """Method A: shift each timestamp by a random integer in [0, delta/4], avoiding duplicate events."""
    new_tedges = []
    for (v, w, t) in g.temporal_edges:
        while True:
            off_set = random.randint(0, int(delta / 4))
            if (v, w, t + off_set) not in new_tedges:
                break
        new_tedges.append((v, w, t + off_set))
    return pp.TemporalGraph.from_edge_list(new_tedges)


def shuffle(g: pp.TemporalGraph) -> pp.TemporalGraph:
    """Method C: randomly permute the timestamps of `g` (in place) until no event is duplicated."""
    while True:
        g.shuffle_time()
        if max(Counter(g.temporal_edges).values()) == 1:
            break
    return g


def shuffle_components(g: pp.TemporalGraph, delta: int):
    """Method B: reorder the connected components of the temporal event graph in time.

    Relative timestamps within each component are preserved; consecutive components are separated by
    a random gap between 2*delta and 50*delta. Returns the new temporal graph, the number of components
    and the component sizes.
    """
    eg = temporal_event_graph(g, delta=delta)
    n, labels = pp.algorithms.connected_components(eg)

    nodes_per_component = defaultdict(list)
    for node, label in zip(eg.nodes, labels):
        nodes_per_component[label].append(node)

    intervals = {}
    for label, nodes in nodes_per_component.items():
        times = [node.split('-')[2] for node in nodes]
        intervals[label] = (min(times), max(times))

    max_t = int(max(g.data.time))
    new_edges = []
    for label in np.random.permutation(range(n)):
        off_set = random.randint(2 * delta, 50 * delta) + max_t
        for node in nodes_per_component[label]:
            v, w, t = node.split('-')
            new_edges.append((v, w, int(t) + off_set))
        max_t = int(intervals[label][1]) + off_set

    return pp.TemporalGraph.from_edge_list(new_edges), n, list(Counter(labels).values())
