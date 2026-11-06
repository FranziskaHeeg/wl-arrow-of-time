"""Static representations of temporal graphs.

* temporal event graph (Definition 4)
* augmented event graph (Definition 8)
* compressed augmented event graph (Definition 10, Appendix B)
* time-concatenated static graph (Definition 5), used as input for the GAT baseline
"""

from collections import defaultdict

import pathpyG as pp
import torch


def temporal_event_graph(g: pp.TemporalGraph, delta: int = 1, mapping: dict = None) -> pp.Graph:
    """Temporal event graph of `g` with maximum waiting time `delta`.

    Nodes are the timestamped edges `(v, w, t)`, named ``"v-w-t"``; two nodes are connected if the
    corresponding timestamped edges form a time-respecting path of length two.
    """
    ho_index = pp.algorithms.lift_order_temporal(g, delta)
    if mapping:
        m = pp.IndexMap([f"{mapping[v]}-{mapping[w]}-{t}" for (v, w, t) in g.temporal_edges])
    else:
        m = pp.IndexMap([f"{v}-{w}-{t}" for (v, w, t) in g.temporal_edges])
    return pp.Graph.from_edge_index(ho_index, mapping=m, num_nodes=g.m)


def augmented_event_graph(g: pp.TemporalGraph, delta: int = 1, mapping: dict = None) -> pp.Graph:
    """Augmented event graph of `g` with maximum waiting time `delta`.

    Adds the first-order nodes of `g` to its temporal event graph, with edges `v -> (v, w, t)` and
    `(v, w, t) -> w`. Stores the node type in ``data.node_labels`` (0: event node, 1: first-order node)
    and the edge type in ``data.edge_labels`` (0: event graph edge, 1: E^out, 2: E^in).
    """
    eg = temporal_event_graph(g, delta=delta, mapping=mapping)
    event_nodes = set(eg.nodes)

    edge_list = eg.edges
    edge_labels = [0] * len(edge_list)
    for (v, w, t) in g.temporal_edges:
        if mapping:
            v, w = mapping[v], mapping[w]
        edge_list.append((f"{v}-{w}-{t}", w))
        edge_list.append((v, f"{v}-{w}-{t}"))
        edge_labels.append(2)
        edge_labels.append(1)

    au_eg = pp.Graph.from_edge_list(edge_list)
    au_eg.data['edge_labels'] = edge_labels
    au_eg.data['node_labels'] = torch.Tensor([0 if x in event_nodes else 1 for x in au_eg.nodes])
    return au_eg


def graph_equality(g1: pp.Graph, g2: pp.Graph, m: dict = None) -> bool:
    """Checks whether two static graphs are identical under a fixed node mapping `m` (default: identity)."""
    if (g1.n != g2.n) or (g1.m != g2.m):
        return False
    if not m:
        m = {v: v for v in g1.nodes}
    if 'edge_weight' in g1.edge_attrs():
        e1 = set([(m[v], m[w], g1['edge_weight', v, w].item()) for (v, w) in g1.edges])
    else:
        e1 = set([(m[v], m[w]) for (v, w) in g1.edges])
    if 'edge_weight' in g2.edge_attrs():
        e2 = set([(v, w, g2['edge_weight', v, w].item()) for (v, w) in g2.edges])
    else:
        e2 = set([(v, w) for (v, w) in g2.edges])
    return e1 == e2


def augmented_event_graph_compressed(g: pp.TemporalGraph, delta: int = 1) -> pp.Graph:
    """Compressed augmented event graph (Definition 10).

    Connected components of the temporal event graph that are equal after replacing the timestamps of
    each node pair by their rank are merged into one representative, whose edges are weighted by the
    number of merged components (``data.edge_weight``). Node names must not contain ``'-'``.
    """
    eg = temporal_event_graph(g, delta=delta)
    _, labels = pp.algorithms.connected_components(eg)

    # split the event graph into its connected components
    component_edges = defaultdict(list)
    for v, w in eg.edges:
        if labels[eg.mapping.to_idx(v)] == labels[eg.mapping.to_idx(w)]:
            component_edges[labels[eg.mapping.to_idx(v)]].append((v, w))
    components = []
    for c in sorted(set(labels.tolist())):
        if component_edges[c]:
            components.append(pp.Graph.from_edge_list(component_edges[c]))
        else:
            # isolated event: represented by a self-loop, which is dropped again below
            v = next(v for v in eg.nodes if labels[eg.mapping.to_idx(v)] == c)
            components.append(pp.Graph.from_edge_list([(v, v)]))

    # replace timestamps by their rank among the timestamped edges of the same node pair
    for c in components:
        times = defaultdict(list)
        for node in c.nodes:
            base, time = node.rsplit('-', 1)
            times[base].append((float(time), node))
        rename_map = {}
        for base, entries in times.items():
            for i, (_, node) in enumerate(sorted(entries)):
                rename_map[node] = f'{base}-{i}'
        c.mapping = pp.IndexMap([rename_map[v] for v in c.nodes])

    # merge equivalent components
    unique_components = []
    for comp in components:
        for i, (unique_comp, count) in enumerate(unique_components):
            if graph_equality(comp, unique_comp):
                unique_components[i] = (unique_comp, count + 1)
                break
        else:
            unique_components.append((comp, 1))

    # augment each representative with the first-order nodes
    final_components = []
    for i, (comp, count) in enumerate(unique_components):
        comp.mapping = pp.IndexMap([v + f'-c{i}' for v in comp.nodes])
        edge_list = comp.edges
        if edge_list[0][0] == edge_list[0][1]:
            edge_list = []
        for node in comp.nodes:
            v, w, _, _ = node.split('-')
            edge_list.append((node, w))
            edge_list.append((v, node))

        g_comp = pp.Graph.from_edge_list(edge_list)
        g_comp.data.edge_weight = torch.full((g_comp.m,), float(count))
        final_components.append(g_comp)

    g_comp = final_components[0]
    for comp in final_components[1:]:
        g_comp += comp

    g_comp.data.node_labels = torch.Tensor([0 if '-' in v else 1 for v in g_comp.nodes])
    return g_comp


def time_concatenated_graph(g: pp.TemporalGraph) -> pp.Graph:
    """Static graph with one edge per timestamped edge, labelled with its timestamp (``data.edge_labels``)."""
    edge_list = []
    edge_labels = []
    for i in range(len(g.edges)):
        edge_list.append(g.edges[i])
        edge_labels.append(g.data.time[i])

    concat = pp.Graph.from_edge_list(edge_list)
    concat.data.edge_labels = edge_labels
    return concat
