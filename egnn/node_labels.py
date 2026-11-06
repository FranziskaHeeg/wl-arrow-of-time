"""EGNN with (temporal) node labels on the temporal dissemination benchmarks (RQ4, Table 2, Table 11, Appendix J).

Motivation
----------
In the temporal TU datasets of Oettershagen et al. (SDM 2020) — ``highschool_ct1/2``,
``infectious_ct1/2``, ``dblp_ct1/2``, ``facebook_ct1/2``, ``mit_ct1/2``, ``tumblr_ct1/2`` —
the class of a graph is *not* determined by the temporal topology.  Each graph is an
ego-/sub-network of contacts on which a dissemination (SI-type) process was simulated,
and the class distinguishes two dissemination scenarios.  The process is stored in the
``*_node_labels.txt`` file as a *temporal node label*: every node carries a sequence of
``(time, label)`` pairs, e.g. ``0, 0, 28, 1`` = "susceptible at t=0, infected from t=28",
while ``0, 0`` = "never infected".

Empirically (``highschool_ct1``, 180 graphs):

* structural summary statistics (#nodes, #events) separate the classes at chance level
  (~0.48 accuracy),
* the fraction of infections that are *explained* by a time-respecting contact with an
  already infected node is ``0.03 ± 0.04`` for class 0 and ``0.83 ± 0.18`` for class 1.

So the label carries the signal, but only *in combination with the temporal structure*:
the classes differ in whether the infection times are causally consistent with the
time-respecting paths of the contact sequence.  This is exactly the object an event
graph represents, which makes the task a natural (and, for the EGNN, a fair) benchmark.
A model that sees only the aggregated contact network plus static labels cannot solve
it, because the infected fraction is ~0.5 in both classes.

What this module adds to :mod:`egnn.model`
------------------------------------
1. A raw loader for the temporal TU datasets that **keeps the temporal node labels**
   (``pathpyG``'s TU reader deliberately skips the ragged ``node_labels`` file).
2. Feature construction on the *augmented event graph*: an event node ``(v, w, t)``
   receives the label states of ``v`` and ``w`` *at the time of the event*, a first-order
   node ``v`` receives its own label trajectory.  Four nested feature modes
   (``structure`` / ``static`` / ``temporal`` / ``transition``) give a clean ablation.
3. :class:`EGNNNodeLabels`, an EGNN variant with (a) *type-specific input encoders*,
   since event nodes and first-order nodes carry semantically different features, and
   (b) a *masked readout* that pools over event nodes, first-order nodes, or both.
   The masked mean/max pooling is normalised by the number of nodes of the pooled type
   (the plain masking in :class:`egnn.model.EGNN` zeroes rows before ``global_mean_pool``, so
   the mean is diluted by the nodes of the other type).
4. Reference baselines: a static (time-aggregated) graph with the same model, and the
   hand-crafted "causal consistency" feature.

Typical use::

    from egnn.node_labels import load_dataset, run_experiment
    res = run_experiment('highschool_ct1', delta=1, feature_mode='temporal', epochs=60)
"""

from __future__ import annotations

import contextlib
import io
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module, ModuleList, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import DirGNNConv, GINConv, GraphConv
from torch_geometric.nn.pool import global_add_pool, global_max_pool

import pathpyG as pp

from .model import DirGNN


class GIN(GINConv):
    """GIN convolution with a single linear layer, usable inside :class:`DirGNNConv`."""

    def __init__(self, in_channels, out_channels, train_eps=False, eps=0.):
        super().__init__(nn=Sequential(Linear(in_channels, out_channels)), train_eps=train_eps, eps=eps)
        self.in_channels = in_channels
        self.out_channels = out_channels


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

#: TU datasets that store a dissemination process as temporal node labels.
TEMPORAL_TU_DATASETS = (
    'dblp_ct1', 'dblp_ct2',
    'facebook_ct1', 'facebook_ct2',
    'highschool_ct1', 'highschool_ct2',
    'infectious_ct1', 'infectious_ct2',
    'mit_ct1', 'mit_ct2',
    'tumblr_ct1', 'tumblr_ct2',
)

_DEFAULT_TU_ROOT = Path.home() / '.pathpyG' / 'tu_datasets'
_DEFAULT_CACHE = Path('cache') / 'node_labels'


def _ensure_raw(name: str, root: Path) -> Path:
    """Return the raw directory of a TU dataset, downloading it if necessary."""
    raw_dir = root / name / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / f'{name}_A.txt').exists():
        return raw_dir

    import urllib.request
    import zipfile

    url = f'https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip'
    print(f'Downloading {url} ...')
    with urllib.request.urlopen(url) as resp:
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            zf.extractall(root / name)
    extracted = root / name / name
    if extracted.exists():
        for f in extracted.iterdir():
            f.rename(raw_dir / f.name)
        extracted.rmdir()
    return raw_dir


def _read_rows(path: Path) -> List[List[float]]:
    return [[float(v) for v in line.split(',')]
            for line in path.read_text().strip().splitlines()]


def parse_temporal_node_labels(rows: Sequence[Sequence[float]]) -> Tuple[np.ndarray, int]:
    """Parse ragged ``(time, label)`` pairs into infection times.

    Args:
        rows: one row per node, flat list ``[t_0, l_0, t_1, l_1, ...]``.

    Returns:
        ``(infection_time, max_label)`` where ``infection_time[i]`` is the first time at
        which node ``i`` takes a non-zero label, and ``inf`` if it never does.
    """
    inf_time = np.full(len(rows), np.inf)
    max_label = 0
    for i, row in enumerate(rows):
        for j in range(0, len(row) - 1, 2):
            t, lab = row[j], int(row[j + 1])
            max_label = max(max_label, lab)
            if lab > 0 and not np.isfinite(inf_time[i]):
                inf_time[i] = t
    return inf_time, max_label


def load_temporal_tu_raw(name: str, root: Optional[Path] = None) -> List[Dict]:
    """Load a temporal TU dataset from its raw files, *including* node labels.

    Args:
        name: dataset name, e.g. ``'highschool_ct1'``.
        root: TU cache directory (default ``~/.pathpyG/tu_datasets``).

    Returns:
        One record per graph with keys ``edge_index`` (``[2, m]``, long, 0-based),
        ``time`` (``[m]``, long), ``infection_time`` (``[n]``, float, ``inf`` = never),
        ``num_nodes``, ``y`` and ``graph_id``.
    """
    root = Path(root) if root is not None else _DEFAULT_TU_ROOT
    raw = _ensure_raw(name, root)

    edges = np.array(_read_rows(raw / f'{name}_A.txt'), dtype=np.int64) - 1        # [E, 2]
    edge_attr = np.array([r[0] for r in _read_rows(raw / f'{name}_edge_attributes.txt')])
    indicator = np.array([int(r[0]) for r in _read_rows(raw / f'{name}_graph_indicator.txt')]) - 1
    graph_labels = np.array([int(r[0]) for r in _read_rows(raw / f'{name}_graph_labels.txt')])
    inf_time_all, _ = parse_temporal_node_labels(_read_rows(raw / f'{name}_node_labels.txt'))

    src, dst = edges[:, 0], edges[:, 1]
    records = []
    for gid in range(len(graph_labels)):
        node_mask = indicator == gid
        node_idx = np.nonzero(node_mask)[0]
        offset = int(node_idx.min())
        edge_mask = node_mask[src] & node_mask[dst]
        local = (edges[edge_mask] - offset).T                                      # [2, m]
        records.append({
            'edge_index': torch.tensor(local, dtype=torch.long),
            'time': torch.tensor(edge_attr[edge_mask], dtype=torch.long),
            'infection_time': torch.tensor(inf_time_all[node_idx], dtype=torch.float),
            'num_nodes': int(node_idx.size),
            'y': int(graph_labels[gid]),
            'graph_id': gid,
        })
    return records


# ---------------------------------------------------------------------------
# Node features on the augmented event graph
# ---------------------------------------------------------------------------

#: Column layout of the full feature matrix built by :func:`_all_features`.
FEATURE_COLUMNS = [
    'is_event',            # 0  node type indicator
    'is_first_order',      # 1
    'ev_src_state',        # 2  label state of the source *before* the event
    'ev_dst_state',        # 3  label state of the target *before* the event
    'ev_src_state_after',  # 4  label state of the source after the event (t + lag)
    'ev_dst_state_after',  # 5  label state of the target after the event (t + lag)
    'ev_time',             # 6  event time, normalised to [0, 1]
    'fo_state_final',      # 7  first-order node: ever labelled?
    'fo_label_time',       # 8  first-order node: label time, normalised to [0, 1]
    'ev_src_ever',         # 9  source ever labelled (no timing)
    'ev_dst_ever',         # 10 target ever labelled (no timing)
]

#: Nested feature modes, from purely structural to fully label-aware.
FEATURE_MODES: Dict[str, List[int]] = {
    # node type only — the features used for RQ1-RQ3
    'structure':  [0, 1],
    # labels without any timing: who was ever infected, but not when
    'static':     [0, 1, 9, 10, 7],
    # label state at event time + event/label times: the model has to relate the label
    # time of a node to the times of its incident events (via the event graph)
    'temporal':   [0, 1, 2, 3, 6, 7, 8],
    # additionally the post-event state, i.e. transmission is visible on the event itself
    'transition': [0, 1, 2, 3, 4, 5, 6, 7, 8],
}


def feature_dim(feature_mode: str) -> int:
    return len(FEATURE_MODES[feature_mode])


def _all_features(edge_index: Tensor, time: Tensor, infection_time: Tensor,
                  num_nodes: int, infection_lag: int = 1) -> Tensor:
    """Build the full ``[m + n, len(FEATURE_COLUMNS)]`` feature matrix.

    Rows ``0..m-1`` are event nodes (in the order of ``edge_index`` / ``time``),
    rows ``m..m+n-1`` are the first-order nodes.
    """
    m = time.numel()
    src, dst = edge_index[0], edge_index[1]
    tau = infection_time                                  # [n], inf == never labelled
    t = time.float()
    t_max = float(t.max().item()) if m > 0 else 1.0
    t_max = max(t_max, 1.0)

    x = torch.zeros(m + num_nodes, len(FEATURE_COLUMNS))
    # --- event nodes -------------------------------------------------------
    x[:m, 0] = 1.0
    x[:m, 2] = (tau[src] < t).float()
    x[:m, 3] = (tau[dst] < t).float()
    x[:m, 4] = (tau[src] <= t + infection_lag).float()
    x[:m, 5] = (tau[dst] <= t + infection_lag).float()
    x[:m, 6] = t / t_max
    x[:m, 9] = torch.isfinite(tau[src]).float()
    x[:m, 10] = torch.isfinite(tau[dst]).float()
    # --- first-order nodes -------------------------------------------------
    labelled = torch.isfinite(tau)
    x[m:, 1] = 1.0
    x[m:, 7] = labelled.float()
    x[m:, 8] = torch.where(labelled, tau / t_max, torch.zeros_like(tau))
    return x


def build_augmented_event_graph(record: Dict, delta: int = 1,
                                feature_mode: str = 'temporal',
                                infection_lag: int = 1) -> Data:
    """Convert one temporal-graph record into an augmented event graph.

    Node ``i < m`` is the event ``(v, w, t)`` given by column ``i`` of the (time-sorted)
    temporal edge index; node ``m + j`` is the first-order node ``j``.  Edges are

    * event → event   (``edge_labels == 0``): the second-order/event-graph edges of
      :func:`utils_network_generation.temporal_event_graph`, i.e. pairs of events
      ``(u, v, t) → (v, w, t')`` with ``0 < t' - t <= delta``,
    * node → event    (``edge_labels == 1``) for the source of the event,
    * event → node    (``edge_labels == 2``) for the target of the event.

    Args:
        record: an entry of :func:`load_temporal_tu_raw`.
        delta: maximum time difference between two causally connected events.
        feature_mode: key of :data:`FEATURE_MODES`.
        infection_lag: time offset at which a transmission becomes visible in the label
            trajectory (1 for the ``_ct*`` datasets, where a contact at ``t`` produces a
            label change at ``t + 1``).

    Returns:
        A PyG ``Data`` object with ``x``, ``edge_index``, ``edge_labels``,
        ``node_labels`` (0 = event node, 1 = first-order node) and ``y``.
    """
    n = record['num_nodes']
    tg = pp.TemporalGraph(Data(edge_index=record['edge_index'].clone(),
                               time=record['time'].clone(),
                               num_nodes=n))
    # canonical (time-sorted) event order — identical to the one lift_order_temporal uses
    ei = torch.stack([tg.data.edge_index[0].clone(), tg.data.edge_index[1].clone()])
    t = tg.data.time.clone()
    m = t.numel()

    try:
        with contextlib.redirect_stderr(io.StringIO()):    # silence the tqdm bar
            ho_index = pp.algorithms.lift_order_temporal(tg, delta)
        ho_index = torch.as_tensor(ho_index, dtype=torch.long).view(2, -1)
    except RuntimeError:
        # no pair of events is causally connected within delta -> no event-graph edges
        ho_index = torch.empty(2, 0, dtype=torch.long)

    events = torch.arange(m)
    src_edges = torch.stack([m + ei[0], events])          # first-order node -> event
    dst_edges = torch.stack([events, m + ei[1]])          # event -> first-order node
    edge_index = torch.cat([ho_index, src_edges, dst_edges], dim=1)
    edge_labels = torch.cat([torch.zeros(ho_index.size(1)),
                             torch.ones(m), torch.full((m,), 2.0)])

    x_full = _all_features(ei, t, record['infection_time'], n, infection_lag=infection_lag)
    x = x_full[:, FEATURE_MODES[feature_mode]].contiguous()

    node_labels = torch.cat([torch.zeros(m), torch.ones(n)])
    return Data(x=x, edge_index=edge_index, edge_labels=edge_labels,
                node_labels=node_labels, num_nodes=m + n,
                y=torch.tensor([float(record['y'])]),
                num_events=m, num_first_order=n)


def build_static_graph(record: Dict, feature_mode: str = 'temporal') -> Data:
    """Baseline representation: the time-aggregated contact graph with node labels.

    Every node keeps its label information (``fo_state_final`` / ``fo_label_time``), but
    the temporal ordering of the contacts is discarded.  Used to show that the label
    alone, on a static topology, does not identify the class.
    """
    n = record['num_nodes']
    ei = record['edge_index']
    edge_index = torch.unique(ei, dim=1) if ei.numel() else ei
    tau = record['infection_time']
    t_max = max(float(record['time'].max().item()) if record['time'].numel() else 1.0, 1.0)
    labelled = torch.isfinite(tau)

    x_full = torch.zeros(n, len(FEATURE_COLUMNS))
    x_full[:, 1] = 1.0
    x_full[:, 7] = labelled.float()
    x_full[:, 8] = torch.where(labelled, tau / t_max, torch.zeros_like(tau))
    x = x_full[:, FEATURE_MODES[feature_mode]].contiguous()

    return Data(x=x, edge_index=edge_index, node_labels=torch.ones(n), num_nodes=n,
                y=torch.tensor([float(record['y'])]), num_events=0, num_first_order=n)


def load_dataset(name: str, delta: int = 1, feature_mode: str = 'temporal',
                 representation: str = 'event_graph', infection_lag: int = 1,
                 limit: Optional[int] = None, cache_dir: Optional[Path] = None,
                 verbose: bool = True) -> Tuple[List[Data], Dict]:
    """Load a temporal TU dataset and convert it to the requested representation.

    Args:
        name: dataset name, e.g. ``'highschool_ct1'``.
        delta: maximum time difference between causally connected events.
        feature_mode: key of :data:`FEATURE_MODES`.
        representation: ``'event_graph'`` (augmented event graph) or ``'static'``
            (time-aggregated contact graph).
        infection_lag: see :func:`build_augmented_event_graph`.
        limit: only convert the first ``limit`` graphs (for quick tests).
        cache_dir: directory for the pickled conversion; ``None`` (default) disables
            caching — the conversion takes a few seconds, while the pickles are large
            (~10 MB per variant of ``highschool_ct1``).  Pass :data:`_DEFAULT_CACHE` or
            any path to enable it.
        verbose: print progress.

    Returns:
        ``(dataset, meta)`` with ``meta`` holding ``num_classes``, ``in_channels`` and
        dataset size statistics.
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f'unknown feature_mode {feature_mode!r}, '
                         f'choose from {sorted(FEATURE_MODES)}')

    key = f'{name}_{representation}_d{delta}_{feature_mode}_lag{infection_lag}'
    if limit is not None:
        key += f'_n{limit}'
    cache_file = Path(cache_dir) / f'{key}.pkl' if cache_dir is not None else None
    if cache_file is not None and cache_file.exists():
        with open(cache_file, 'rb') as f:
            dataset, meta = pickle.load(f)
        if verbose:
            print(f'Loaded {len(dataset)} graphs from cache {cache_file.name}')
        return dataset, meta

    records = load_temporal_tu_raw(name)
    if limit is not None:
        records = records[:limit]

    dataset = []
    t0 = time.time()
    for i, rec in enumerate(records):
        if representation == 'event_graph':
            data = build_augmented_event_graph(rec, delta=delta, feature_mode=feature_mode,
                                               infection_lag=infection_lag)
        elif representation == 'static':
            data = build_static_graph(rec, feature_mode=feature_mode)
        else:
            raise ValueError(f'unknown representation {representation!r}')
        dataset.append(data)
        if verbose and (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(records)} graphs converted '
                  f'({time.time() - t0:.1f}s)')

    labels = sorted({int(d.y.item()) for d in dataset})
    meta = {
        'name': name,
        'representation': representation,
        'delta': delta,
        'feature_mode': feature_mode,
        'labels': labels,
        'num_classes': len(labels),
        'in_channels': feature_dim(feature_mode),
        'num_graphs': len(dataset),
        'avg_nodes': float(np.mean([d.num_nodes for d in dataset])),
        'avg_edges': float(np.mean([d.edge_index.size(1) for d in dataset])),
    }
    if verbose:
        print(f"{name}: {meta['num_graphs']} graphs, "
              f"avg {meta['avg_nodes']:.0f} nodes / {meta['avg_edges']:.0f} edges, "
              f"classes {labels} ({time.time() - t0:.1f}s)")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((dataset, meta), f)
    return dataset, meta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def masked_pool(x: Tensor, batch: Tensor, mask: Tensor, pooling: str) -> Tensor:
    """Pool ``x`` over the nodes selected by ``mask``, per graph in the batch.

    In contrast to zeroing the rows of the other node type before a global pooling
    (as done in :class:`EGNN.EGNN`), ``'mean'`` normalises by the number of *selected*
    nodes and ``'max'`` ignores unselected rows entirely.
    """
    w = mask.float().unsqueeze(-1)
    if pooling == 'add':
        return global_add_pool(x * w, batch)
    if pooling == 'mean':
        summed = global_add_pool(x * w, batch)
        counts = global_add_pool(w, batch).clamp(min=1.0)
        return summed / counts
    if pooling == 'max':
        neg = torch.finfo(x.dtype).min
        pooled = global_max_pool(torch.where(mask.unsqueeze(-1), x, torch.full_like(x, neg)), batch)
        return torch.where(pooled <= neg / 2, torch.zeros_like(pooled), pooled)
    raise ValueError(f'unknown pooling {pooling!r}')


class EGNNNodeLabels(Module):
    """EGNN variant for augmented event graphs carrying node-label features.

    Args:
        in_channels: number of input features (``feature_dim(feature_mode)``).
        out_channels: 1 for binary classification (BCE), ``num_classes`` otherwise.
        hidden_dims: widths of the message-passing layers.
        dense_dims: widths of the dense head (the readout is projected onto
            ``dense_dims[0]``).
        pooling: ``'add'``, ``'mean'`` or ``'max'``.
        readout: which node type is pooled — ``'event'``, ``'first_order'`` or
            ``'both'`` (concatenation of the two pooled vectors).
        type_encoder: use separate input projections for event and first-order nodes.
            They carry different features (endpoint states at an event vs. the label
            trajectory of a node), so a shared projection has to squeeze both into one
            linear map.
        dirGNN: use the direction-aware convolution of :mod:`EGNN` (recommended, the
            event graph is directed by time).
        gin: use GIN instead of GraphConv inside the directional wrapper.
        batch_norm: apply batch normalisation between message-passing layers.
        dropout: dropout probability applied to the graph-level representation.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 hidden_dims: Sequence[int] = (32, 16, 8),
                 dense_dims: Sequence[int] = (8, 4),
                 pooling: str = 'mean',
                 readout: str = 'both',
                 type_encoder: bool = True,
                 dirGNN: bool = True,
                 gin: bool = False,
                 batch_norm: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        if readout not in ('event', 'first_order', 'both'):
            raise ValueError(f'unknown readout {readout!r}')

        self.in_channels = in_channels
        self.hidden_dims = list(hidden_dims)
        self.dense_dims = list(dense_dims)
        self.pooling = pooling
        self.readout = readout
        self.type_encoder = type_encoder
        self.dropout = dropout

        enc_dim = self.hidden_dims[0]
        self.enc_event = Linear(in_channels, enc_dim)
        self.enc_node = Linear(in_channels, enc_dim) if type_encoder else None

        def make_conv(i: int, o: int):
            if not dirGNN:
                return GraphConv(i, o)
            if gin:
                return DirGNNConv(GIN(i, o))
            return DirGNN(GraphConv(i, o), GraphConv(i, o))

        self.convs = ModuleList()
        self.batch_norms = ModuleList() if batch_norm else None
        dims = [enc_dim] + self.hidden_dims
        for i in range(1, len(dims)):
            self.convs.append(make_conv(dims[i - 1], dims[i]))
            if batch_norm:
                self.batch_norms.append(BatchNorm1d(dims[i]))

        pooled_dim = self.hidden_dims[-1] * (2 if readout == 'both' else 1)
        self.dense = ModuleList()
        self.dense.append(Linear(pooled_dim, self.dense_dims[0]))
        for i in range(1, len(self.dense_dims)):
            self.dense.append(Linear(self.dense_dims[i - 1], self.dense_dims[i]))
        self.dense.append(Linear(self.dense_dims[-1], out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.enc_event] + ([self.enc_node] if self.enc_node is not None else []):
            torch.nn.init.xavier_normal_(lin.weight, gain=1.0)
            torch.nn.init.zeros_(lin.bias)
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.dense:
            torch.nn.init.xavier_normal_(lin.weight, gain=1.0)
            torch.nn.init.zeros_(lin.bias)

    def encode(self, x: Tensor, is_event: Tensor) -> Tensor:
        """Type-specific input projection of the heterogeneous node features."""
        h = self.enc_event(x)
        if self.enc_node is not None:
            h = torch.where(is_event.unsqueeze(-1), h, self.enc_node(x))
        return F.elu(h)

    def forward(self, x: Tensor, edge_index: Tensor, node_labels: Tensor,
                batch: Tensor, edge_weights: Optional[Tensor] = None):
        """Returns ``(logits, node_embedding, graph_embedding)``."""
        is_event = node_labels.view(-1) == 0

        # GIN / DirGNNConv do not accept edge weights, so only forward them if given
        conv_kwargs = {} if edge_weights is None else {'edge_weight': edge_weights}

        h = self.encode(x, is_event)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, **conv_kwargs)
            if self.batch_norms is not None:
                h = self.batch_norms[i](h)
            h = F.elu(h)
        embedding = h

        pooled = []
        if self.readout in ('event', 'both'):
            pooled.append(masked_pool(h, batch, is_event, self.pooling))
        if self.readout in ('first_order', 'both'):
            pooled.append(masked_pool(h, batch, ~is_event, self.pooling))
        graph_embedding = torch.cat(pooled, dim=-1)

        out = graph_embedding
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        for i, lin in enumerate(self.dense):
            out = lin(out)
            if i < len(self.dense) - 1:
                out = F.elu(out)
        return out, embedding, graph_embedding


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def stratified_split(dataset: Sequence[Data], fractions=(0.7, 0.1, 0.2),
                     seed: int = 0) -> List[List[int]]:
    """Split indices into (train, val, test) keeping the class ratio in every part."""
    rng = np.random.default_rng(seed)
    y = np.array([int(d.y.item()) for d in dataset])
    splits: List[List[int]] = [[] for _ in fractions]
    for cls in np.unique(y):
        idx = np.nonzero(y == cls)[0]
        rng.shuffle(idx)
        bounds = np.cumsum([int(round(f * len(idx))) for f in fractions[:-1]])
        for part, chunk in enumerate(np.split(idx, bounds)):
            splits[part].extend(chunk.tolist())
    for part in splits:
        rng.shuffle(part)
    return splits


def _forward_batch(model, data, device):
    data = data.to(device)
    out, _, _ = model(x=data.x, edge_index=data.edge_index,
                      node_labels=data.node_labels.view(-1, 1), batch=data.batch,
                      edge_weights=None)
    return out, data


def train_epoch(loader, model, optimizer, criterion, num_classes, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for data in loader:
        optimizer.zero_grad()
        out, data = _forward_batch(model, data, device)
        if num_classes == 2:
            loss = criterion(out.view(-1), data.y.view(-1).float())
            pred = (out.view(-1) > 0).float()
        else:
            loss = criterion(out, data.y.view(-1).long())
            pred = out.argmax(dim=1).float()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct += int(pred.eq(data.y.view(-1).float()).sum())
        n += data.num_graphs
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(loader, model, num_classes, device):
    model.eval()
    correct, n = 0, 0
    for data in loader:
        out, data = _forward_batch(model, data, device)
        pred = (out.view(-1) > 0).float() if num_classes == 2 else out.argmax(dim=1).float()
        correct += int(pred.eq(data.y.view(-1).float()).sum())
        n += data.num_graphs
    return correct / n


def run_experiment(name: str = 'highschool_ct1',
                   delta: int = 1,
                   feature_mode: str = 'temporal',
                   representation: str = 'event_graph',
                   hidden_dims: Sequence[int] = (32, 16, 8),
                   dense_dims: Sequence[int] = (8, 4),
                   pooling: str = 'mean',
                   readout: str = 'both',
                   type_encoder: bool = True,
                   gin: bool = False,
                   dirGNN: bool = True,
                   batch_norm: bool = False,
                   dropout: float = 0.0,
                   lr: float = 0.01,
                   weight_decay: float = 0.0,
                   epochs: int = 60,
                   batch_size: int = 32,
                   seed: int = 0,
                   fractions=(0.7, 0.1, 0.2),
                   device: Optional[torch.device] = None,
                   dataset: Optional[List[Data]] = None,
                   meta: Optional[Dict] = None,
                   verbose: bool = True) -> Dict:
    """Train :class:`EGNNNodeLabels` on one dataset and return the run summary.

    The reported ``test_acc`` is taken at the epoch with the best validation accuracy
    (ties broken by the later epoch), ``final_test_acc`` is the accuracy after the last
    epoch.  Pass ``dataset``/``meta`` to reuse an already converted dataset.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset is None:
        dataset, meta = load_dataset(name, delta=delta, feature_mode=feature_mode,
                                     representation=representation, verbose=verbose)
    assert meta is not None
    num_classes = meta['num_classes']
    out_channels = 1 if num_classes == 2 else num_classes

    tr, va, te = stratified_split(dataset, fractions=fractions, seed=seed)
    loaders = [DataLoader([dataset[i] for i in part], batch_size=batch_size,
                          shuffle=(k == 0))
               for k, part in enumerate((tr, va, te))]
    train_loader, val_loader, test_loader = loaders

    model = EGNNNodeLabels(in_channels=meta['in_channels'], out_channels=out_channels,
                           hidden_dims=hidden_dims, dense_dims=dense_dims,
                           pooling=pooling, readout=readout, type_encoder=type_encoder,
                           dirGNN=dirGNN, gin=gin, batch_norm=batch_norm,
                           dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = (torch.nn.BCEWithLogitsLoss() if num_classes == 2
                 else torch.nn.CrossEntropyLoss())

    history = {'loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
    best = {'val_acc': -1.0, 'test_acc': 0.0, 'epoch': 0}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss, train_acc = train_epoch(train_loader, model, optimizer, criterion,
                                      num_classes, device)
        val_acc = evaluate(val_loader, model, num_classes, device) if len(va) else train_acc
        test_acc = evaluate(test_loader, model, num_classes, device)
        history['loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        if val_acc >= best['val_acc']:
            best = {'val_acc': val_acc, 'test_acc': test_acc, 'epoch': epoch}
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f'  epoch {epoch:3d} | loss {loss:.4f} | train {train_acc:.3f} '
                  f'| val {val_acc:.3f} | test {test_acc:.3f}')

    result = {
        'dataset': meta['name'], 'representation': meta['representation'],
        'feature_mode': feature_mode, 'delta': delta, 'seed': seed,
        'readout': readout, 'pooling': pooling, 'gin': gin,
        'in_channels': meta['in_channels'],
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'test_acc': best['test_acc'], 'val_acc': best['val_acc'],
        'best_epoch': best['epoch'], 'final_test_acc': history['test_acc'][-1],
        'train_time_s': time.time() - t0, 'history': history,
    }
    if verbose:
        print(f"  -> test acc {result['test_acc']:.3f} (val {result['val_acc']:.3f}, "
              f"epoch {result['best_epoch']}), {result['train_time_s']:.1f}s")
    return result


def run_ablation(name: str = 'highschool_ct1',
                 feature_modes: Sequence[str] = ('structure', 'static', 'temporal', 'transition'),
                 seeds: Sequence[int] = (0, 1, 2),
                 delta: int = 1,
                 representation: str = 'event_graph',
                 verbose: bool = False, **kwargs) -> List[Dict]:
    """Run every ``feature_mode`` for every seed; datasets are converted once per mode."""
    results = []
    for mode in feature_modes:
        dataset, meta = load_dataset(name, delta=delta, feature_mode=mode,
                                     representation=representation, verbose=True)
        for seed in seeds:
            res = run_experiment(name=name, delta=delta, feature_mode=mode,
                                 representation=representation, seed=seed,
                                 dataset=dataset, meta=meta, verbose=verbose, **kwargs)
            results.append(res)
            print(f'{name} | {representation:11s} | {mode:10s} | seed {seed} '
                  f'| test {res["test_acc"]:.3f}')
    return results


def summarize(results: Sequence[Dict], keys=('dataset', 'representation', 'feature_mode')) -> Dict:
    """Aggregate ``run_experiment`` results into ``mean ± std`` of the test accuracy."""
    groups: Dict[Tuple, List[float]] = {}
    for r in results:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r['test_acc'])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in groups.items()}


# ---------------------------------------------------------------------------
# Reference baseline: hand-crafted causal-consistency feature
# ---------------------------------------------------------------------------

def causal_consistency_features(records: Sequence[Dict], infection_lag: int = 1) -> np.ndarray:
    """Hand-crafted per-graph features describing the label/topology interplay.

    Columns: fraction of labelled nodes whose label time is preceded by a contact with an
    already labelled node ("explained"), fraction of events whose source/target is
    already labelled, fraction of transmission events, labelled fraction, median label
    time (normalised) and graph size.
    """
    feats = []
    for rec in records:
        ei, t = rec['edge_index'].numpy(), rec['time'].numpy().astype(float)
        tau = rec['infection_time'].numpy()
        src, dst = ei[0], ei[1]
        t_max = max(t.max() if t.size else 1.0, 1.0)

        labelled = np.isfinite(tau)
        explained, n_lab = 0, 0
        for v in np.nonzero(labelled)[0]:
            n_lab += 1
            inc = (dst == v) & (t <= tau[v]) & (t >= tau[v] - infection_lag)
            if inc.any() and np.any(tau[src[inc]] < tau[v]):
                explained += 1
        s_before = tau[src] < t
        transmit = s_before & (tau[dst] >= t) & (tau[dst] <= t + infection_lag)
        feats.append([
            explained / max(n_lab, 1),
            s_before.mean() if t.size else 0.0,
            (tau[dst] < t).mean() if t.size else 0.0,
            transmit.mean() if t.size else 0.0,
            labelled.mean(),
            (np.median(tau[labelled]) / t_max) if labelled.any() else -1.0,
            rec['num_nodes'],
        ])
    return np.array(feats)


def causal_consistency_baseline(name: str = 'highschool_ct1', cv: int = 5,
                                infection_lag: int = 1) -> Dict:
    """Cross-validated logistic regression / random forest on the hand-crafted features.

    Reported as a reference point: it tells us how much of the task is captured by the
    single "are the labels causally consistent with the contact sequence" statistic.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    records = load_temporal_tu_raw(name)
    X = causal_consistency_features(records, infection_lag=infection_lag)
    y = np.array([r['y'] for r in records])
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return {
        'dataset': name,
        'logreg_all': float(cross_val_score(logreg, X, y, cv=cv).mean()),
        'rf_all': float(cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0),
                                        X, y, cv=cv).mean()),
        'rf_explained_only': float(cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0),
                                                   X[:, :1], y, cv=cv).mean()),
        'rf_size_only': float(cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0),
                                              X[:, -1:], y, cv=cv).mean()),
    }


if __name__ == '__main__':
    res = run_experiment('highschool_ct1', delta=1, feature_mode='temporal', epochs=40)
    print(res['test_acc'])
