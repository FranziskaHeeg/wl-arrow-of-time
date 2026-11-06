"""De Bruijn GNN (Qarkaxhija et al., 2022; pathpyG implementation) adapted to temporal graph classification.

Path-aware baseline in Table 1: classes 1/3 vs. classes 2/4 of the empirical tasks.
"""

import json
import os
import pickle
import time

import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import degree

import pathpyG as pp
from pathpyG.nn.dbgnn import BipartiteGraphOperator

DATA_ROOT = "data/processed/temporal"
CACHE_ROOT = "cache/dbgnn"
POSITIVE_CLASSES = (1, 3)
NEGATIVE_CLASSES = (2, 4)


class DBGNNData(Data):
    """Data object that batches first-order, higher-order and bipartite edge indices correctly."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_higher_order":
            return self.x_h.size(0)
        if key == "bipartite_edge_index":
            return torch.tensor([[self.x_h.size(0)], [self.num_nodes]])
        return super().__inc__(key, value, *args, **kwargs)


def _degree_features(edge_index, edge_weight, num_nodes):
    src, dst = edge_index[0], edge_index[1]
    w = edge_weight.float()
    out_deg = degree(src, num_nodes)
    in_deg = degree(dst, num_nodes)
    out_str = torch.zeros(num_nodes).scatter_add_(0, src, w)
    in_str = torch.zeros(num_nodes).scatter_add_(0, dst, w)
    return torch.log1p(torch.stack([out_deg, in_deg, out_str, in_str], dim=1))


def temporal_graph_to_dbgnn_data(g, delta, node_features="degree", node_ids=None):
    """Convert a pathpyG TemporalGraph into a batchable De Bruijn graph Data object."""
    model = pp.MultiOrderModel.from_temporal_graph(g, delta=delta, max_order=2)
    d = model.to_dbgnn_data(max_order=2)

    if node_features == "degree":
        x = _degree_features(d.edge_index, d.edge_weights, d.num_nodes)
        x_h = _degree_features(d.edge_index_higher_order, d.edge_weights_higher_order, d.x_h.size(0))
    elif node_features == "onehot":
        # align one-hot identities across graphs of a slice via the global node id order
        pos = {node_id: i for i, node_id in enumerate(node_ids)}
        perm = torch.tensor([pos[str(node_id)] for node_id in g.mapping.node_ids])
        x = torch.zeros(d.num_nodes, len(node_ids))
        x[torch.arange(d.num_nodes), perm] = 1.0
        seq = model.layers[2].data.node_sequence
        x_h = torch.cat([x[seq[:, 0]], x[seq[:, 1]]], dim=1)
    else:
        raise ValueError(f"unknown node_features: {node_features}")

    return DBGNNData(
        num_nodes=d.num_nodes,
        x=x,
        x_h=x_h,
        edge_index=d.edge_index,
        edge_index_higher_order=d.edge_index_higher_order,
        edge_weights=d.edge_weights.float(),
        edge_weights_higher_order=d.edge_weights_higher_order.float(),
        bipartite_edge_index=d.bipartite_edge_index,
    )


def get_dataset(data, slice, node_features="degree", root=DATA_ROOT, cache=True, limit=None):
    """Load a slice as DBGNN data, labelling classes 1/3 with y=1 and classes 2/4 with y=0."""
    cache_file = os.path.join(CACHE_ROOT, f"{data}_slice{slice}_{node_features}.pt")
    if cache and os.path.exists(cache_file):
        return torch.load(cache_file, weights_only=False)

    folder = os.path.join(root, data, f"slice{slice}")
    with open(os.path.join(folder, "meta.json")) as f:
        delta = json.load(f)["delta"]

    graphs = []
    for label, classes in ((1.0, POSITIVE_CLASSES), (0.0, NEGATIVE_CLASSES)):
        for c in classes:
            files = sorted(f for f in os.listdir(os.path.join(folder, f"class{c}")) if f.endswith(".pickle"))
            for file in files[:limit]:
                with open(os.path.join(folder, f"class{c}", file), "rb") as f:
                    graphs.append((pickle.load(f), label))

    node_ids = sorted({str(node_id) for g, _ in graphs for node_id in g.mapping.node_ids})

    dataset = []
    for g, label in graphs:
        d = temporal_graph_to_dbgnn_data(g, delta, node_features=node_features, node_ids=node_ids)
        d.y = torch.tensor([[label]])
        dataset.append(d)

    if cache:
        os.makedirs(CACHE_ROOT, exist_ok=True)
        torch.save(dataset, cache_file)
    return dataset


class DBGNNGraphClassifier(Module):
    """DBGNN with graph-level readout: first- and second-order message passing, bipartite fusion, pooling, MLP head."""

    def __init__(self, num_features, hidden_dims, dense_dims=(), out_channels=1, p_dropout=0.0, pooling="mean"):
        super().__init__()
        self.p_dropout = p_dropout
        self.pooling = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}[pooling]

        self.first_order_layers = ModuleList([GCNConv(num_features[0], hidden_dims[0])])
        self.higher_order_layers = ModuleList([GCNConv(num_features[1], hidden_dims[0])])
        for i in range(1, len(hidden_dims) - 1):
            self.first_order_layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))
            self.higher_order_layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))

        self.bipartite_layer = BipartiteGraphOperator(hidden_dims[-2], hidden_dims[-1])

        dims = [hidden_dims[-1], *dense_dims, out_channels]
        self.dense_layers = ModuleList([Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, data):
        x, x_h = data.x, data.x_h

        for layer in self.first_order_layers:
            x = F.dropout(x, p=self.p_dropout, training=self.training)
            x = F.elu(layer(x, data.edge_index, data.edge_weights))

        for layer in self.higher_order_layers:
            x_h = F.dropout(x_h, p=self.p_dropout, training=self.training)
            x_h = F.elu(layer(x_h, data.edge_index_higher_order, data.edge_weights_higher_order))

        # sizes passed positionally: the keyword names differ between pathpyG versions
        x = F.elu(self.bipartite_layer((x_h, x), data.bipartite_edge_index, x_h.size(0), x.size(0)))

        x = self.pooling(x, data.batch)
        for layer in self.dense_layers[:-1]:
            x = F.elu(layer(x))
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        return self.dense_layers[-1](x)


def train(loader, model, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.y.numel()
        correct += data.y.view(-1).eq(torch.sigmoid(out.view(-1)).round()).sum().item()
        n += data.y.numel()
    return total_loss / n, correct / n


@torch.no_grad()
def test(loader, model, device):
    model.eval()
    correct, n = 0, 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        correct += data.y.view(-1).eq(torch.sigmoid(out.view(-1)).round()).sum().item()
        n += data.y.numel()
    return correct / n


def run_experiment(
    data,
    slice,
    lr=0.01,
    layers=(32, 16, 8),
    lin_layers=(8, 4),
    weight_decay=0.0,
    epochs=50,
    batch_size=80,
    node_features="degree",
    p_dropout=0.0,
    device=None,
    verbose=True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(data, slice, node_features=node_features)

    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_features = (dataset[0].x.size(1), dataset[0].x_h.size(1))
    model = DBGNNGraphClassifier(
        num_features=num_features,
        hidden_dims=list(layers),
        dense_dims=list(lin_layers),
        p_dropout=p_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    losses = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        loss, train_acc = train(train_loader, model, optimizer, criterion, device)
        losses.append(loss)
        if verbose:
            print(
                f"epoch {epoch} loss {loss:.4f} train acc {train_acc:.3f} test acc {test(test_loader, model, device):.3f}",
                flush=True,
            )
    if verbose:
        print(f"{time.time() - start:.1f}s")
    return test(test_loader, model, device), losses
