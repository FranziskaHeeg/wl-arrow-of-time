"""GAT on the time-concatenated static graph (baseline following timewise isomorphism, Table 1 and Figure 7).

By default (`use_timestamps=False`), `GATConv` is constructed without `edge_dim`, in which case PyG ignores
`edge_attr`: the timestamps do not enter the computation. This is the configuration used for the results in the
submitted paper. With `use_timestamps=True`, the timestamps are min-max normalised to [0, 1] per graph and used as
one-dimensional edge features in the attention of both layers.
"""

import pickle
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import global_mean_pool


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int = 8, heads: int = 8, edge_dim: int = None):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=0, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=0, edge_dim=edge_dim)
        self.lin = torch.nn.Linear(hidden * heads, 1)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, edge_attr=data.edge_labels))
        x = F.elu(self.conv2(x, data.edge_index, edge_attr=data.edge_labels))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


def load_dataset(files: Iterable[Tuple[str, int]], use_timestamps: bool = False) -> List[Data]:
    """Loads pickled time-concatenated graphs from `(path, label)` pairs.

    Node features are a one-hot encoding of the nodes in a random order per graph. With `use_timestamps`, the
    edge labels are the timestamps min-max normalised to [0, 1] per graph.
    """
    dataset = []
    for path, label in files:
        with open(path, 'rb') as f:
            data = pickle.load(f).data
        data.y = torch.tensor([[label]]).float()
        data.x = torch.eye(data.num_nodes)[torch.randperm(data.num_nodes)]
        if use_timestamps:
            t = torch.tensor(data.edge_labels, dtype=torch.float64)
            t = (t - t.min()) / (t.max() - t.min()).clamp(min=1)
            data.edge_labels = t.float().view(-1, 1)
        else:
            data.edge_labels = torch.tensor(data.edge_labels).int().view(-1, 1)
        dataset.append(data)
    return dataset


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    correct, n = 0, 0
    for data in loader:
        data = data.to(device)
        correct += data.y.eq(torch.sigmoid(model(data)).round()).sum().item()
        n += data.y.numel()
    return correct / n


def fit(dataset, epochs=100, lr=0.001, weight_decay=0.0, batch_size=50, test_batch_size=20,
        train_fraction=0.8, use_timestamps=False, device=None, verbose=True):
    """Trains a GAT on a random train/test split; returns test and train accuracy after the last epoch."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_size = int(train_fraction * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    model = GAT(in_channels=dataset[0].x.shape[1], edge_dim=1 if use_timestamps else None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.binary_cross_entropy_with_logits(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.y.numel()
            correct += data.y.eq(torch.sigmoid(out).round()).sum().item()
            n += data.y.numel()
        test_acc = evaluate(test_loader, model, device)
        if verbose:
            print(f'epoch {epoch} loss {total_loss / n:.4f} train acc {correct / n:.3f} test acc {test_acc:.3f}', flush=True)

    return {'accuracy': evaluate(test_loader, model, device), 'train_accuracy': correct / n}
