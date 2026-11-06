"""Data loading and training loop for EGNN graph classification (RQ1-RQ3, Table 10)."""

import pickle
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .model import EGNN


def to_data(graph, label: int) -> Data:
    """PyG data object of a pickled (compressed) augmented event graph, with a one-hot node type as features."""
    data = graph.data
    data.x = F.one_hot(data.node_labels.long(), num_classes=2).float()
    data.y = torch.tensor([[label]]).float()
    if 'edge_weight' in data:
        data.edge_weight = torch.as_tensor(data.edge_weight).float()
    return data


def load_dataset(files: Iterable[Tuple[str, int]]) -> List[Data]:
    """Loads pickled augmented event graphs from `(path, label)` pairs."""
    dataset = []
    for path, label in files:
        with open(path, 'rb') as f:
            dataset.append(to_data(pickle.load(f), label))
    return dataset


def _forward(model, data):
    return model(x=data.x, edge_index=data.edge_index, batch=data.batch,
                 node_labels=data.node_labels.view(-1, 1), edge_weights=getattr(data, 'edge_weight', None))


def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _, _ = _forward(model, data)
        loss = criterion(out.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.y.numel()
        correct += data.y.eq(torch.sigmoid(out).round()).sum().item()
        n += data.y.numel()
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    correct, n = 0, 0
    for data in loader:
        data = data.to(device)
        out, _, _ = _forward(model, data)
        correct += data.y.eq(torch.sigmoid(out).round()).sum().item()
        n += data.y.numel()
    return correct / n


def fit(
    dataset: Sequence[Data],
    hidden_dims=(32, 16, 8),
    dense_dims=(8, 4),
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 150,
    batch_size: int = 200,
    test_batch_size: int = 20,
    pooling: str = 'add',
    pool_all: bool = False,
    consider_node_labels: bool = True,
    dirGNN: bool = True,
    train_fraction: float = 0.8,
    device=None,
    verbose: bool = True,
):
    """Trains an EGNN on a random train/test split of `dataset`.

    Returns the test accuracy after the last epoch and the per-epoch training losses and test accuracies.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_size = int(train_fraction * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    model = EGNN(in_channels=dataset[0].x.shape[1], out_channels=1, hidden_dims=list(hidden_dims),
                 dense_dims=list(dense_dims), pooling=pooling, pool_all=pool_all,
                 consider_node_labels=consider_node_labels, dirGNN=dirGNN).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    losses, accs = [], []
    for epoch in range(1, epochs + 1):
        loss, train_acc = train_epoch(train_loader, model, optimizer, criterion, device)
        acc = evaluate(test_loader, model, device)
        losses.append(loss)
        accs.append(acc)
        if verbose:
            print(f'epoch {epoch} loss {loss:.4f} train acc {train_acc:.3f} test acc {acc:.3f}', flush=True)

    return accs[-1], losses, accs
