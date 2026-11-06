"""Graph classification from temporal node embeddings of a TGNN (TGAT, TGN, CAW, PINT rows of Table 1).

The TGNNs are trained self-supervised for link prediction on each temporal graph and save the embeddings
of the source node of every event as ``{embeddings_dir}/{MODEL}_{id}_embedding.npy`` (with the source
node ids in ``..._src_nodes.npy``). This script turns them into one feature matrix per graph and trains
an MLP to separate classes 1/3 from classes 2/4 of one slice.

Node embeddings (``--mode``): ``all`` temporal embeddings, the ``last`` one per node, or the ``mean`` per
node. Graph readout (``--pooling``): ``mean`` or ``max`` over the rows, or ``all`` to concatenate them.

    python baselines/tgnn/classify_embeddings.py --embeddings_dir baselines/tgnn/tgat/embeddings \\
        --name ants_1_1_slice0 --mode mean --pooling max --lr 0.001 --layer_sizes 128
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split

from egnn.datasets import RESULTS_DIR


def get_embedding(path: Path, mode: str = 'last'):
    embeddings = np.load(path)
    nodes = np.load(path.with_name(path.name.replace('embedding', 'src_nodes')))

    if mode == 'all':
        return nodes, embeddings

    emb_dict = defaultdict(list)
    for node, emb in zip(nodes, embeddings):
        emb_dict[node].append(emb)

    if mode == 'last':
        final = [embs[-1] for embs in emb_dict.values()]
    elif mode == 'mean':
        final = [np.mean(embs, axis=0) for embs in emb_dict.values()]
    else:
        raise ValueError(f'invalid mode: {mode}')
    return np.array(list(emb_dict)), np.array(final)


class GraphSample:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_dataset(embeddings_dir: str, name: str, mode: str = 'last'):
    """Embedding matrices of all graphs of slice `name`: label 0 for classes 1/3, label 1 for classes 2/4."""
    dataset = []
    for file in sorted(os.listdir(embeddings_dir)):
        if name not in file or not file.endswith('embedding.npy'):
            continue
        if '_c1_' in file or '_c3_' in file:
            label = 0
        elif '_c2_' in file or '_c4_' in file:
            label = 1
        else:
            continue
        _, emb = get_embedding(Path(embeddings_dir) / file, mode)
        dataset.append(GraphSample(torch.Tensor(emb), torch.Tensor([label])))
    return dataset


class GraphMLP(nn.Module):
    def __init__(self, d=64, hidden=(128,), num_classes=2, pooling='max', num_nodes=None):
        super().__init__()
        self.pooling = pooling

        in_dim = d if pooling in ('mean', 'max') else d * num_nodes
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.pooling == 'mean':
            x = torch.mean(x, dim=0)
        elif self.pooling == 'max':
            x = torch.max(x, dim=0)[0]
        else:
            x = torch.cat(list(x), dim=0)
        return self.net(x)


def accuracy(model, samples, device):
    correct = 0
    with torch.no_grad():
        for data in samples:
            predicted = torch.argmax(model(data.x.to(device)).cpu())
            correct += (predicted == data.y.long().squeeze()).sum().item()
    return correct / len(samples)


def run_experiment(dataset, epochs=100, lr=0.001, layer_sizes=(128,), pooling='max', train_fraction=0.7, verbose=True):
    train_size = int(train_fraction * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphMLP(d=dataset[0].x.shape[1], hidden=layer_sizes, num_classes=2, pooling=pooling,
                     num_nodes=dataset[0].x.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_dataset:
            optimizer.zero_grad()
            out = model(data.x.to(device)).cpu()
            y = data.y.item()
            loss = criterion(out.squeeze(), torch.Tensor([1 - y, y]))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        test_accuracy = accuracy(model, test_dataset, device)
        train_accuracy = accuracy(model, train_dataset, device)
        if verbose:
            print(f'epoch {epoch + 1}/{epochs} test acc {test_accuracy:.2f} train acc {train_accuracy:.2f} '
                  f'loss {total_loss / len(train_dataset):.4f}', flush=True)

    return {'accuracy': test_accuracy, 'train_accuracy': train_accuracy}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--embeddings_dir', required=True)
    parser.add_argument('--name', required=True, help='slice, e.g. ants_1_1_slice0')
    parser.add_argument('--mode', default='last', choices=['last', 'mean', 'all'])
    parser.add_argument('--pooling', default='max', choices=['mean', 'max', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=[128])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_name', default=None, help='name of the results subdirectory (default: embeddings_dir parent)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = get_dataset(args.embeddings_dir, args.name, args.mode)
    result = run_experiment(dataset, epochs=args.epochs, lr=args.lr, layer_sizes=args.layer_sizes, pooling=args.pooling)

    model_name = args.model_name or Path(args.embeddings_dir).resolve().parent.name
    params = {k: getattr(args, k) for k in ('mode', 'pooling', 'epochs', 'lr', 'layer_sizes', 'seed')}
    out_dir = Path(args.out or RESULTS_DIR / 'empirical' / model_name) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    file_id = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    (out_dir / f'exp_{file_id}.json').write_text(json.dumps({'name': args.name, 'params': params, 'metrics': result}, indent=2))


if __name__ == '__main__':
    main()
