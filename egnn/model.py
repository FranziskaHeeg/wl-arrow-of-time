"""Message passing on augmented event graphs (Section 5 of the paper).

The model applies a directed GNN (Dir-GNN, Rossi et al. 2023) to the augmented event graph,
pools the embeddings of one node type and classifies the pooled graph embedding with an MLP.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList
from torch_geometric.nn.conv import GraphConv, MessagePassing
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool


class DirGNN(Module):
    """Dir-GNN layer: separate aggregation over incoming and outgoing neighbours.

    Same as ``torch_geometric.nn.conv.DirGNNConv`` but with two independent convolutions and a
    small-gain initialisation.
    """

    def __init__(self, conv_in: MessagePassing, conv_out: MessagePassing, alpha: float = 0.5, root_weight: bool = True):
        super().__init__()

        self.alpha = alpha
        self.root_weight = root_weight

        self.conv_in = conv_in
        self.conv_out = conv_out

        if hasattr(conv_in, 'add_self_loops'):
            self.conv_in.add_self_loops = False
            self.conv_out.add_self_loops = False
        if hasattr(conv_out, 'root_weight'):
            self.conv_in.root_weight = False
            self.conv_out.root_weight = False

        if root_weight:
            self.lin = Linear(conv_in.in_channels, conv_out.out_channels)
        else:
            self.lin = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.conv_in.lin_rel.weight, gain=0.01)
        torch.nn.init.xavier_normal_(self.conv_in.lin_root.weight, gain=0.01)

        if self.lin is not None:
            self.lin.reset_parameters()
            torch.nn.init.xavier_normal_(self.lin.weight, gain=0.01)
            torch.nn.init.zeros_(self.lin.bias)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        x_in = self.conv_in(x, edge_index, **kwargs)
        x_out = self.conv_out(x, edge_index.flip([0]), **kwargs)

        out = self.alpha * x_out + (1 - self.alpha) * x_in

        if self.root_weight:
            out = out + self.lin(x)

        return out


class EGNN(Module):
    """Event graph neural network for temporal graph classification.

    Args:
        in_channels: dimension of the input node features (2 for a one-hot node type).
        out_channels: number of outputs (1 for binary classification with a sigmoid).
        hidden_dims: widths of the message passing layers, e.g. ``[32, 16, 8]``.
        dense_dims: widths of the classification MLP, starting with ``hidden_dims[-1]``, e.g. ``[8, 4]``.
        pooling: graph readout, one of ``'add'``, ``'mean'``, ``'max'``.
        pool_all: pool over all nodes instead of only the first-order nodes.
        consider_node_labels: whether node types are available to restrict the readout.
        dirGNN: use Dir-GNN (messages along and against edge directions). If ``False``, messages
            are only passed along edge directions (ablation in Table 10).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: Optional[List[int]] = None,
        dense_dims: Optional[List[int]] = None,
        pooling: str = 'add',
        pool_all: bool = False,
        consider_node_labels: bool = True,
        dirGNN: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.dense_dims = dense_dims
        self.out_channels = out_channels
        self.pooling = pooling
        self.pool_all = pool_all
        self.node_labels = consider_node_labels

        dims = [in_channels, *hidden_dims]
        self.convs = ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            if dirGNN:
                self.convs.append(DirGNN(GraphConv(d_in, d_out), GraphConv(d_in, d_out)))
            else:
                self.convs.append(GraphConv(d_in, d_out))

        self.dense = ModuleList()
        for d_in, d_out in zip(dense_dims[:-1], dense_dims[1:]):
            self.dense.append(Linear(d_in, d_out))
        self.dense.append(Linear(dense_dims[-1], out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for lin in self.dense:
            torch.nn.init.xavier_normal_(lin.weight, gain=0.01)
            torch.nn.init.zeros_(lin.bias)

    def forward(self, x, edge_index, node_labels, batch, edge_weights=None):
        """Returns the logits, the node embeddings and the graph embeddings."""
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_weight=edge_weights))

        if not self.pool_all and self.node_labels:
            # node label 0 marks event nodes (timestamped edges), 1 marks first-order nodes;
            # only the embeddings of first-order nodes enter the readout
            x = x.masked_fill(node_labels.view(-1, 1) == 0, 0)

        embedding = x

        if self.pooling == 'add':
            x = global_add_pool(x, batch=batch)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch=batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch=batch)
        else:
            raise ValueError(f'unknown pooling: {self.pooling}')

        graph_embedding = x
        for i, lin in enumerate(self.dense):
            x = lin(x)
            if i < len(self.dense) - 1:
                x = F.elu(x)

        return x, embedding, graph_embedding
