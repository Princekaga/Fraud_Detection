"""GNN architectures for illicit transaction detection on the Elliptic dataset."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN(nn.Module):
    """2-layer Graph Convolutional Network (Kipf & Welling, 2017)."""
    def __init__(self, in_dim, hidden=64, out_dim=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        self.embeddings = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE (Hamilton et al., 2017) - inductive, mean aggregator."""
    def __init__(self, in_dim, hidden=64, out_dim=2, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        self.embeddings = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class GAT(nn.Module):
    """2-layer Graph Attention Network (Velickovic et al., 2018)."""
    def __init__(self, in_dim, hidden=64, out_dim=2, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.3)
        self.head = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        self.embeddings = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017): down-weights easy examples so training
    focuses on the hard, rare illicit class."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


MODELS = {"GCN": GCN, "GraphSAGE": GraphSAGE, "GAT": GAT}
