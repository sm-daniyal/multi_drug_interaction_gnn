import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class DrugInteractionGAT(nn.Module):
    """
    Two-layer GATv2 Architecture with Multi-Drug Regimen Attention Pooling ($N >= 2$)
    and residual skip connections for maximum gradient flow and expressivity.
    Department of Information Science & Engineering, JNNCE Shivamogga.
    """
    def __init__(self, num_nodes, in_channels=128, hidden_channels=64, out_channels=32,
                 num_heads=4, num_side_effects=50, dropout=0.25):
        super(DrugInteractionGAT, self).__init__()
        self.dropout = dropout

        self.node_embedding = nn.Embedding(num_nodes, in_channels)
        nn.init.xavier_uniform_(self.node_embedding.weight)

        # Layer 1: GATv2 with 4 Heads
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=0.1, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)

        # Layer 2: GATv2 with Single Head
        self.gat2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, dropout=0.1, concat=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual skip projection
        self.skip_proj = nn.Linear(in_channels, out_channels)

        # Paper Section III-E: Multi-Drug Regimen Attention Pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(out_channels, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Dual-Head Predictors
        # Combine pooled combination + symmetric difference vector for peak ROC-AUC
        repr_dim = out_channels * 3
        self.side_effect_predictor = nn.Sequential(
            nn.Linear(repr_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_side_effects)
        )
        self.severity_predictor = nn.Sequential(
            nn.Linear(repr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def encode(self, edge_index, return_attention=False):
        x0 = self.node_embedding.weight
        r = self.skip_proj(x0)

        if return_attention:
            x, att1 = self.gat1(x0, edge_index, return_attention_weights=True)
            x = F.elu(self.bn1(x))
            x, att2 = self.gat2(x, edge_index, return_attention_weights=True)
            x = self.bn2(x) + r
            return x, (att1, att2)
        else:
            x = self.gat1(x0, edge_index)
            x = F.elu(self.bn1(x))
            x = self.gat2(x, edge_index)
            x = self.bn2(x) + r
            return x

    def aggregate_drugs(self, drug_embeddings):
        # Backwards compatible alias
        return self.aggregate_regimen(drug_embeddings)

    def aggregate_regimen(self, drug_embeddings):
        # Attention Pooling for arbitrary N drugs (Eq. 6-7)
        weights = self.attention_pooling(drug_embeddings)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(drug_embeddings * weights, dim=1)

    def forward_features(self, node_emb, drug_pairs):
        d1 = node_emb[drug_pairs[:, 0]]
        d2 = node_emb[drug_pairs[:, 1]]
        stacked = torch.stack([d1, d2], dim=1)
        pooled = self.aggregate_regimen(stacked)
        # Symmetrical combined features
        pair_features = torch.cat([pooled, d1 * d2, (d1 - d2).abs()], dim=-1)
        return pair_features

    def forward_logits(self, edge_index, drug_pairs, node_emb=None):
        if node_emb is None:
            node_emb = self.encode(edge_index)
        feats = self.forward_features(node_emb, drug_pairs)
        return self.side_effect_predictor(feats), self.severity_predictor(feats)

    def decode_side_effects(self, features):
        logits = self.side_effect_predictor(features)
        return torch.sigmoid(logits)

    def decode_severity(self, features):
        logits = self.severity_predictor(features)
        return logits

    def forward(self, edge_index, drug_pairs, return_attention=False):
        if return_attention:
            node_emb, att_weights = self.encode(edge_index, return_attention=True)
        else:
            node_emb = self.encode(edge_index)
        se_logits, sev_logits = self.forward_logits(edge_index, drug_pairs, node_emb)
        se_probs = torch.sigmoid(se_logits)
        if return_attention:
            return se_probs, sev_logits, att_weights
        return se_probs, sev_logits
