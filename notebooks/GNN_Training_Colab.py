# %% [markdown]
# # Explainable Multi-Drug Interaction GAT Model Training
# This notebook is designed to run in Google Colab (with a GPU runtime).
# It constructs a biomedical knowledge graph, trains a Graph Attention Network (GAT), 
# and saves the checkpoint for local interactive visualization in Streamlit.
#
# ## Developer & Owner: sm-daniyal
# ## Copyright (c) 2026 sm-daniyal. All rights reserved.

# %% [markdown]
# ## 1. Environment Setup
# Install required scientific computing packages for GNN training.

# %%
import subprocess
import sys

print("Installing PyTorch Geometric and required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch-geometric", "pandas", "numpy", "scikit-learn", "requests", "matplotlib", "networkx"])

import os
import urllib.request
import tarfile
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

print(f"PyTorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Data Download (Stanford SNAP Decagon)
# Downloads the networks from the Stanford Network Analysis Project.

# %%
DATA_DIR = "decagon_data"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "http://snap.stanford.edu/decagon/"
FILES = [
    "bio-decagon-ppi.tar.gz",
    "bio-decagon-targets.tar.gz",
    "bio-decagon-combo.tar.gz"
]

for fname in FILES:
    fpath = os.path.join(DATA_DIR, fname)
    csv_name = fname.replace(".tar.gz", ".csv")
    csv_path = os.path.join(DATA_DIR, csv_name)

    if not os.path.exists(csv_path):
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(BASE_URL + fname, fpath)
        print(f"Extracting {fname}...")
        with tarfile.open(fpath, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print(f"  -> {csv_name} ready.")
    else:
        print(f"{csv_name} already exists, skipping download.")

# %% [markdown]
# ## 3. Entrez ID to Gene Symbol Resolution
# Queries MyGene.info API in batches to map Entrez Gene IDs (like `5743`) to clinical Gene Symbols (like `PTGS2`/`COX-2`).

# %%
def resolve_gene_symbols(gene_ids):
    """Query MyGene.info API to batch-map Entrez IDs to Gene Symbols."""
    print(f"Resolving {len(gene_ids)} unique Gene IDs to Symbols...")
    url = "https://mygene.info/v3/query"
    gene_map = {}
    gene_ids_list = list(gene_ids)
    
    # Query in chunks of 1000 to prevent API timeouts
    for i in range(0, len(gene_ids_list), 1000):
        batch = gene_ids_list[i:i+1000]
        q = ",".join(str(x) for x in batch)
        payload = {"q": q, "scopes": "entrezgene", "fields": "symbol", "species": "human"}
        try:
            response = requests.post(url, data=payload, timeout=12)
            if response.status_code == 200:
                for item in response.json():
                    query_id = item.get("query")
                    symbol = item.get("symbol")
                    if query_id and symbol:
                        gene_map[str(query_id)] = symbol
        except Exception as e:
            print(f"API query warning in batch: {e}")
            
    print(f"Successfully resolved {len(gene_map)} gene symbols.")
    return gene_map

# %% [markdown]
# ## 4. Preprocessing and Graph Construction

# %%
# --- Configuration ---
TOP_K_SIDE_EFFECTS = 50  # Limit to top 50 side effects for GPU convergence

print("Loading CSV files...")
ppi_df = pd.read_csv(os.path.join(DATA_DIR, "bio-decagon-ppi.csv"))
targets_df = pd.read_csv(os.path.join(DATA_DIR, "bio-decagon-targets.csv"))
combo_df = pd.read_csv(os.path.join(DATA_DIR, "bio-decagon-combo.csv"))

# Filter to top-K most frequent side effects
se_counts = Counter(combo_df['Polypharmacy Side Effect'])
top_se = [se for se, _ in se_counts.most_common(TOP_K_SIDE_EFFECTS)]
combo_df_filtered = combo_df[combo_df['Polypharmacy Side Effect'].isin(top_se)].copy()

# Unique entities
drugs = sorted(set(targets_df['STITCH']).union(combo_df_filtered['STITCH 1']).union(combo_df_filtered['STITCH 2']))
proteins = sorted(set(ppi_df['Gene 1']).union(ppi_df['Gene 2']).union(targets_df['Gene']))

# Resolve gene IDs to symbols
raw_gene_ids = {str(p) for p in proteins}
gene_symbol_map = resolve_gene_symbols(raw_gene_ids)

num_drugs = len(drugs)
num_proteins = len(proteins)
num_nodes = num_drugs + num_proteins

drug_map = {d: i for i, d in enumerate(drugs)}
protein_map = {p: i + num_drugs for i, p in enumerate(proteins)}

# Human-readable labels map (resolved gene symbols)
protein_name_map = {}
for p in proteins:
    str_p = str(p)
    protein_name_map[protein_map[p]] = gene_symbol_map.get(str_p, f"Gene:{str_p}")

# Side effect label mapping
se_df = combo_df_filtered[['Polypharmacy Side Effect', 'Side Effect Name']].drop_duplicates()
side_effect_map = dict(zip(se_df['Polypharmacy Side Effect'], se_df['Side Effect Name']))
se_to_idx = {se: i for i, se in enumerate(top_se)}

print(f"Drugs in graph: {num_drugs}, Proteins in graph: {num_proteins}, Nodes in graph: {num_nodes}")

# %% [markdown]
# ## 5. Building Edge Tensors

# %%
# PPI edges
ppi_src = [protein_map[g] for g in ppi_df['Gene 1'] if g in protein_map]
ppi_dst = [protein_map[g] for g in ppi_df['Gene 2'] if g in protein_map]
ppi_edge_index = torch.tensor([ppi_src + ppi_dst, ppi_dst + ppi_src], dtype=torch.long)

# Drug-Target edges
dt_pairs = [(drug_map[d], protein_map[p])
            for d, p in zip(targets_df['STITCH'], targets_df['Gene'])
            if d in drug_map and p in protein_map]
dt_src = [p[0] for p in dt_pairs]
dt_dst = [p[1] for p in dt_pairs]
dt_edge_index = torch.tensor([dt_src + dt_dst, dt_dst + dt_src], dtype=torch.long)

# Drug-Drug edges (deduplicated for messaging)
dd_pairs_set = set()
for d1, d2 in zip(combo_df_filtered['STITCH 1'], combo_df_filtered['STITCH 2']):
    if d1 in drug_map and d2 in drug_map:
        dd_pairs_set.add((drug_map[d1], drug_map[d2]))
dd_src = [p[0] for p in dd_pairs_set]
dd_dst = [p[1] for p in dd_pairs_set]
dd_edge_index = torch.tensor([dd_src + dd_dst, dd_dst + dd_src], dtype=torch.long)

all_edge_index = torch.cat([ppi_edge_index, dt_edge_index, dd_edge_index], dim=1)

# Training Labels: Multi-hot targets
grouped = combo_df_filtered.groupby(['STITCH 1', 'STITCH 2'])['Polypharmacy Side Effect'].apply(list).reset_index()
pairs = []
labels = []
for _, row in grouped.iterrows():
    d1 = drug_map.get(row['STITCH 1'])
    d2 = drug_map.get(row['STITCH 2'])
    if d1 is not None and d2 is not None:
        pairs.append([d1, d2])
        label = np.zeros(TOP_K_SIDE_EFFECTS)
        for se in row['Polypharmacy Side Effect']:
            if se in se_to_idx:
                label[se_to_idx[se]] = 1
        labels.append(label)

pairs_tensor = torch.tensor(pairs, dtype=torch.long)
labels_tensor = torch.tensor(np.array(labels), dtype=torch.float32)

indices = np.arange(len(pairs))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_pairs, train_labels = pairs_tensor[train_idx], labels_tensor[train_idx]
test_pairs, test_labels = pairs_tensor[test_idx], labels_tensor[test_idx]

# %% [markdown]
# ## 6. GAT Model Architecture

# %%
class DrugInteractionGAT(nn.Module):
    def __init__(self, num_nodes, in_channels=128, hidden_channels=64,
                 out_channels=32, num_heads=4, num_side_effects=50, dropout=0.3):
        super().__init__()
        
        self.node_embedding = nn.Embedding(num_nodes, in_channels)
        
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout, concat=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.attention_pooling = nn.Linear(out_channels, 1)
        
        self.side_effect_predictor = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_side_effects)
        )
        
        self.severity_predictor = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def encode(self, edge_index, return_attention=False):
        x = self.node_embedding.weight
        if return_attention:
            x, att1 = self.gat1(x, edge_index, return_attention_weights=True)
            x = self.bn1(x)
            x = F.elu(x)
            x, att2 = self.gat2(x, edge_index, return_attention_weights=True)
            x = self.bn2(x)
            return x, (att1, att2)
        else:
            x = self.gat1(x, edge_index)
            x = self.bn1(x)
            x = F.elu(x)
            x = self.gat2(x, edge_index)
            x = self.bn2(x)
            return x
    
    def aggregate_drugs(self, drug_embeddings):
        weights = self.attention_pooling(drug_embeddings)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(drug_embeddings * weights, dim=1)
    
    def forward(self, edge_index, drug_pairs, return_attention=False):
        if return_attention:
            node_emb, att_weights = self.encode(edge_index, return_attention=True)
        else:
            node_emb = self.encode(edge_index)
        
        d1_emb = node_emb[drug_pairs[:, 0]]
        d2_emb = node_emb[drug_pairs[:, 1]]
        drug_stack = torch.stack([d1_emb, d2_emb], dim=1)
        combo = self.aggregate_drugs(drug_stack)
        
        se_probs = torch.sigmoid(self.side_effect_predictor(combo))
        sev_logits = self.severity_predictor(combo)
        
        if return_attention:
            return se_probs, sev_logits, att_weights
        return se_probs, sev_logits

# %% [markdown]
# ## 7. GNN Dataset Visualization (Sub-network Layout)
# Renders a visualization showing how GNN nodes are clustered (drugs connected to target protein symbols).

# %%
def visualize_subnetwork(edge_index, drug_map, protein_name_map, num_to_plot=8):
    """Draw a local subgraph of the GNN with color-coded drug and protein nodes."""
    print("Drawing GNN neighborhood structure...")
    G = nx.Graph()
    
    # Select a small subset of drug nodes
    target_drugs = list(drug_map.values())[:num_to_plot]
    drug_set = set(target_drugs)
    
    # Filter edge index for connections with these drugs
    edge_list = edge_index.cpu().numpy()
    nodes_to_add = set()
    edges_to_add = []
    
    for i in range(edge_list.shape[1]):
        src, dst = int(edge_list[0, i]), int(edge_list[1, i])
        if src in drug_set or dst in drug_set:
            nodes_to_add.add(src)
            nodes_to_add.add(dst)
            edges_to_add.append((src, dst))
            
    # Sub-sample to keep layout readable
    sub_nodes = list(nodes_to_add)[:50]
    G.add_nodes_from(sub_nodes)
    
    for src, dst in edges_to_add:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)
            
    # Node Colors
    colors = []
    for node in G.nodes:
        if node < len(drug_map):
            colors.append("#2563eb") # Drug nodes: Blue
        else:
            colors.append("#ea580c") # Protein nodes: Orange
            
    labels = {}
    for node in G.nodes:
        if node < len(drug_map):
            # Reverse lookup drug key name
            inv_map = {v: k for k, v in drug_map.items()}
            labels[node] = inv_map.get(node, f"Drug:{node}")[:8]
        else:
            labels[node] = protein_name_map.get(node, f"P-{node}")
            
    plt.figure(figsize=(10, 8), dpi=100)
    pos = nx.spring_layout(G, k=0.15, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=200, edgecolors="black")
    nx.draw_networkx_edges(G, pos, edge_color="#cbd5e1", width=1.0)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family="sans-serif")
    plt.title("Biomedical GNN Topology Map (Blue: Drugs, Orange: Protein Symbols)", fontsize=12)
    plt.axis("off")
    plt.show()

visualize_subnetwork(dt_edge_index, drug_map, protein_name_map)

# %% [markdown]
# ## 8. Fully Batched Training Loop
# Trains on the full dataset using PyTorch DataLoader mini-batch loops.

# %%
# Training Hyperparameters
EPOCHS = 15  # Loop 15 full epochs over all batches (highly accurate representation)
BATCH_SIZE = 256
LEARNING_RATE = 0.001

def get_severity_labels(se_labels):
    counts = se_labels.sum(dim=1)
    severity = torch.zeros(counts.shape[0], dtype=torch.long, device=se_labels.device)
    severity[(counts >= 3) & (counts <= 5)] = 1
    severity[counts >= 6] = 2
    return severity

# Build loader
train_dataset = TensorDataset(train_pairs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate Model
model = DrugInteractionGAT(num_nodes=num_nodes, num_side_effects=TOP_K_SIDE_EFFECTS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

all_edge_index_gpu = all_edge_index.to(device)

epoch_losses = []
eval_aucs = []
eval_aps = []

print("="*60)
print(f"Beginning batched model training for {EPOCHS} Epochs...")
print("="*60)

best_ap = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Negative sampling
        neg_x = torch.randint(0, num_drugs, size=batch_x.shape, device=device)
        neg_y = torch.zeros_like(batch_y)
        
        all_x = torch.cat([batch_x, neg_x], dim=0)
        all_y = torch.cat([batch_y, neg_y], dim=0)
        sev_y = get_severity_labels(all_y)
        
        # Forward Pass
        probs, sev_logits = model(all_edge_index_gpu, all_x)
        
        loss_se = bce_loss(probs, all_y)
        loss_sev = ce_loss(sev_logits, sev_y)
        loss = loss_se + 0.5 * loss_sev
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    avg_loss = total_loss / num_batches
    epoch_losses.append(avg_loss)
    
    # Evaluate epoch
    model.eval()
    with torch.no_grad():
        test_probs, _ = model(all_edge_index_gpu, test_pairs.to(device))
        y_true = test_labels.cpu().numpy()
        y_pred = test_probs.cpu().numpy()
        try:
            auc = roc_auc_score(y_true, y_pred, average='micro')
            ap = average_precision_score(y_true, y_pred, average='micro')
        except ValueError:
            auc, ap = 0.0, 0.0
            
    eval_aucs.append(auc)
    eval_aps.append(ap)
    
    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | ROC-AUC: {auc:.4f} | PR-AUC (AP): {ap:.4f}")
    
    if ap > best_ap:
        best_ap = ap
        # Save model checkpoint with mapping attributes
        torch.save({
            'model_state': model.state_dict(),
            'drug_map': drug_map,
            'protein_map': protein_map,
            'side_effect_map': side_effect_map,
            'drug_name_map': protein_name_map, # Incorporate Gene Symbol map directly!
            'num_nodes': num_nodes,
            'num_side_effects': TOP_K_SIDE_EFFECTS,
            'all_edge_index': all_edge_index.cpu(),
            'hyperparams': {'in_channels': 128, 'hidden_channels': 64, 'out_channels': 32}
        }, 'model_checkpoint.pt')
        print(f"  -> Model checkpoint updated (AP: {best_ap:.4f})")

print("="*60)
print(f"Training completed successfully. Best Average Precision: {best_ap:.4f}")
print("="*60)

# %% [markdown]
# ## 9. Plot Model Performance Curves
# Visualizes training metrics over all epochs.

# %%
plt.figure(figsize=(12, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), epoch_losses, label="Loss", color="black", linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GNN Training Loss")
plt.legend()

# Metrics curves
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), eval_aucs, label="ROC-AUC", color="#2563eb", linewidth=1.5)
plt.plot(range(1, EPOCHS + 1), eval_aps, label="PR-AUC", color="#ea580c", linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Model Convergence Metrics")
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Download Checkpoint File (Google Colab only)

# %%
try:
    from google.colab import files
    print("Triggering download of model_checkpoint.pt...")
    files.download('model_checkpoint.pt')
except ImportError:
    print("Download script running locally. File saved in model_checkpoint.pt")
