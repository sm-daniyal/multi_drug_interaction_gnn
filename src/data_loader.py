import os
import urllib.request
import tarfile
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def download_decagon_data(data_dir):
    """
    Downloads and extracts the SNAP Decagon dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    base_url = "http://snap.stanford.edu/decagon/"
    files = [
        "bio-decagon-ppi.tar.gz",
        "bio-decagon-targets.tar.gz",
        "bio-decagon-combo.tar.gz"
    ]
    
    csv_paths = {}
    for file in files:
        file_path = os.path.join(data_dir, file)
        csv_name = file.replace(".tar.gz", ".csv")
        csv_path = os.path.join(data_dir, csv_name)
        
        if not os.path.exists(csv_path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, file_path)
            print(f"Extracting {file}...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
        else:
            print(f"{csv_name} already exists.")
            
        csv_paths[csv_name] = csv_path
        
    return csv_paths["bio-decagon-ppi.csv"], csv_paths["bio-decagon-targets.csv"], csv_paths["bio-decagon-combo.csv"]

def build_knowledge_graph(data_dir, top_k_side_effects=50):
    """
    Builds the knowledge graph from the CSV files.
    """
    ppi_path, targets_path, combo_path = download_decagon_data(data_dir)
    
    print("Loading data...")
    ppi_df = pd.read_csv(ppi_path)
    targets_df = pd.read_csv(targets_path)
    combo_df = pd.read_csv(combo_path)
    
    print("Filtering side effects...")
    se_counts = Counter(combo_df['Polypharmacy Side Effect'])
    top_se = [se for se, count in se_counts.most_common(top_k_side_effects)]
    combo_df = combo_df[combo_df['Polypharmacy Side Effect'].isin(top_se)]
    
    print("Building mappings...")
    # Unique entities
    drugs = set(targets_df['STITCH']).union(set(combo_df['STITCH 1'])).union(set(combo_df['STITCH 2']))
    proteins = set(ppi_df['Gene 1']).union(set(ppi_df['Gene 2'])).union(set(targets_df['Gene']))
    
    num_drugs = len(drugs)
    num_proteins = len(proteins)
    
    drug_map = {drug: i for i, drug in enumerate(drugs)}
    protein_map = {prot: i + num_drugs for i, prot in enumerate(proteins)}
    
    se_df = combo_df[['Polypharmacy Side Effect', 'Side Effect Name']].drop_duplicates()
    side_effect_map = {row['Polypharmacy Side Effect']: row['Side Effect Name'] for _, row in se_df.iterrows()}
    
    # Just a placeholder for drug names as they are not explicitly provided
    drug_name_map = {drug: str(drug) for drug in drugs}
    
    print("Creating edge indices...")
    # PPI edges
    ppi_src = [protein_map[g] for g in ppi_df['Gene 1']]
    ppi_dst = [protein_map[g] for g in ppi_df['Gene 2']]
    ppi_edge_index = torch.tensor([ppi_src + ppi_dst, ppi_dst + ppi_src], dtype=torch.long)
    
    # Drug-Target edges
    dt_src = [drug_map[d] for d in targets_df['STITCH']]
    dt_dst = [protein_map[p] for p in targets_df['Gene']]
    dt_edge_index = torch.tensor([dt_src + dt_dst, dt_dst + dt_src], dtype=torch.long)
    
    # Drug-Drug edges (simplified representation for the graph structure)
    dd_src = [drug_map[d] for d in combo_df['STITCH 1']]
    dd_dst = [drug_map[d] for d in combo_df['STITCH 2']]
    dd_edge_index = torch.tensor([dd_src + dd_dst, dd_dst + dd_src], dtype=torch.long)
    
    edge_indices = {
        'ppi': ppi_edge_index,
        'drug_target': dt_edge_index,
        'drug_drug': dd_edge_index
    }
    
    print("Initializing node features...")
    # Initial random embeddings
    total_nodes = num_drugs + num_proteins
    node_features = torch.randn(total_nodes, 128)
    
    return {
        'node_features': node_features,
        'edge_indices': edge_indices,
        'drug_map': drug_map,
        'protein_map': protein_map,
        'side_effect_map': side_effect_map,
        'drug_name_map': drug_name_map,
        'num_drugs': num_drugs,
        'num_proteins': num_proteins,
        'num_side_effects': len(top_se),
        'combo_df': combo_df,
        'top_se': top_se
    }

def create_train_test_split(combo_df, drug_map, top_se, test_ratio=0.2):
    """
    Creates train/test split for DDI prediction.
    Returns drug pairs and their side effect multi-hot labels.
    """
    print("Creating labels...")
    grouped = combo_df.groupby(['STITCH 1', 'STITCH 2'])['Polypharmacy Side Effect'].apply(list).reset_index()
    
    se_to_idx = {se: i for i, se in enumerate(top_se)}
    
    pairs = []
    labels = []
    for _, row in grouped.iterrows():
        d1 = drug_map[row['STITCH 1']]
        d2 = drug_map[row['STITCH 2']]
        pairs.append([d1, d2])
        
        label = np.zeros(len(top_se))
        for se in row['Polypharmacy Side Effect']:
            label[se_to_idx[se]] = 1
        labels.append(label)
        
    pairs = torch.tensor(pairs, dtype=torch.long)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)
    
    indices = np.arange(len(pairs))
    train_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=42)
    
    return (pairs[train_idx], labels[train_idx]), (pairs[test_idx], labels[test_idx])
