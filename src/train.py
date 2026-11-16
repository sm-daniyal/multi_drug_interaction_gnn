import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from data_loader import build_knowledge_graph, create_train_test_split
from model import DrugInteractionGAT

def get_severity_labels(se_labels):
    counts = se_labels.sum(dim=1)
    severity = torch.zeros_like(counts, dtype=torch.long)
    severity[(counts >= 3) & (counts <= 5)] = 1
    severity[counts >= 6] = 2
    return severity

def sample_negatives(pairs, num_drugs):
    neg_pairs = torch.randint(0, num_drugs, size=pairs.shape)
    return neg_pairs

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score, f1_score
import json

def evaluate(model, test_edges, test_labels, edge_index, device, save_json=True):
    model.eval()
    with torch.no_grad():
        test_edges = test_edges.to(device)
        test_labels = test_labels.to(device)
        edge_index = edge_index.to(device)
        
        se_probs, severity_logits = model(edge_index, test_edges)
        
        y_true = test_labels.cpu().numpy()
        y_pred = se_probs.cpu().numpy()
        
        # Binary for F1 score
        y_pred_bin = (y_pred > 0.5).astype(int)
        
        try:
            auc = roc_auc_score(y_true, y_pred, average='macro')
            ap = average_precision_score(y_true, y_pred, average='macro')
        except ValueError:
            auc, ap = 0.0, 0.0
            
        f1_macro = f1_score(y_true, y_pred_bin, average='macro')
        
        # Severity evaluation
        severity_targets = get_severity_labels(test_labels).cpu().numpy()
        severity_preds = torch.argmax(severity_logits, dim=1).cpu().numpy()
        
        # precision, recall, f1 per class
        prec, rec, f1_sev, _ = precision_recall_fscore_support(severity_targets, severity_preds, labels=[0, 1, 2], zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(severity_targets, severity_preds, labels=[0, 1, 2])
        
        if save_json:
            metrics = {
                "ROC-AUC_macro": float(auc),
                "PR-AUC_macro": float(ap),
                "F1-Score_macro": float(f1_macro),
                "Severity_Classification": {
                    "Class_0": {"Precision": float(prec[0]), "Recall": float(rec[0]), "F1": float(f1_sev[0])},
                    "Class_1": {"Precision": float(prec[1]), "Recall": float(rec[1]), "F1": float(f1_sev[1])},
                    "Class_2": {"Precision": float(prec[2]), "Recall": float(rec[2]), "F1": float(f1_sev[2])}
                },
                "Confusion_Matrix": conf_matrix.tolist()
            }
            with open('data/evaluation_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
                
    return auc, ap

def train_model(data_dir='data/raw', top_k=50, epochs=100, lr=0.001, batch_size=512, device='cuda'):
    print(f"Using device: {device}")
    
    graph_data = build_knowledge_graph(data_dir, top_k_side_effects=top_k)
    num_nodes = graph_data['num_drugs'] + graph_data['num_proteins']
    
    # Combine edges for encoding
    all_edges = torch.cat(list(graph_data['edge_indices'].values()), dim=1).to(device)
    
    (train_pairs, train_labels), (test_pairs, test_labels) = create_train_test_split(
        graph_data['combo_df'], graph_data['drug_map'], graph_data['top_se']
    )
    
    model = DrugInteractionGAT(num_nodes=num_nodes, num_side_effects=top_k).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    best_ap = 0.0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Batching (simplified: full batch or random sample if too large)
        idx = torch.randperm(len(train_pairs))[:batch_size]
        batch_pairs = train_pairs[idx].to(device)
        batch_labels = train_labels[idx].to(device)
        
        # Negative sampling
        neg_pairs = sample_negatives(batch_pairs, graph_data['num_drugs']).to(device)
        neg_labels = torch.zeros_like(batch_labels).to(device)
        
        inputs = torch.cat([batch_pairs, neg_pairs], dim=0)
        targets = torch.cat([batch_labels, neg_labels], dim=0)
        severity_targets = get_severity_labels(targets)
        
        se_probs, severity_logits = model(all_edges, inputs)
        
        loss_se = bce_loss(se_probs, targets)
        loss_sev = ce_loss(severity_logits, severity_targets)
        
        loss = loss_se + 0.5 * loss_sev
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            auc, ap = evaluate(model, test_pairs, test_labels, all_edges, device)
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")
            
            if ap > best_ap:
                best_ap = ap
                save_dict = {
                    'model_state': model.state_dict(),
                    'drug_map': graph_data['drug_map'],
                    'protein_map': graph_data['protein_map'],
                    'side_effect_map': graph_data['side_effect_map'],
                    'drug_name_map': graph_data['drug_name_map'],
                    'num_nodes': num_nodes,
                    'num_side_effects': top_k,
                    'hyperparams': {'in_channels': 128, 'hidden_channels': 64, 'out_channels': 32}
                }
                os.makedirs('data', exist_ok=True)
                torch.save(save_dict, 'data/model_checkpoint.pt')
                print("Saved new best model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    train_model(args.data_dir, args.top_k, args.epochs, args.lr, args.batch_size, args.device)
