import torch

def extract_attention_weights(model, edge_index, drug_indices, top_k=5):
    """
    Extracts attention weights for edges connected to the given drug nodes.
    """
    model.eval()
    with torch.no_grad():
        _, att_weights = model.encode(edge_index, return_attention=True)
        
        # Unpack attention weights from the first GAT layer
        edge_index_layer, alpha = att_weights[0]
        alpha = alpha.mean(dim=-1) # average across heads
        
        relevant_edges = []
        for i in range(edge_index_layer.shape[1]):
            src = edge_index_layer[0, i].item()
            dst = edge_index_layer[1, i].item()
            if src in drug_indices or dst in drug_indices:
                relevant_edges.append((src, dst, alpha[i].item()))
                
        relevant_edges.sort(key=lambda x: x[2], reverse=True)
        return relevant_edges[:top_k]

def get_explanation(model, edge_index, drug_indices, drug_map, protein_map, side_effect_map, top_k=5):
    """
    Builds an explanation for the predicted interaction between drugs.
    """
    important_edges = extract_attention_weights(model, edge_index, drug_indices, top_k=top_k)
    
    inv_drug_map = {v: k for k, v in drug_map.items()}
    inv_protein_map = {v: k for k, v in protein_map.items()}
    
    important_proteins = []
    subgraph_edges = []
    
    for src, dst, weight in important_edges:
        subgraph_edges.append((src, dst, weight))
        
        for node in (src, dst):
            if node in inv_protein_map:
                protein_id = inv_protein_map[node]
                important_proteins.append({'id': protein_id, 'score': weight})
                
    explanation_text = f"The model's prediction is highly influenced by {len(important_proteins)} key protein interactions."
    if len(important_proteins) > 0:
        explanation_text += f" Top protein involved: {important_proteins[0]['id']}."
        
    return {
        'important_proteins': important_proteins,
        'important_pathways': [], # simplified
        'explanation_text': explanation_text,
        'subgraph_edges': subgraph_edges
    }

def explain_severity(predicted_side_effects, side_effect_map, threshold=0.5):
    """
    Explains the predicted severity based on the number of side effects above a threshold.
    """
    active_se_indices = (predicted_side_effects > threshold).nonzero(as_tuple=True)[1].tolist()
    
    # Map index back to names
    side_effect_names = list(side_effect_map.values())
    active_se_names = [side_effect_names[i] for i in active_se_indices if i < len(side_effect_names)]
    
    return active_se_names
