# Multi-Drug Interaction Prediction using Graph Attention Networks

A Graph Neural Network (GNN)-based system that predicts adverse drug-drug interactions (DDIs) and their severity using a biomedical knowledge graph constructed from the Stanford SNAP Decagon dataset.

## Features

- **Brand Name Support**: Enter drug names like "Dolo 650", "Combiflam", or "Advil" — the system automatically resolves them to their active generic ingredients.
- **Multi-Drug Prediction**: Supports 2+ drugs simultaneously (not just pairwise), using attention-based aggregation.
- **Severity Classification**: Predicts interaction severity as Minor, Moderate, or Major.
- **Explainable AI**: Uses GAT attention weights to explain why interactions are predicted.
- **Interactive Dashboard**: Streamlit-based UI with risk gauges, side effect tables, and network visualizations.

## Quick Start

### Step 1: Train the Model (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/GNN_Training_Colab.py` or paste its contents into Colab cells
3. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
4. Run all cells — training takes ~10-15 minutes
5. Download the generated `model_checkpoint.pt` file
6. Place it in the `data/` directory of this project

### Step 2: Run the Dashboard (Local)

```powershell
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
```

> **Note**: The app works in Demo Mode even without the trained model checkpoint. Demo mode generates mock predictions so you can test the full UI.

## Project Structure

```
multi_drug_interaction_gnn/
├── data/                        # Data files
│   ├── brand_mappings.json      # 100+ brand-to-generic drug mappings
│   └── model_checkpoint.pt      # Trained model weights (from Colab)
├── src/                         # Core ML pipeline
│   ├── data_loader.py           # Downloads SNAP datasets, builds KG
│   ├── model.py                 # GAT model architecture
│   ├── train.py                 # Training loop with metrics
│   ├── explainer.py             # GAT attention extraction & XAI
│   ├── resolver.py              # Brand name → generic → STITCH ID
│   └── inference.py             # CPU inference with demo fallback
├── app/
│   └── app.py                   # Streamlit dashboard
├── notebooks/
│   └── GNN_Training_Colab.py    # Google Colab training script
├── requirements.txt
└── README.md
```

## Datasets

All datasets are downloaded automatically from [Stanford SNAP](http://snap.stanford.edu/decagon/):

| File | Description | Size |
|------|-------------|------|
| bio-decagon-ppi.csv | Protein-protein interactions | ~720K edges |
| bio-decagon-targets.csv | Drug-target associations | ~18K edges |
| bio-decagon-combo.csv | DDI side effects (964 types) | ~4.6M entries |

## Architecture

```
User Input (Brand Names) → Resolver (RxNorm API) → Generic Ingredients
    → STITCH IDs → GAT Encoder → Attention Pooling → Decoder
    → Side Effect Probabilities + Severity Classification
    → GAT Attention Weights → Explainability
```

## References

1. Zitnik, M., Agrawal, M., & Leskovec, J. (2018). Modeling polypharmacy side effects with graph convolutional networks. *Bioinformatics*, 34(13), i457–i466.
2. HODDI: Higher-Order Drug-Drug Interaction Dataset (TIML-Group/HODDI)
3. KnowDDI: Accurate and Interpretable DDI Prediction (LARS-research/KnowDDI)
