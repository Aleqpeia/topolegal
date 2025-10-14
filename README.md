# TopoLegal - Legal Knowledge Graph System

A graph-based legal reference validation system for Ukrainian legal documents using PyTorch Geometric and frozen transformers.

## Features

- 📊 **BigQuery Integration**: Direct connection to Google Cloud BigQuery for large-scale legal datasets
- 🇺🇦 **Ukrainian Legal Codes**: Support for КК України, КПК України, ЦК України, КоАП України
- 📈 **PyTorch Geometric**: Graph neural networks for legal knowledge representation
- 🚀 **Scalable Training**: Comprehensive training pipeline with monitoring

## Quick Start

### 1. Installation
```bash
pip install torch torch-geometric transformers google-cloud-bigquery
```

### 2. Basic Usage

```python
from src.legal_graph.models import GraphDataset, create_dataloader

# Create dataset (falls back to sample data if BigQuery not available)
dataset = GraphDataset(
    table_id="your-project.dataset.legal_documents",
    max_nodes=100,
    max_edges=200
)

# Create dataloader for training
dataloader = create_dataloader(dataset, batch_size=32, shuffle=True)
```

### 3. BigQuery Setup (Optional)
To use real BigQuery data:

```bash
# Install BigQuery client
pip install google-cloud-bigquery

# Authenticate with Google Cloud
gcloud auth application-default login

# Or set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

### 4. Training
```bash
# Run training notebook
jupyter notebook models/notebooks/training.ipynb

# Or use demo scripts
python -m models bigquery-demo --table your-project.dataset.table
```

## Architecture

The system implements a vision-compliant architecture with:
- 🔒 **Frozen Components**: Pre-trained transformers (red blocks)
- 🔄 **Trainable Components**: NER → Synthetic → GNN → Projector → Fusion (teal blocks)

**Data Flow**: `INPUT → NER → SYNTHETIC → GNN → PROJECTOR → FUSION → OUTPUT`



