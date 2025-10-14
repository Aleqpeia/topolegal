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

## Dataset Reformatting

### Overview
The dataset reformatting tools extract JSON triplet arrays into columnar format and create stratified train/test/val splits.

### Quick Commands

```bash
# Analyze dataset statistics
python bigquery_dataset_reformatter.py analyze \
    --table project.dataset.source_table

# Reformat dataset (extract JSON to columns)
python bigquery_dataset_reformatter.py reformat \
    --source-table project.dataset.source_table \
    --dest-table project.dataset.reformatted_table

# Create stratified splits
python bigquery_dataset_reformatter.py split \
    --source-table project.dataset.reformatted_table \
    --train-table project.dataset.train \
    --test-table project.dataset.test \
    --val-table project.dataset.val \
    --stratify-by legal_reference

# Validate splits
python bigquery_dataset_reformatter.py validate \
    --train-table project.dataset.train \
    --test-table project.dataset.test \
    --val-table project.dataset.val


### Schema Transformation

**Before:** JSON array in `triplets` column
```json
{"source": "ОСОБА_1", "relation": "визнаний_винним", "target": "крадіжка",
 "legal_reference": "ч.2 ст.185 КК України", "confidence": 0.95}
```

**After:** Separate columns
- `legal_reference`, `source`, `target`, `relation`, `confidence`
- Plus metadata: `article_number`, `part_number`, `legal_code`

### Stratification Options
- `legal_reference` - By legal article (default, recommended)
- `relation` - By relation type
- `legal_code` - By legal code (КК, КПК, etc.)

### Files
- `bigquery_dataset_reformatter.py` - Python utility
- `bigquery_reformatting_queries.sql` - SQL queries
- `DATASET_REFORMATTING_GUIDE.md` - Complete guide
- `DATASET_REFORMATTING_QUICKREF.md` - Quick reference
- `example_dataset_reformatting.py` - Example workflow


## Architecture

The system implements a vision-compliant architecture with:
- 🔒 **Frozen Components**: Pre-trained transformers (red blocks)
- 🔄 **Trainable Components**: NER → Synthetic → GNN → Projector → Fusion (teal blocks)

**Data Flow**: `INPUT → NER → SYNTHETIC → GNN → PROJECTOR → FUSION → OUTPUT`



