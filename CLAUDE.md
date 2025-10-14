# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TopoLegal is a graph-based legal reference validation system for Ukrainian legal documents using PyTorch Geometric and frozen transformers. The system implements a vision-compliant architecture with frozen pre-trained components and trainable graph neural networks.

## Development Environment

### Package Management
- **Poetry** is used for dependency management
- Main config: `pyproject.toml` (simplified version)
- Full config: `pyproject.toml.full` (backup with all dependencies)

### Installation
```bash
# Install dependencies
poetry install

# Or use pip
pip install torch torch-geometric transformers google-cloud-bigquery
```

### Python Version
- Requires Python >=3.10,<4.0

## Common Commands

### Testing
```bash
# Run negative sampling tests
python test_negative_sampling.py
python test_negative_sampling_simple.py
python test_punkt_support.py

# Run consistency tests
python test_consistency_fix.py
python test_text_modification.py
```

### Running Processors

#### Entity Extraction
```bash
python -m legal_graph.processors.entities
```

#### Knowledge Graph Extraction
```bash
# From CSV
python -m legal_graph.processors.graph_extraction data.csv --max-docs 10

# From BigQuery
python processors/graph_extraction/bigquery_triplets.py \
  --table project.dataset.table \
  --batch 500
```

#### Negative Sampling (Hard Negatives)
```bash
# From CSV
python -m legal_graph.processors.negative_sampling sample-csv \
  input.csv output.csv \
  --strategy mixed \
  --probability 0.5

# From JSON
python -m legal_graph.processors.negative_sampling sample-json \
  input.json output.json \
  --strategy article

# From BigQuery
python -m legal_graph.processors.negative_sampling sample-bigquery \
  project.dataset.input_table \
  project.dataset.output_table \
  --strategy aggressive

# List available strategies
python -m legal_graph.processors.negative_sampling strategies
```

#### Data Corruption (Balance Module)
```bash
python -m legal_graph.processors.balance
```

### Training
```bash
# Run training notebooks
jupyter notebook models/notebooks/training.ipynb

# Or use demo scripts
python -m models bigquery-demo --table your-project.dataset.table
```

## High-Level Architecture

### Vision-Compliant Model Flow
```
INPUT → NER → SYNTHETIC → GNN → PROJECTOR → FUSION → OUTPUT
```

**Frozen Components (Red):** Pre-trained transformers
**Trainable Components (Teal):** NER, Synthetic Data Processor, GNN, Projector, Fusion Layer

### Module Structure

#### 1. Models (`src/legal_graph/models/`)
- **GraphCheck**: Main vision-compliant model implementing the full architecture
- **graph_dataset.py**: PyTorch Geometric dataset with BigQuery integration
- **gnn.py**: Graph neural network implementations
- **trainer.py**: Training utilities and loops

#### 2. Processors (`src/legal_graph/processors/`)
The processors module has a **unified interface via DocumentProcessor** that combines all processing capabilities:

**entities/**: spaCy-based NER for legal entities
- Extracts: ORG, PER, LOC, ROLE, INFO, CRIME, DTYPE, NUM

**graph_extraction/**: LLM-powered knowledge graph generation
- Extracts triplets: (source, relation, target, legal_reference)
- Supports BigQuery integration for large-scale processing
- Uses LangChain with OpenAI models

**balance/**: Data corruption for binary classification
- Class: `LegalReferenceCorruptor` extends `DataCorruptor` ABC
- Randomly replaces entire article numbers for simple negative samples

**negative_sampling/**: Advanced hard negative mining (NEW)
- Class: `NegativeSampler` extends `DataCorruptor` ABC
- Component-level modifications (article numbers, part numbers, or both)
- 8 strategies: article offset, part offset, combined, swap, random, mixed, conservative, aggressive
- Creates "near-miss" negatives that are harder to distinguish

**Key Difference:** Balance module does random full article replacement for binary classification. Negative sampling does fine-grained component-level modifications for hard negative mining in GNN training.

#### 3. Unified DocumentProcessor
```python
from legal_graph.processors import DocumentProcessor

processor = DocumentProcessor(
    enable_corruption=True,           # For balance module
    enable_negative_sampling=True,    # For hard negatives
    sampling_strategy='mixed'
)

# Extract entities
entities = processor.extract_entities(text)

# Extract knowledge graph
kg = await processor.extract_knowledge_graph(text)

# Create negative samples (hard negatives)
negative_samples = processor.create_negative_samples(dataset)

# Or corrupt data (balance module)
corrupted_data = processor.corrupt_data(dataset)
```

### Data Flow

1. **Input**: Ukrainian legal documents from BigQuery or CSV
2. **Entity Extraction**: NER identifies legal entities
3. **Graph Extraction**: LLM generates triplets (source, relation, target, legal_reference)
4. **Optional Negative Sampling**: Create hard negatives with fine-grained modifications
5. **Optional Data Corruption**: Simple binary classification balancing
6. **Graph Construction**: Convert to PyTorch Geometric Data objects
7. **Training**: GNN learns to validate legal references

## Ukrainian Legal Codes

The system supports these legal codes with specific article ranges:

- **КК України** (Criminal Code): Articles 1-446
- **КПК України** (Criminal Procedure Code): Articles 1-615
- **ЦК України** (Civil Code): Articles 1-1308
- **КоАП України** (Administrative Code): Articles 1-328
- **СК України** (Family Code): Articles 1-270
- **КЗпП України** (Labor Code): Articles 1-265

Legal reference pattern: `ч.X ст.Y КК України` (part X, article Y, Criminal Code of Ukraine)

## BigQuery Integration

### Authentication
```bash
# Method 1: gcloud CLI
gcloud auth application-default login

# Method 2: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

# Method 3: In code
# Use service account JSON file (lab-test-project-1-305710-30eed237388b.json)
```

### Table Schema Requirements
```sql
-- Required columns for dataset loading
doc_id STRING NOT NULL
text STRING
tags STRING              -- JSON string of entities
triplets STRING          -- JSON string of knowledge graph triplets
triplets_count INT64
```

### Example Dataset Creation
```python
from legal_graph.models import GraphDataset, create_dataloader

# From BigQuery
dataset = GraphDataset(
    table_id="project.dataset.table",
    max_nodes=100,
    max_edges=200
)

# Or from CSV
data = dataset.load_from_csv("legal_documents.csv")

# Create dataloader
dataloader = create_dataloader(dataset, batch_size=32, shuffle=True)
```

## Negative Sampling Strategies

When working with negative sampling (hard negatives for GNN training):

1. **article**: Modify article numbers by small offset (±1-5)
2. **part**: Modify part numbers by small offset (±1-2)
3. **combined**: Modify both article and part numbers
4. **swap**: Swap article/part numbers (creates invalid references)
5. **random**: Completely random but valid components
6. **mixed**: Balanced mix of all strategies (DEFAULT)
7. **conservative**: Small offsets only (subtle modifications)
8. **aggressive**: All strategies with maximum diversity

### Strategy Selection Guide
- Use **conservative** for subtle near-misses
- Use **aggressive** for maximum diversity in training
- Use **mixed** (default) for balanced negative samples
- Use **article** or **part** for specific component testing

## Important Notes

### File Modifications
When modifying legal documents:
- The negative_sampling module modifies text AND triplets AND entities consistently
- Balance module only modifies text
- Both track modifications with detailed statistics

### Known Issues
- See `POETRY_ISSUE_WORKAROUND.md` for Poetry lock file issues
- See `PUNKT_FORMAT_SUPPORT.md` for punkt-separated references (e.g., `ч.2 ст.185`)
- See `CONSISTENCY_FIX.md` for triplet/entity consistency fixes
- See `NUMPY_ARRAY_FIX.md` for NumPy array handling in triplets

### Testing Strategy
- Always test negative sampling with small datasets first
- Verify modifications appear in text, triplets, AND entities
- Check `is_valid` labels are correctly set
- Monitor statistics from `get_sampling_stats()`

### Processor Module Status
Check available modules:
```python
from legal_graph.processors import get_module_status, check_dependencies

status = get_module_status()
deps = check_dependencies()
```

## Project Structure
```
src/legal_graph/
├── models/              # GraphCheck model, datasets, training
├── processors/          # Document processing pipeline
│   ├── entities/        # NER extraction
│   ├── graph_extraction/  # Knowledge graph extraction
│   ├── balance/         # Simple data corruption
│   └── negative_sampling/  # Hard negative mining
└── tests/               # Unit tests
```

## Notebooks
- `1.0_data-load.ipynb`: Initial data loading
- `1.1_rtf_2_txt_conversion.ipynb`: RTF to text conversion
- `2.0_dataset_preparing.ipynb`: Dataset preparation
- `3.0_legal_gnn_dataset_usage.ipynb`: GNN dataset usage examples
- `test_model.ipynb`: Model testing

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

# Complete workflow
python example_dataset_reformatting.py
```

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

## Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```
