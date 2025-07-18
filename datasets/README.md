# Legal Knowledge Graph Dataset

This directory contains the PyTorch dataset implementation for training Graph Neural Networks (GNNs) on legal document data.

## Files

- `legal_gnn_dataset.py` - Main dataset implementation
- `example_usage.py` - Simple example script for quick testing
- `README.md` - This file

## Dataset Features

The `LegalKnowledgeGraphDataset` provides:

- **Multiple Data Sources**: Load from BigQuery or CSV files
- **Automatic Vocabulary Building**: Creates vocabularies for entities, relations, and legal references
- **Feature Encoding**: Converts text data to numerical features suitable for GNNs
- **Legal Reference Integration**: Optional legal reference features for enhanced context
- **Custom Collate Function**: Proper batching for graph data
- **PyTorch Geometric Compatibility**: Easy integration with advanced GNN libraries

## Usage

### Basic Usage

```python
from legal_gnn_dataset import LegalKnowledgeGraphDataset, create_dataloader

# Initialize dataset from BigQuery
dataset = LegalKnowledgeGraphDataset(
    data_source='bigquery',
    table_id='your-project.your-dataset.your-table',
    max_nodes=50,
    max_edges=100,
    include_legal_references=True,
    node_features_dim=128,
    edge_features_dim=64
)

# Create DataLoader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)
```

### CSV Usage

```python
# Load from CSV file
dataset = LegalKnowledgeGraphDataset(
    data_source='csv',
    csv_file='your_data.csv',
    max_nodes=30,
    max_edges=50,
    include_legal_references=False
)
```

### Data Structure

The dataset expects the following data structure:

**BigQuery Table Schema:**
- `doc_id`: Document identifier
- `text`: Document text
- `tags`: JSON array of entities with `text` and `label` fields
- `triplets`: JSON array of triplets with `source`, `target`, `relation`, and optional `legal_reference` fields
- `triplets_count`: Number of triplets in the document

**CSV File Structure:**
- Same columns as BigQuery table
- JSON strings for `tags` and `triplets` columns

### Output Format

Each sample returns a dictionary with:
- `node_features`: Tensor of node features [max_nodes, node_features_dim]
- `edge_index`: Tensor of edge indices [2, num_edges]
- `edge_features`: Tensor of edge features [max_edges, edge_features_dim]
- `num_nodes`: Number of actual nodes
- `num_edges`: Number of actual edges
- `doc_id`: Document identifier
- `triplets_count`: Number of triplets

## Configuration Options

- `max_nodes`: Maximum number of nodes per graph (default: 100)
- `max_edges`: Maximum number of edges per graph (default: 200)
- `include_legal_references`: Whether to include legal reference features (default: True)
- `node_features_dim`: Dimension of node features (default: 128)
- `edge_features_dim`: Dimension of edge features (default: 64)
- `transform`: Optional transform function to apply to each sample

## Vocabulary Information

The dataset automatically builds vocabularies and provides information for model initialization:

```python
vocab_info = dataset.get_vocabulary_info()
print(vocab_info)
# Output: {
#   'node_vocab_size': 1234,
#   'edge_vocab_size': 567,
#   'legal_ref_vocab_size': 89,
#   'node_features_dim': 128,
#   'edge_features_dim': 64
# }
```

## Example Training

See `example_usage.py` for a complete training example, or use the Jupyter notebook `3.0_legal_gnn_dataset_usage.ipynb` for detailed analysis and examples.

## Dependencies

- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0 (optional, for advanced GNN layers)
- Google Cloud BigQuery (for BigQuery data source)
- Pandas (for CSV data source)
- NumPy

## Notes

- The dataset automatically handles padding and truncation to maintain consistent tensor shapes
- Unknown entities/relations are encoded as zero vectors
- Legal references are concatenated with edge features when enabled
- The collate function properly handles batching of variable-sized graphs 