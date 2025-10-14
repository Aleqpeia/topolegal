# Negative Sampling Module

Advanced negative sampling for Ukrainian legal documents by intelligently modifying legal reference components (article numbers and part numbers).

## Overview

This module provides fine-grained control over creating negative samples for training graph neural networks on legal document validation tasks. Unlike the existing `balance` module which replaces entire article numbers randomly, this module:

- **Component-level modifications**: Modify article numbers (`ст.#`) or part numbers (`ч.#`) independently
- **Multiple strategies**: Choose from offset-based, swap-based, random, or mixed strategies
- **Syntactic validity control**: Create both valid and invalid legal references
- **Hard negative mining**: Generate "near-miss" samples that are more challenging for models to distinguish

## Key Differences from Balance Module

| Feature | Balance Module | Negative Sampling Module |
|---------|---------------|-------------------------|
| Granularity | Full article replacement | Component-level (article, part, or both) |
| Strategy | Random article from range | Controlled offsets, swaps, or random |
| Use Case | Binary classification | Hard negative mining for GNN training |
| Validity | Always syntactically valid | Can create invalid formats |
| Example | `ст.537` → `ст.191` | `ч.2 ст.185` → `ч.3 ст.187` |

## Installation

The module is part of the `legal_graph.processors` package. No additional dependencies required beyond the base package.

## Quick Start

### Python API

```python
from legal_graph.processors.negative_sampling import NegativeSampler, create_negative_samples

# Method 1: Using the convenience function
dataset = [
    {
        'doc_id': '001',
        'text': 'Суд визнав винним згідно з ч.2 ст.185 КК України',
        'entities': [...],
        'triplets': [...]
    }
]

negative_samples = create_negative_samples(
    dataset,
    strategy='article',  # or 'part', 'mixed', etc.
    sampling_probability=0.5
)

# Method 2: Using the NegativeSampler class
sampler = NegativeSampler(
    sampling_probability=0.5,
    strategy='mixed'
)

negative_samples = sampler.create_negative_samples(dataset)

# Check results
for sample in negative_samples:
    print(f"Valid: {sample['is_valid']}, Text: {sample['text']}")
```

### Command Line Interface

```bash
# Create negative samples from CSV
python -m legal_graph.processors.negative_sampling sample-csv \
    input.csv output.csv \
    --strategy mixed \
    --probability 0.5

# Create negative samples from JSON
python -m legal_graph.processors.negative_sampling sample-json \
    input.json output.json \
    --strategy article \
    --probability 0.3

# Create negative samples from BigQuery
python -m legal_graph.processors.negative_sampling sample-bigquery \
    project.dataset.input_table \
    project.dataset.output_table \
    --strategy aggressive \
    --probability 0.7 \
    --batch-size 100

# Show available strategies
python -m legal_graph.processors.negative_sampling strategies
```

## Sampling Strategies

### 1. ArticleOffsetStrategy (`article`)
Modifies article numbers by a small offset (±1-5).

**Example:**
```
Original: ч.2 ст.185 КК України
Modified: ч.2 ст.187 КК України  (article +2)
```

**Use case:** Create "near-miss" negatives that are syntactically valid but semantically incorrect.

### 2. PartOffsetStrategy (`part`)
Modifies part numbers by a small offset (±1-2).

**Example:**
```
Original: ч.2 ст.185 КК України
Modified: ч.3 ст.185 КК України  (part +1)
```

**Use case:** Different parts of the same article often have different legal meanings.

### 3. CombinedOffsetStrategy (`combined`)
Modifies both article and part numbers.

**Example:**
```
Original: ч.2 ст.185 КК України
Modified: ч.3 ст.187 КК України  (both modified)
```

**Use case:** More diverse negative samples.

### 4. IndexSwapStrategy (`swap`)
Swaps article and part numbers, creating syntactically invalid references.

**Example:**
```
Original: ч.2 ст.185 КК України
Modified: ч.185 ст.2 КК України  (swapped - INVALID)
```

**Use case:** Test if model can detect structurally invalid references.

### 5. RandomComponentStrategy (`random`)
Replaces with completely random but valid components.

**Example:**
```
Original: ч.2 ст.185 КК України
Modified: ч.7 ст.341 КК України  (random valid)
```

**Use case:** Maximum diversity in negative samples.

### 6. MixedStrategy (`mixed`) - Default
Combines multiple strategies with configurable weights.

**Example:**
```
Uses a mix of all above strategies with balanced probabilities
```

**Use case:** General-purpose negative sampling with good diversity.

### 7. Conservative Strategy (`conservative`)
Small offsets only, focused on subtle modifications.

**Configuration:**
- 70% ArticleOffset (±1-3)
- 30% PartOffset (±1)

### 8. Aggressive Strategy (`aggressive`)
All strategies with equal weights, maximum diversity.

**Configuration:**
- 30% ArticleOffset (±1-10)
- 20% PartOffset (±1-5)
- 20% Combined
- 10% Swap
- 20% Random

## API Reference

### NegativeSampler

```python
NegativeSampler(
    sampling_probability: float = 0.5,
    strategy: Union[str, SamplingStrategy] = 'mixed',
    article_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    max_part: int = 10,
    preserve_format: bool = True
)
```

**Parameters:**
- `sampling_probability`: Probability of sampling each document (0.0-1.0)
- `strategy`: Strategy name or custom SamplingStrategy instance
- `article_ranges`: Custom article number ranges per legal code
- `max_part`: Maximum part number (default 10)
- `preserve_format`: Preserve original formatting (spaces, etc.)

**Methods:**
- `create_negative_samples(dataset, label_column='is_valid')`: Create negative samples
- `get_sampling_stats()`: Get detailed statistics

### create_negative_samples (Function)

```python
create_negative_samples(
    dataset: List[Dict],
    strategy: Union[str, SamplingStrategy] = 'mixed',
    sampling_probability: float = 0.5,
    label_column: str = 'is_valid'
) -> List[Dict]
```

Convenience function for one-off negative sampling.

## Supported Legal Codes

The module supports all major Ukrainian legal codes with appropriate article number ranges:

- **КК України** (Criminal Code): Articles 1-446
- **КПК України** (Criminal Procedure Code): Articles 1-615
- **ЦК України** (Civil Code): Articles 1-1308
- **КоАП України** (Administrative Code): Articles 1-328
- **СК України** (Family Code): Articles 1-270
- **КЗпП України** (Labor Code): Articles 1-265

## Examples

### Example 1: Basic Usage with CSV

```python
import pandas as pd
from legal_graph.processors.negative_sampling import NegativeSampler

# Load your data
df = pd.read_csv('legal_documents.csv')

# Convert to required format
dataset = []
for _, row in df.iterrows():
    dataset.append({
        'doc_id': row.doc_id,
        'text': row.text,
        'entities': json.loads(row.tags),
        'triplets': json.loads(row.triplets)
    })

# Create sampler
sampler = NegativeSampler(
    sampling_probability=0.5,
    strategy='article'
)

# Generate negative samples
result = sampler.create_negative_samples(dataset)

# Save results
output_df = pd.DataFrame(result)
output_df.to_csv('negative_samples.csv', index=False)

# View statistics
stats = sampler.get_sampling_stats()
print(f"Created {stats['corrupted']} negative samples")
print(f"Modifications by type: {stats['modifications_by_type']}")
```

### Example 2: Custom Strategy

```python
from legal_graph.processors.negative_sampling import (
    NegativeSampler,
    ArticleOffsetStrategy,
    PartOffsetStrategy,
    MixedStrategy
)

# Create custom strategy
custom_strategy = MixedStrategy(
    strategies=[
        ArticleOffsetStrategy(offset_range=(1, 3)),
        PartOffsetStrategy(offset_range=(1, 1))
    ],
    weights=[0.8, 0.2]  # 80% article, 20% part
)

# Use custom strategy
sampler = NegativeSampler(
    sampling_probability=0.7,
    strategy=custom_strategy
)

negative_samples = sampler.create_negative_samples(dataset)
```

### Example 3: Integration with PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
from legal_graph.processors.negative_sampling import create_negative_samples

class LegalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'text': item['text'],
            'label': 1 if item['is_valid'] else 0
        }

# Load and sample data
original_data = load_legal_documents()
sampled_data = create_negative_samples(
    original_data,
    strategy='mixed',
    sampling_probability=0.5
)

# Create PyTorch dataset and dataloader
dataset = LegalDataset(sampled_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
for batch in dataloader:
    texts = batch['text']
    labels = batch['label']
    # ... training code
```

### Example 4: BigQuery Integration

```python
from google.cloud import bigquery
from legal_graph.processors.negative_sampling import NegativeSampler
import json

client = bigquery.Client()

# Query documents
query = """
SELECT doc_id, text, tags, triplets
FROM `project.dataset.legal_documents`
WHERE triplets_count > 0
LIMIT 1000
"""

results = client.query(query).to_dataframe()

# Convert to dataset format
dataset = []
for _, row in results.iterrows():
    dataset.append({
        'doc_id': row.doc_id,
        'text': row.text,
        'entities': json.loads(row.tags),
        'triplets': json.loads(row.triplets)
    })

# Create negative samples
sampler = NegativeSampler(strategy='aggressive')
negative_samples = sampler.create_negative_samples(dataset)

# Insert back to BigQuery
# ... (see CLI tool for full implementation)
```

## Statistics and Monitoring

```python
sampler = NegativeSampler(strategy='mixed')
result = sampler.create_negative_samples(dataset)

# Get detailed statistics
stats = sampler.get_sampling_stats()

print(f"Total processed: {stats['total_processed']}")
print(f"Negative samples created: {stats['corrupted']}")
print(f"Original samples kept: {stats['skipped']}")
print(f"Errors: {stats['errors']}")

# Modifications breakdown
for mod_type, count in stats['modifications_by_type'].items():
    print(f"{mod_type}: {count}")

# Example output:
# article_offset_+2: 145
# part_offset_+1: 67
# combined_article_offset_+3_part_offset_-1: 23
# random_component: 41
```

## Advanced Topics

### Custom SamplingStrategy

You can create your own sampling strategy by extending the `SamplingStrategy` base class:

```python
from legal_graph.processors.negative_sampling import SamplingStrategy

class CustomStrategy(SamplingStrategy):
    def modify_reference(self, article, part, code_name,
                         min_article, max_article, max_part=10):
        # Your custom logic here
        new_article = article + 10  # Example: always add 10
        new_part = part
        return new_article, new_part, "custom_modification"

    def get_strategy_name(self):
        return "CustomStrategy"

# Use your custom strategy
sampler = NegativeSampler(strategy=CustomStrategy())
```

### Filtering by Legal Code

```python
# Only sample from specific legal codes
sampler = NegativeSampler(
    strategy='article',
    article_ranges={
        'КК України': (1, 446),
        'КПК України': (1, 615)
    }
)
```

## Troubleshooting

### Issue: No modifications applied

**Cause:** Text doesn't contain recognizable legal references.

**Solution:** Check that your text contains patterns like `ч.X ст.Y КК України`.

### Issue: Invalid article numbers

**Cause:** Article ranges not set correctly for legal code.

**Solution:** Provide custom `article_ranges` parameter with correct ranges.

### Issue: Low negative sample rate

**Cause:** Low `sampling_probability` or documents without legal references.

**Solution:** Increase `sampling_probability` or verify input data quality.

## Contributing

To add new sampling strategies:

1. Create a new class extending `SamplingStrategy` in `strategies.py`
2. Implement `modify_reference()` and `get_strategy_name()` methods
3. Add to the strategy factory in `create_strategy()` function
4. Add tests and documentation

## License

Part of the TopoLegal project. See main project LICENSE.

## See Also

- [Balance Module](../balance/README.md) - Simple corruption for binary classification
- [Graph Extraction Module](../graph_extraction/README.md) - Knowledge graph extraction
- [Main Processors Documentation](../README.md) - Overview of all processors
