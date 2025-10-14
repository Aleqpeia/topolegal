# Dataset Reformatting Quick Reference

Quick reference for common dataset reformatting operations.

## Common Commands

### Analyze Dataset
```bash
python bigquery_dataset_reformatter.py analyze \
    --table lab-test-project-1-305710.court_data_2022.processing_doc_links \
    --output stats.json
```

### Reformat Dataset
```bash
python bigquery_dataset_reformatter.py reformat \
    --source-table lab-test-project-1-305710.court_data_2022.processing_doc_links \
    --dest-table lab-test-project-1-305710.court_data_2022.reformatted_triplets
```

### Create Splits
```bash
python bigquery_dataset_reformatter.py split \
    --source-table lab-test-project-1-305710.court_data_2022.reformatted_triplets \
    --train-table lab-test-project-1-305710.court_data_2022.train \
    --test-table lab-test-project-1-305710.court_data_2022.test \
    --val-table lab-test-project-1-305710.court_data_2022.val \
    --train-ratio 0.7 --test-ratio 0.2 --val-ratio 0.1 \
    --stratify-by legal_reference
```

### Validate Splits
```bash
python bigquery_dataset_reformatter.py validate \
    --train-table lab-test-project-1-305710.court_data_2022.train \
    --test-table lab-test-project-1-305710.court_data_2022.test \
    --val-table lab-test-project-1-305710.court_data_2022.val
```

### Create Grouping View
```bash
python bigquery_dataset_reformatter.py create-view \
    --source-table lab-test-project-1-305710.court_data_2022.reformatted_triplets \
    --view-name lab-test-project-1-305710.court_data_2022.grouping_stats \
    --group-by legal_reference relation
```

## Complete Workflow Example
```bash
# Run all steps in one go
python example_dataset_reformatting.py
```

## Schema Transformation

### Before (JSON Array)
```json
{
  "doc_id": "doc_001",
  "triplets": "[{\"source\": \"ОСОБА_1\", \"relation\": \"визнаний_винним\", \"target\": \"крадіжка\", \"legal_reference\": \"ч.2 ст.185 КК України\", \"confidence\": 0.95}]"
}
```

### After (Columnar)
```
doc_id | triplet_index | legal_reference        | source   | target   | relation          | confidence | article_number | part_number | legal_code
-------|---------------|------------------------|----------|----------|-------------------|------------|----------------|-------------|------------
doc_001| 1             | ч.2 ст.185 КК України  | ОСОБА_1  | крадіжка | визнаний_винним   | 0.95       | 185            | 2           | КК
```

## Stratification Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `legal_reference` | Stratify by legal reference (e.g., "ч.2 ст.185 КК України") | Default, ensures all legal articles represented |
| `relation` | Stratify by relation type (e.g., "визнаний_винним") | Relation-focused tasks |
| `legal_code` | Stratify by legal code (e.g., "КК", "КПК") | Code-level analysis |

## Split Ratios

Common configurations:
- **Standard**: 70/20/10 (train/test/val)
- **Large dataset**: 80/10/10
- **Small dataset**: 60/20/20

## Validation Checklist

- [ ] No document overlap (overlapping_docs = 0)
- [ ] Split ratios approximately correct (±2%)
- [ ] All major legal references in all splits
- [ ] All major relations in all splits
- [ ] Distribution coverage ≥85% for important features

## Export Splits

```bash
# To CSV
bq extract --destination_format=CSV \
    lab-test-project-1-305710:court_data_2022.train \
    gs://your-bucket/train.csv

# To JSON
bq extract --destination_format=NEWLINE_DELIMITED_JSON \
    lab-test-project-1-305710:court_data_2022.train \
    gs://your-bucket/train.jsonl
```

## Use in Python

```python
from legal_graph.models import GraphDataset

# Load splits
train_dataset = GraphDataset(
    table_id="lab-test-project-1-305710.court_data_2022.train"
)
test_dataset = GraphDataset(
    table_id="lab-test-project-1-305710.court_data_2022.test"
)
val_dataset = GraphDataset(
    table_id="lab-test-project-1-305710.court_data_2022.val"
)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Authentication failed" | Run `gcloud auth application-default login` |
| "Table not found" | Check table ID format: `project.dataset.table` |
| "Overlapping documents" | Re-run split with different random seed |
| "Unbalanced splits" | Check ratios sum to 1.0 |

## Files

- `bigquery_dataset_reformatter.py` - Main utility script
- `bigquery_reformatting_queries.sql` - SQL queries
- `DATASET_REFORMATTING_GUIDE.md` - Complete guide
- `example_dataset_reformatting.py` - Example workflow

## Quick Test

```bash
# Test with small sample
python bigquery_dataset_reformatter.py analyze \
    --table lab-test-project-1-305710.court_data_2022.processing_doc_links \
    | head -20
```
