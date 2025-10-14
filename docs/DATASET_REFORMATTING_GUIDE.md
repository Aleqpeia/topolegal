# BigQuery Dataset Reformatting Guide

Complete guide for reformatting legal graph datasets from JSON triplets into columnar format with stratified train/test/validation splits.

## Overview

This guide covers:
1. Extracting triplet JSON arrays into separate columns (legal_reference, source, target, relation)
2. Defining grouping conditions for stratified splitting
3. Creating train/test/val splits with proper stratification
4. Validating splits to ensure no data leakage

## Files

- `bigquery_dataset_reformatter.py` - Python utility for automated reformatting
- `bigquery_reformatting_queries.sql` - SQL queries for manual execution

## Prerequisites

```bash
# Install dependencies
pip install google-cloud-bigquery pandas numpy

# Authenticate with Google Cloud
gcloud auth application-default login

# Or set service account credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## Quick Start

### Option 1: Using Python Utility (Recommended)

```bash
# 1. Analyze source data
python bigquery_dataset_reformatter.py analyze \
    --table lab-test-project-1-305710.court_data_2022.processing_doc_links \
    --output stats.json

# 2. Reformat dataset
python bigquery_dataset_reformatter.py reformat \
    --source-table lab-test-project-1-305710.court_data_2022.processing_doc_links \
    --dest-table lab-test-project-1-305710.court_data_2022.reformatted_triplets

# 3. Create stratified splits
python bigquery_dataset_reformatter.py split \
    --source-table lab-test-project-1-305710.court_data_2022.reformatted_triplets \
    --train-table lab-test-project-1-305710.court_data_2022.train \
    --test-table lab-test-project-1-305710.court_data_2022.test \
    --val-table lab-test-project-1-305710.court_data_2022.val \
    --train-ratio 0.7 \
    --test-ratio 0.2 \
    --val-ratio 0.1 \
    --stratify-by legal_reference

# 4. Validate splits
python bigquery_dataset_reformatter.py validate \
    --train-table lab-test-project-1-305710.court_data_2022.train \
    --test-table lab-test-project-1-305710.court_data_2022.test \
    --val-table lab-test-project-1-305710.court_data_2022.val

# 5. Create grouping view for analysis
python bigquery_dataset_reformatter.py create-view \
    --source-table lab-test-project-1-305710.court_data_2022.reformatted_triplets \
    --view-name lab-test-project-1-305710.court_data_2022.grouping_stats \
    --group-by legal_reference relation legal_code
```

### Option 2: Using SQL Queries

Run the queries from `bigquery_reformatting_queries.sql` in BigQuery console or via `bq` CLI.

```bash
# Execute SQL file
bq query --use_legacy_sql=false < bigquery_reformatting_queries.sql
```

## Detailed Workflow

### Step 1: Analyze Source Data

Understand the distribution of legal references and relations before splitting.

**Python:**
```bash
python bigquery_dataset_reformatter.py analyze \
    --table lab-test-project-1-305710.court_data_2022.processing_doc_links \
    --output statistics.json
```

**Output:**
```
Total triplets: 45,231
Unique legal references: 387
Unique relations: 156

Top 10 Legal References:
  ч.2 ст.185 КК України: 3,456 occurrences
  ст.391 КПК України: 2,891 occurrences
  ...

Top 10 Relations:
  визнаний_винним: 5,678 occurrences
  застосовано_стаття: 4,234 occurrences
  ...
```

### Step 2: Reformat Dataset

Extract triplet JSON into columnar format.

**Input Schema:**
```sql
doc_id: STRING
text: STRING
tags: STRING  -- JSON array
triplets: STRING  -- JSON array like:
  [
    {
      "source": "ОСОБА_1",
      "relation": "визнаний_винним",
      "target": "крадіжка",
      "legal_reference": "ч.2 ст.185 КК України",
      "confidence": 0.95
    },
    ...
  ]
```

**Output Schema:**
```sql
doc_id: STRING
text: STRING
tags: STRING
triplets_count: INT64
triplet_index: INT64  -- Position in array (1, 2, 3, ...)
legal_reference: STRING  -- ч.2 ст.185 КК України
source: STRING  -- ОСОБА_1
target: STRING  -- крадіжка
relation: STRING  -- визнаний_винним
confidence: FLOAT64
article_number: STRING  -- 185
part_number: STRING  -- 2
legal_code: STRING  -- КК
reformatted_timestamp: TIMESTAMP
```

**Python:**
```bash
python bigquery_dataset_reformatter.py reformat \
    --source-table lab-test-project-1-305710.court_data_2022.processing_doc_links \
    --dest-table lab-test-project-1-305710.court_data_2022.reformatted_triplets
```

**SQL (from bigquery_reformatting_queries.sql #1):**
```sql
CREATE OR REPLACE TABLE `project.dataset.reformatted_table` AS
WITH triplet_data AS (
    SELECT
        doc_id,
        text,
        tags,
        triplets,
        triplets_count,
        JSON_EXTRACT_ARRAY(triplets) AS triplet_array
    FROM `project.dataset.source_table`
    WHERE triplets IS NOT NULL AND triplets != '[]' AND triplets_count > 0
),
exploded_triplets AS (
    SELECT
        doc_id,
        text,
        tags,
        triplets_count,
        ROW_NUMBER() OVER (PARTITION BY doc_id) AS triplet_index,
        JSON_EXTRACT_SCALAR(triplet, '$.legal_reference') AS legal_reference,
        JSON_EXTRACT_SCALAR(triplet, '$.source') AS source,
        JSON_EXTRACT_SCALAR(triplet, '$.target') AS target,
        JSON_EXTRACT_SCALAR(triplet, '$.relation') AS relation,
        CAST(JSON_EXTRACT_SCALAR(triplet, '$.confidence') AS FLOAT64) AS confidence
    FROM triplet_data, UNNEST(triplet_array) AS triplet
)
SELECT *, CURRENT_TIMESTAMP() AS reformatted_timestamp
FROM exploded_triplets;
```

### Step 3: Define Grouping Conditions

Create views to analyze grouping by legal references and relations.

**Legal Reference Groups:**
```bash
# Using SQL query #3 from bigquery_reformatting_queries.sql
```

Creates view with:
- `frequency_category`: high_frequency (≥100 docs), medium_frequency (≥20), low_frequency (≥5), rare (<5)
- `code_category`: criminal, criminal_procedure, civil, administrative, family, labor

**Relation Groups:**
```bash
# Using SQL query #4 from bigquery_reformatting_queries.sql
```

Creates view with:
- `frequency_category`: Similar to legal references
- `semantic_category`: conviction, acquittal, sentencing, violation, other

### Step 4: Create Stratified Splits

Split dataset while maintaining distribution of legal references and relations.

**Why Stratification Matters:**
- Ensures all splits have representative samples of each legal reference
- Prevents training on common references and testing on rare ones
- Maintains relation distribution across splits

**Python (Recommended):**
```bash
python bigquery_dataset_reformatter.py split \
    --source-table lab-test-project-1-305710.court_data_2022.reformatted_triplets \
    --train-table lab-test-project-1-305710.court_data_2022.train \
    --test-table lab-test-project-1-305710.court_data_2022.test \
    --val-table lab-test-project-1-305710.court_data_2022.val \
    --train-ratio 0.7 \
    --test-ratio 0.2 \
    --val-ratio 0.1 \
    --stratify-by legal_reference \
    --seed 42
```

**Stratification Options:**
- `--stratify-by legal_reference` - Stratify by legal reference (default)
- `--stratify-by relation` - Stratify by relation type
- `--stratify-by legal_code` - Stratify by legal code (КК, КПК, etc.)

**SQL (Alternative):**
Uses hash-based deterministic splitting (queries #6, #7, #8):
```sql
-- Training set (70%)
CREATE OR REPLACE TABLE `project.dataset.train` AS
WITH doc_assignment AS (
    SELECT DISTINCT
        doc_id,
        MOD(ABS(FARM_FINGERPRINT(CONCAT(doc_id, 'salt42'))), 100) AS hash_value
    FROM `project.dataset.reformatted_table`
)
SELECT r.*
FROM `project.dataset.reformatted_table` r
INNER JOIN doc_assignment da ON r.doc_id = da.doc_id
WHERE da.hash_value < 70;
```

### Step 5: Validate Splits

Ensure no data leakage between splits.

**Python:**
```bash
python bigquery_dataset_reformatter.py validate \
    --train-table lab-test-project-1-305710.court_data_2022.train \
    --test-table lab-test-project-1-305710.court_data_2022.test \
    --val-table lab-test-project-1-305710.court_data_2022.val
```

**Output:**
```
✅ Validation passed: No document overlap detected

TRAIN: 31,662 docs, 85,432 triplets
TEST: 9,048 docs, 24,123 triplets
VAL: 4,524 docs, 12,061 triplets
```

**SQL Validation (Query #9):**
```sql
-- Check for overlapping documents
WITH all_docs AS (
    SELECT doc_id, 'train' as split FROM `project.dataset.train`
    UNION ALL
    SELECT doc_id, 'test' as split FROM `project.dataset.test`
    UNION ALL
    SELECT doc_id, 'val' as split FROM `project.dataset.val`
),
doc_counts AS (
    SELECT doc_id, COUNT(DISTINCT split) as split_count
    FROM all_docs
    GROUP BY doc_id
    HAVING split_count > 1
)
SELECT COUNT(*) as overlapping_docs FROM doc_counts;
-- Should return 0
```

### Step 6: Check Distribution Coverage

Verify legal references and relations are represented across all splits.

**SQL Query #11 (Legal Reference Coverage):**
```sql
WITH legal_ref_coverage AS (
    SELECT
        legal_reference,
        COUNT(DISTINCT CASE WHEN split = 'train' THEN doc_id END) as in_train,
        COUNT(DISTINCT CASE WHEN split = 'test' THEN doc_id END) as in_test,
        COUNT(DISTINCT CASE WHEN split = 'val' THEN doc_id END) as in_val,
        COUNT(DISTINCT split) as num_splits
    FROM (
        SELECT doc_id, legal_reference, 'train' as split FROM `project.dataset.train`
        UNION ALL
        SELECT doc_id, legal_reference, 'test' as split FROM `project.dataset.test`
        UNION ALL
        SELECT doc_id, legal_reference, 'val' as split FROM `project.dataset.val`
    )
    GROUP BY legal_reference
)
SELECT
    COUNT(*) as total_legal_refs,
    COUNTIF(num_splits = 3) as in_all_three_splits,
    ROUND(100.0 * COUNTIF(num_splits = 3) / COUNT(*), 2) as pct_in_all_splits
FROM legal_ref_coverage;
```

**Expected Output:**
```
total_legal_refs: 387
in_all_three_splits: 342 (88.37%)
```

## Advanced Usage

### Create Custom Grouping View

```bash
python bigquery_dataset_reformatter.py create-view \
    --source-table lab-test-project-1-305710.court_data_2022.reformatted_triplets \
    --view-name lab-test-project-1-305710.court_data_2022.custom_grouping \
    --group-by legal_reference relation legal_code article_number
```

### Export to CSV

```bash
# Export training set
bq extract --destination_format=CSV \
    lab-test-project-1-305710:court_data_2022.train \
    gs://your-bucket/train.csv

# Export test set
bq extract --destination_format=CSV \
    lab-test-project-1-305710:court_data_2022.test \
    gs://your-bucket/test.csv

# Export validation set
bq extract --destination_format=CSV \
    lab-test-project-1-305710:court_data_2022.val \
    gs://your-bucket/val.csv
```

### Load Splits into Python

```python
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

# Load training data
query = "SELECT * FROM `lab-test-project-1-305710.court_data_2022.train`"
train_df = client.query(query).to_dataframe()

# Load test data
query = "SELECT * FROM `lab-test-project-1-305710.court_data_2022.test`"
test_df = client.query(query).to_dataframe()

# Load validation data
query = "SELECT * FROM `lab-test-project-1-305710.court_data_2022.val`"
val_df = client.query(query).to_dataframe()

print(f"Train: {len(train_df)} rows")
print(f"Test: {len(test_df)} rows")
print(f"Val: {len(val_df)} rows")
```

### Use with PyTorch Geometric

```python
from legal_graph.models import GraphDataset, create_dataloader

# Create datasets from reformatted tables
train_dataset = GraphDataset(
    table_id="lab-test-project-1-305710.court_data_2022.train",
    max_nodes=100,
    max_edges=200
)

test_dataset = GraphDataset(
    table_id="lab-test-project-1-305710.court_data_2022.test",
    max_nodes=100,
    max_edges=200
)

val_dataset = GraphDataset(
    table_id="lab-test-project-1-305710.court_data_2022.val",
    max_nodes=100,
    max_edges=200
)

# Create dataloaders
train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)
val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)
```

## Best Practices

### 1. Always Stratify
- Use legal_reference for most cases
- Use relation for relation-focused tasks
- Use legal_code for code-level stratification

### 2. Check Coverage
- Verify all important legal references appear in all splits
- Rare references (<5 docs) may only appear in one split

### 3. Validate Thoroughly
- Check for document overlap (should be 0)
- Compare distributions across splits
- Verify proportions (70/20/10)

### 4. Document Your Splits
- Save statistics and analysis results
- Record random seed for reproducibility
- Keep track of table names and versions

### 5. Handle Rare Cases
- Consider grouping rare legal references together
- Or exclude rare references from test/val if they're too rare
- Balance between representation and statistical significance

## Troubleshooting

### Issue: "No overlapping documents should be 0 but found X"
**Solution:** Check if you're using the same random seed. Re-run split with different seed.

### Issue: "Some legal references only in training set"
**Solution:** This is expected for very rare references (<5 docs). Either:
- Exclude them from analysis
- Group similar references together
- Use different stratification approach

### Issue: "Unbalanced splits"
**Solution:** Check your ratios sum to 1.0. Adjust train/test/val ratios as needed.

### Issue: "BigQuery authentication failed"
**Solution:**
```bash
gcloud auth application-default login
# Or
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

## Summary

This reformatting process:
1. ✅ Extracts JSON triplets into columnar format
2. ✅ Adds metadata columns (article_number, part_number, legal_code)
3. ✅ Provides stratified splitting by legal_reference, relation, or legal_code
4. ✅ Validates splits for data leakage
5. ✅ Checks distribution coverage across splits
6. ✅ Integrates with PyTorch Geometric datasets

The reformatted data is ready for GNN training with proper train/test/val splits and no data leakage.
