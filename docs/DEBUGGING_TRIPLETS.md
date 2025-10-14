# Debugging: No Valid Documents with Triplets

## Issue

The negative sampling process is reporting "⚠️  No valid documents with triplets" even though the source table should contain many documents with triplets.

## Diagnostic Changes Added

### 1. Source Table Statistics Check

Before processing begins, the tool now queries the source table to show:
- Total documents with text
- How many have non-null `triplets` field
- How many have `triplets_count > 0`
- How many have JSON-extractable triplets

This helps identify if the issue is at the database level or the parsing level.

### 2. Sample Data Inspection

The first 3 rows in the first 2 batches (first 200 documents) now log:
- Document ID
- Triplets field type (list, str, dict, etc.)
- First 200 characters of triplets value
- Triplets count value

This shows exactly what format BigQuery is returning the data in.

### 3. Enhanced Parsing Logic

The code now handles multiple data formats:
- **Already parsed lists**: If BigQuery returns triplets as a Python list, use it directly
- **JSON strings**: If it's a string, parse with `json.loads()`
- **Invalid data**: Log when parsing fails

### 4. Better Error Messages

- Shows how many rows were fetched vs how many were valid
- Indicates data format issues
- Points to debug logs for investigation

## How to Run Diagnostic

Run the negative sampling with these changes:

```bash
poetry run python -m legal_graph.processors.negative_sampling sample-bigquery \
    lab-test-project-1-305710.main_dataset.extracted_data \
    lab-test-project-1-305710.main_dataset.negative_samples_debug \
    --strategy mixed \
    --probability 0.5 \
    --batch-size 100
```

## What to Look For in Output

### Expected Output (Good Case):

```
🔍 Checking source table statistics...
   Total documents: 50000
   With triplets (not null): 25000
   With triplets_count > 0: 25000
   With JSON extractable triplets: 25000

🔍 Row 0 - doc_id: 102524715
  triplets type: <class 'list'>
  triplets value (first 200 chars): [{'source': '...', 'relation': '...', 'target': '...'}]
  triplets_count: 3

📊 Processed 100 documents
   Valid documents with triplets: 78
   Sampled: 42, Skipped: 36
```

### Problematic Output (Data Issue):

```
🔍 Checking source table statistics...
   Total documents: 50000
   With triplets (not null): 100         ← Very few!
   With triplets_count > 0: 5000         ← Mismatch!
   With JSON extractable triplets: 50    ← Almost none!

🔍 Row 0 - doc_id: 102524715
  triplets type: <class 'str'>
  triplets value (first 200 chars): "null"         ← String "null" not actual null
  triplets_count: 0

⚠️  Skipping - no valid triplets (triplets was: <class 'str'>)
⚠️  No valid documents with triplets in batch at offset 0
    Fetched 100 rows from BigQuery, but none had valid triplets after parsing
```

## Possible Issues and Solutions

### Issue 1: Triplets field contains string "null" instead of NULL

**Symptom**: 
- SQL shows many rows with `triplets IS NOT NULL`
- But parsing shows `triplets_value = "null"` (string)

**Solution**: Update SQL filter to exclude string "null":
```sql
AND triplets IS NOT NULL
AND triplets != 'null'
AND triplets != '[]'
```

### Issue 2: Triplets in different column or format

**Symptom**:
- Statistics show low counts
- Sample data shows unexpected format

**Solution**: 
- Check the actual table schema
- Verify column names match expectations
- Check if JSON is stored as STRING vs JSON type

### Issue 3: triplets_count doesn't match actual triplet array length

**Symptom**:
- `triplets_count > 0` but triplets array is empty
- Mismatch between count field and actual data

**Solution**: Rely on actual array content, not the count field:
```sql
WHERE JSON_EXTRACT(triplets, '$[0]') IS NOT NULL  -- Has at least one element
```

## Next Steps

After running with diagnostics:

1. **Review the statistics output** - Are there triplets in the source table?
2. **Check sample data format** - What type is the triplets field?
3. **Look for patterns in failures** - Are all rows failing or just some?
4. **Adjust filtering/parsing** - Based on what the actual data format is

Share the output with the debug information and we can adjust the code to handle your specific data format.

