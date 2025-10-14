# Bug Fix: Multiple Legal References and Non-Empty Triplets Filtering

## Issues Identified

### 1. Array Ambiguity Error
**Problem**: When parsing data from BigQuery/CSV, the code used `pd.notna()` which can return an array when the value itself is an array. This caused the error:
```
The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

**Root Cause**: The pattern `if pd.notna(value):` fails when `value` is already an array or list, because pandas returns an array of boolean values rather than a single boolean.

**Solution**: Replaced boolean checks with explicit None checks and string validation:
```python
# Before (problematic)
entities = json.loads(row.tags) if pd.notna(row.tags) else []

# After (fixed)
tags_value = row.tags
if tags_value is not None and str(tags_value).strip() and str(tags_value) != 'nan':
    entities = json.loads(str(tags_value))
```

### 2. Multiple Legal References Not Handled
**Problem**: Documents can contain multiple legal references (multiple triplets, multiple entities with legal refs), but the processing was failing when encountering them.

**Solution**: The code already handles multiple references by iterating through all matches in regex patterns and all items in triplets/entities lists. The fix in #1 allows this to work properly now.

### 3. Missing Filter for Non-Empty Triplets
**Problem**: Processing was including documents without triplets, which are not useful for negative sampling of legal references.

**Solution**: Added filtering at multiple levels:

#### SQL Query Level (BigQuery)
```sql
WHERE   text IS NOT NULL
    AND LENGTH(text) > 10
    AND triplets IS NOT NULL
    AND triplets_count > 0
    AND JSON_EXTRACT(triplets, '$[0]') IS NOT NULL
```

#### Python Processing Level
```python
# Skip if no triplets (we only want documents with triplets)
if not triplets:
    continue
```

## Changes Made

### File: `__main__.py`

#### 1. CSV Data Processing (`sample_csv_data`)
- Added safe parsing of tags and triplets with proper None/nan checks
- Added filtering to skip documents without triplets
- Added informative logging about filtering results

#### 2. JSON Data Processing (`sample_json_data`)
- Added pre-filtering of dataset to only include documents with non-empty triplets
- Added informative logging and early return if no valid documents found

#### 3. BigQuery Data Processing (`sample_bigquery_data`)
- Updated SQL query to filter for non-empty triplets at database level
- Rewrote row parsing to avoid `pd.notna()` ambiguity issues
- Added safe JSON parsing with type checking
- Added validation to skip documents without triplets after parsing
- Added informative logging about valid document counts

## Testing

To test the fixes:

```bash
# Test with BigQuery
poetry run python -m legal_graph.processors.negative_sampling sample-bigquery \
    lab-test-project-1-305710.main_dataset.extracted_data \
    lab-test-project-1-305710.main_dataset.negative_samples_test \
    --strategy mixed \
    --probability 0.5 \
    --batch-size 100
```

## Expected Behavior

1. **No more array ambiguity errors**: Documents with multiple legal references process cleanly
2. **Only documents with triplets**: Filtering excludes documents without legal references
3. **Better logging**: Clear messages about how many documents are filtered vs processed
4. **Efficient querying**: BigQuery filters at database level, reducing data transfer

## Performance Impact

- **Positive**: BigQuery filtering reduces network transfer and processing time
- **Positive**: Early validation prevents wasted processing on invalid documents
- **Neutral**: Additional validation checks add minimal overhead compared to benefits

## Backward Compatibility

- The changes maintain the same public API
- Existing call sites continue to work without modification
- Only behavioral change: documents without triplets are now filtered out (as intended)

