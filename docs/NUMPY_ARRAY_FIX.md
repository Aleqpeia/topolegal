# Fix: BigQuery Returns JSON as numpy.ndarray

## The Issue

When BigQuery returns JSON columns via the Python client, it automatically parses them into Python objects. Specifically, JSON arrays become **`numpy.ndarray`** objects, not plain Python lists.

### What Was Happening

1. BigQuery query returns `triplets` column
2. Pandas/BigQuery client auto-parses JSON → returns `numpy.ndarray`
3. Our code checked `isinstance(triplets_value, list)` → **False** (it's ndarray, not list!)
4. Code falls through to `json.loads(str(triplets_value))`
5. `str(numpy_array)` produces Python repr with **single quotes**: `[{'key': 'value'}]`
6. `json.loads()` expects **double quotes** for valid JSON: `[{"key": "value"}]`
7. **Error**: `Expecting property name enclosed in double quotes: line 1 column 3 (char 2)`

### Diagnostic Output

```
🔍 Checking source table statistics...
   Total documents: 31777
   With triplets (not null): 10496      ← Plenty of triplets exist!
   With triplets_count > 0: 9164
   With JSON extractable triplets: 9164

🔍 Row 0 - doc_id: 102384793
  triplets type: <class 'numpy.ndarray'>  ← The problem!
  triplets value: [{'confidence': 0.8, 'legal_reference': '...'}]
  ❌ Failed to parse triplets: Expecting property name enclosed in double quotes
```

## The Fix

Updated the `isinstance()` checks to handle both `list` and `np.ndarray`:

```python
# Before (only checked for list)
if isinstance(triplets_value, list):
    triplets = triplets_value
elif triplets_value is not None:
    triplets = json.loads(str(triplets_value))  # FAILS for ndarray!

# After (handles both list and numpy.ndarray)
if isinstance(triplets_value, (list, np.ndarray)):
    triplets = list(triplets_value) if hasattr(triplets_value, 'tolist') else triplets_value
elif triplets_value is not None:
    triplets = json.loads(str(triplets_value))
```

### Why This Works

- `isinstance(value, (list, np.ndarray))` catches both types
- `list(numpy_array)` safely converts ndarray to plain Python list
- Avoids the `str()` → `json.loads()` conversion that was breaking
- Falls back to JSON parsing only for actual string values

## Files Changed

- `src/legal_graph/processors/negative_sampling/__main__.py`:
  - Added `import numpy as np`
  - Updated `isinstance()` checks for `tags` and `triplets` parsing
  - Both CSV/JSON and BigQuery code paths now handle numpy arrays

## Testing

Run the negative sampling again:

```bash
poetry run python -m legal_graph.processors.negative_sampling sample-bigquery \
    lab-test-project-1-305710.court_data_2022.processing_doc_links \
    lab-test-project-1-305710.court_data_2022.docs_with_negatives \
    --strategy mixed \
    --probability 0.5 \
    --batch-size 100
```

### Expected Output (Now Fixed)

```
🔍 Checking source table statistics...
   Total documents: 31777
   With triplets (not null): 10496
   With triplets_count > 0: 9164

🔍 Row 0 - doc_id: 102384793
  triplets type: <class 'numpy.ndarray'>
  triplets value: [{'confidence': 0.8, 'legal_reference': 'ч.1 ст.331 КПК України', ...}]
  triplets_count: 1

📊 Processed 100 documents
   Valid documents with triplets: 100  ← All processed successfully!
   Sampled: 47, Skipped: 53
```

## Why BigQuery Does This

BigQuery's Python client uses pandas DataFrames for result sets. When converting JSON columns:

1. **JSON type in BigQuery** → pandas tries to parse it
2. **JSON arrays** → converted to `numpy.ndarray` for efficiency
3. **JSON objects** → might become Python dicts or stay as strings
4. **Nested structures** → numpy arrays of dicts/lists

This is efficient for data science workflows but requires handling multiple types in parsing logic.

## Related Issues

- `BUGFIX_MULTIPLE_REFS.md` - Original array ambiguity issue with `pd.notna()`
- `DEBUGGING_TRIPLETS.md` - Diagnostic steps that revealed this issue

## Lessons Learned

1. **Don't assume JSON strings**: Cloud database clients often auto-parse JSON
2. **Check actual types**: Use diagnostics to see what types you're really getting
3. **Handle numpy arrays**: When working with pandas/BigQuery, expect numpy types
4. **Avoid str() on complex objects**: Converting objects to strings for JSON parsing is fragile

