# Negative Sampling Implementation Fix

## Quick Summary

Fixed **4 critical issues** and added **1 major feature** in the negative sampling implementation:

1. **🔴 CRITICAL**: Text-Triplet Consistency - Modifications now consistently applied across text and triplets
2. Array Ambiguity - Fixed `pd.notna()` issues with arrays
3. BigQuery numpy.ndarray - Handle BigQuery's auto-parsed JSON arrays  
4. Non-Empty Triplets - Filter out documents without triplets
5. **✨ NEW**: Punkt Format Support - Added support for "п." (punkt) in addition to "ст." (stattya)

**All tests passing!** ✅ Ready for production use.

## Issues Fixed (Detailed)

### ✅ 1. **CRITICAL**: Text-Triplet Consistency (RESOLVED)

**Problem**: Modifications to legal references in text were NOT being applied to the same references in triplets, causing critical data inconsistency. The text would say "ч.2 ст.187" but the triplet would say "ч.2 ст.190" - completely different!

**Why Critical**: This creates invalid training data where text and structured data contradict each other, confusing the model and producing poor results.

**Solution**: Implemented a **reference mapping system** that:
1. First modifies the text and tracks all changes (original → modified)
2. Then applies the **exact same modifications** to matching references in triplets
3. Also applies to entities for complete consistency

```python
# Now enforces consistency:
# Text: "ч.2 ст.185" → "ч.2 ст.187"
# Triplet: "ч.2 ст.185" → "ч.2 ст.187"  ← Same modification!
```

**Verification**: All consistency tests pass. See `CONSISTENCY_FIX.md` for details and `test_consistency_fix.py` for test suite.

### ✅ 2. Array Ambiguity Error with Multiple Legal References (RESOLVED)

**Problem**: The error `The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()` was occurring when parsing documents with multiple legal references.

**Root Cause**: Using `pd.notna()` on array/list values in pandas DataFrames returns an array of boolean values, which cannot be used in boolean contexts like `if pd.notna(value):`.

**Solution**: Replaced all `pd.notna()` checks with explicit None checking and string validation:

```python
# Before (problematic):
entities = json.loads(row.tags) if pd.notna(row.tags) else []

# After (fixed):
tags_value = row.tags
if tags_value is not None and str(tags_value).strip() and str(tags_value) != 'nan':
    try:
        entities = json.loads(str(tags_value))
        if not isinstance(entities, list):
            entities = []
    except (json.JSONDecodeError, ValueError):
        entities = []
else:
    entities = []
```

### ✅ 3. BigQuery Returns JSON as numpy.ndarray (RESOLVED)

**Problem**: Even though BigQuery had thousands of documents with triplets, ALL were failing to parse with error: `Expecting property name enclosed in double quotes: line 1 column 3 (char 2)`

**Root Cause**: BigQuery's Python client auto-parses JSON columns into `numpy.ndarray` objects. The code only checked for `list` type, so it fell through to `json.loads(str(numpy_array))`. When you convert a numpy array to string, it uses Python's repr format with **single quotes** `{'key': 'value'}`, but JSON requires **double quotes** `{"key": "value"}`.

**Solution**: Updated `isinstance()` checks to handle both `list` and `np.ndarray`:
```python
if isinstance(triplets_value, (list, np.ndarray)):
    triplets = list(triplets_value) if hasattr(triplets_value, 'tolist') else triplets_value
```

See `NUMPY_ARRAY_FIX.md` for detailed explanation.

### ✅ 4. Non-Empty Triplets Filtering (RESOLVED)

**Problem**: Documents without triplets were being processed, which is wasteful since negative sampling requires legal references in triplets.

**Solution**: Added multi-level filtering:

### ✅ 5. **NEW FEATURE**: Punkt Format Support (ADDED)

**Feature**: Added support for **"п." (punkt)** format in addition to existing **"ст." (stattya)** format.

**Why Important**: Ukrainian legal documents use both formats:
- **"ст."** (стаття) = article - main legal provisions
- **"п."** (пункт) = punkt/point - sub-sections and list items

**Formats Now Supported**:
```
✅ ч.2 ст.185 КК України  (part + article) - existing
✅ ст.185 КК України      (article only) - existing  
✅ ч.1 п.5 КК України     (part + punkt) - NEW
✅ п.12 КК України        (punkt only) - NEW
```

**Consistency**: All modifications maintain text-triplet consistency across both formats.

See `PUNKT_FORMAT_SUPPORT.md` for detailed documentation and examples.

#### Database Level (BigQuery SQL):
```sql
WHERE   text IS NOT NULL
    AND LENGTH(text) > 10
    AND triplets IS NOT NULL
    AND triplets_count > 0
    AND JSON_EXTRACT(triplets, '$[0]') IS NOT NULL
```

#### Application Level:
- CSV/JSON: Pre-filter datasets to only include documents with non-empty triplets
- All sources: Validate and skip documents without triplets during processing
- Added informative logging about filtering results

## Files Modified

### `/src/legal_graph/processors/negative_sampling/__main__.py`

**Changes in imports**:
- Added `import numpy as np` to handle BigQuery's numpy arrays

**Changes in `sample_csv_data()`**:
- Safe parsing of tags/triplets with proper None/nan handling
- Handle both `list` and `np.ndarray` types
- Filter documents without triplets
- Added informative logging

**Changes in `sample_json_data()`**:
- Pre-filter dataset for non-empty triplets
- Added early return if no valid documents
- Added informative logging

**Changes in `sample_bigquery_data()`**:
- Added pre-flight statistics check to diagnose data issues
- Updated SQL query to filter at database level
- Rewrote row parsing to avoid `pd.notna()` issues  
- **Handle `numpy.ndarray` types from BigQuery's auto-parsed JSON**
- Added safe JSON parsing with type validation
- Added batch-level validation
- Enhanced logging with document counts and type information

## Test Results

All tests pass successfully:

```bash
$ poetry run python test_negative_sampling_fix.py

Testing multiple legal references...
✓ Document was corrupted as expected
  Original: згідно з ч.2 ст.185 КК України та ч.1 ст.190 КК України
  Modified: згідно з ч.2 ст.182 КК України та ч.1 ст.185 КК України
✓ Multiple references test passed!

Testing empty triplets filtering...
✓ Filtered 2 -> 1 documents
✓ Empty triplets filtering test passed!

Testing array value handling...
✓ Array handling test passed!

Testing different strategies...
  ✓ Strategy 'article' works
  ✓ Strategy 'part' works
  ✓ Strategy 'combined' works
  ✓ Strategy 'mixed' works
  ✓ Strategy 'conservative' works
  ✓ Strategy 'aggressive' works
✓ Strategy variations test passed!

✅ All tests passed!
```

## Usage

The fixes are transparent to end users. Use the negative sampling tool as before:

```bash
# BigQuery example
poetry run python -m legal_graph.processors.negative_sampling sample-bigquery \
    lab-test-project-1-305710.main_dataset.extracted_data \
    lab-test-project-1-305710.main_dataset.negative_samples \
    --strategy mixed \
    --probability 0.5 \
    --batch-size 100

# CSV example
poetry run python -m legal_graph.processors.negative_sampling sample-csv \
    input.csv output.csv \
    --strategy article \
    --probability 0.3

# JSON example
poetry run python -m legal_graph.processors.negative_sampling sample-json \
    input.json output.json \
    --strategy aggressive \
    --probability 0.7
```

## Expected Behavior Changes

1. **No more crashes**: Documents with multiple legal references process cleanly
2. **Better filtering**: Only documents with triplets are processed
3. **Improved logging**: Clear visibility into how many documents are filtered/processed
4. **Better performance**: BigQuery filtering reduces network transfer and processing time

## Performance Impact

- ✅ **Reduced network transfer**: BigQuery filtering happens at database level
- ✅ **Faster processing**: Skip invalid documents early
- ✅ **Better resource usage**: Don't process documents that can't be sampled
- ✅ **Clearer metrics**: Accurate counts of valid vs filtered documents

## Backward Compatibility

- ✅ Public API unchanged
- ✅ Command-line interface unchanged
- ✅ Output format unchanged
- ⚠️ **Behavioral change**: Documents without triplets are now filtered out (as intended)

## Related Documentation

- `BUGFIX_MULTIPLE_REFS.md` - Detailed technical documentation
- `test_negative_sampling_fix.py` - Test suite validating the fixes
- `README.md` - Usage guide and examples
- `ARCHITECTURE.md` - Architecture overview

## Next Steps

You can now run your negative sampling pipeline on BigQuery data without the array ambiguity errors. The implementation will:

1. Efficiently filter for documents with triplets at the database level
2. Safely parse documents with multiple legal references
3. Apply your chosen sampling strategy to create negative samples
4. Provide clear logging about processing progress

If you encounter any issues, check the logs for filtering statistics and validation messages.

