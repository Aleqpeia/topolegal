# Added Support for "п." (Punkt) Legal Reference Format

## Overview

Extended the negative sampling implementation to support **"п." (punkt/point)** format in addition to the existing **"ст." (stattya/article)** format. This covers all common ways Ukrainian legal documents reference laws.

## Supported Formats

### Before (Only "ст." format)
- ✅ `ч.2 ст.185 КК України` - part + article
- ✅ `ст.185 КК України` - article only
- ❌ `ч.1 п.5 КК України` - NOT supported
- ❌ `п.12 КК України` - NOT supported

### After (Both "ст." and "п." formats)
- ✅ `ч.2 ст.185 КК України` - part + article (stattya)
- ✅ `ст.185 КК України` - article only (stattya)
- ✅ `ч.1 п.5 КК України` - part + punkt (**NEW**)
- ✅ `п.12 КК України` - punkt only (**NEW**)

## Why "п." Format?

In Ukrainian legal documents:
- **"ст."** (стаття) = article - most common format
- **"п."** (пункт) = punkt/point - used for sub-sections or list items

Both are valid ways to reference legal provisions and appear frequently in court documents.

## Implementation

### New Regex Patterns

```python
# Pattern for "ч.X п.Y КК України"
self.part_punkt_pattern = re.compile(
    r'(ч\.\s*)(\d+)(\s+п\.\s*)(\d+)(\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)',
    re.IGNORECASE
)

# Pattern for "п.Y КК України"  
self.punkt_pattern = re.compile(
    r'(п\.\s*)(\d+)(\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)',
    re.IGNORECASE
)
```

### Modification Logic

Punkt numbers are treated the same as article numbers for modification purposes:
- Applied the same sampling strategies (offset, swap, random, etc.)
- Maintains text-triplet consistency
- Tracks modifications with format metadata

## Test Results

All format tests pass with 100% consistency:

```bash
$ poetry run python test_punkt_support.py

Test 1 (ч.X ст.Y): ✅ PASS
Test 2 (ч.X п.Y):  ✅ PASS  ← NEW
Test 3 (п.Y):      ✅ PASS  ← NEW
Test 4 (Mixed):    ✅ PASS

✅ ALL TESTS PASSED!
```

## Examples

### Example 1: "ч.X п.Y" Format

**Input**:
```json
{
  "text": "Згідно з ч.1 п.5 КК України підозрюваний притягується",
  "triplets": [{
    "legal_reference": "ч.1 п.5 КК України"
  }]
}
```

**Output** (Consistent):
```json
{
  "text": "Згідно з ч.1 п.1 КК України підозрюваний притягується",
  "triplets": [{
    "legal_reference": "ч.1 п.1 КК України"  ← Matches text!
  }]
}
```

### Example 2: Standalone "п.Y"

**Input**:
```json
{
  "text": "Відповідно до п.12 КК України застосовується міра покарання",
  "triplets": [{
    "legal_reference": "п.12 КК України"
  }]
}
```

**Output** (Consistent):
```json
{
  "text": "Відповідно до п.15 КК України застосовується міра покарання",
  "triplets": [{
    "legal_reference": "п.15 КК України"  ← Matches text!
  }]
}
```

### Example 3: Mixed Formats

**Input**:
```json
{
  "text": "За ч.2 ст.185 КК України та п.12 КК України",
  "triplets": [
    {"legal_reference": "ч.2 ст.185 КК України"},
    {"legal_reference": "п.12 КК України"}
  ]
}
```

**Output** (Consistent):
```json
{
  "text": "За ч.2 ст.180 КК України та п.12 КК України",
  "triplets": [
    {"legal_reference": "ч.2 ст.180 КК України"},  ← Matches!
    {"legal_reference": "п.12 КК України"}         ← Matches!
  ]
}
```

## Files Modified

### `src/legal_graph/processors/negative_sampling/__init__.py`

**Added patterns**:
- `self.part_punkt_pattern` - for "ч.X п.Y" format
- `self.punkt_pattern` - for standalone "п.Y" format

**Updated methods**:
- `can_corrupt()` - checks for punkt patterns
- `_modify_text()` - processes punkt patterns with consistency

**Processing order**:
1. Process "ч.X ст.Y" patterns
2. Process "ч.X п.Y" patterns (**NEW**)
3. If no previous matches, process standalone patterns:
   - "ст.Y" patterns
   - "п.Y" patterns (**NEW**)

## Backward Compatibility

✅ **Fully backward compatible**
- All existing "ст." format processing unchanged
- New "п." format processing added separately
- No impact on existing functionality

## Coverage

The implementation now covers **all major legal reference formats** used in Ukrainian documents:

| Format | Example | Supported | Notes |
|--------|---------|-----------|-------|
| ч.X ст.Y | ч.2 ст.185 КК України | ✅ | Most common |
| ст.Y | ст.185 КК України | ✅ | Article only |
| ч.X п.Y | ч.1 п.5 КК України | ✅ | **NEW** |
| п.Y | п.12 КК України | ✅ | **NEW** |
| ч.X | ч.2 | ✅ | Standalone part |

## Benefits

1. **Complete coverage** - handles all common legal reference formats
2. **Consistency maintained** - text and triplets always match
3. **Better training data** - more diverse negative samples
4. **Real-world ready** - matches actual document usage

## Testing

Comprehensive test suite in `test_punkt_support.py`:
- Tests all format variations
- Verifies text-triplet consistency
- Checks mixed format documents
- All tests passing ✅

## Related Documentation

- `CONSISTENCY_FIX.md` - Text-triplet consistency implementation
- `NEGATIVE_SAMPLING_FIX_SUMMARY.md` - Complete fix overview
- `test_punkt_support.py` - Comprehensive test suite

