# Critical Fix: Text-Triplet Consistency

## The Problem

The original implementation modified text and triplets **independently**, causing critical data inconsistency:

```python
# Original (BROKEN):
# Step 1: Modify text
text = "ч.2 ст.185 КК України" → "ч.2 ст.187 КК України"

# Step 2: Modify triplets INDEPENDENTLY (with random strategy)
triplet['legal_reference'] = "ч.2 ст.185 КК України" → "ч.2 ст.190 КК України"

# RESULT: INCONSISTENT DATA!
# Text says: "ч.2 ст.187"
# Triplet says: "ч.2 ст.190"  ← DIFFERENT!
```

### Why This Is Critical

1. **Model Confusion**: The model learns from inconsistent examples where text and structured data don't match
2. **Invalid Training Data**: A negative sample should be "wrong but plausible" - not internally contradictory
3. **Downstream Errors**: Systems using these triplets will make incorrect predictions based on mismatched data

## The Solution

**Apply modifications consistently**: When a legal reference is modified in text, apply the **exact same modification** to all matching references in triplets and entities.

### Implementation

```python
# New (FIXED):
# Step 1: Modify text and collect the reference mapping
text, modifications = modify_text(text)
# Creates mapping: {"ч.2 ст.185 КК України": "ч.2 ст.187 КК України"}

# Step 2: Apply the SAME mapping to triplets
triplets = apply_mapping(triplets, mapping)

# Step 3: Apply the SAME mapping to entities
entities = apply_mapping(entities, mapping)

# RESULT: CONSISTENT DATA!
# Text: "ч.2 ст.187 КК України"
# Triplet: "ч.2 ст.187 КК України"  ← MATCHES!
# Entity: "ч.2 ст.187 КК України"   ← MATCHES!
```

## Code Changes

### 1. Updated `corrupt_sample()` Method

The main workflow now ensures consistency:

```python
def corrupt_sample(self, sample: Dict) -> CorruptionResult:
    # Step 1: Modify text and collect reference mappings
    sampled_text, text_mods = self._modify_text(sample.get('text', ''))
    
    # Build mapping: original → modified
    reference_mapping = {}
    for mod in text_mods:
        if 'original' in mod and 'modified' in mod:
            reference_mapping[mod['original']] = mod['modified']
    
    # Step 2: Apply SAME modifications to triplets
    sampled_triplets, triplet_mods = self._modify_triplets_with_mapping(
        sample['triplets'], 
        reference_mapping  # ← Use the mapping from text!
    )
    
    # Step 3: Apply SAME modifications to entities
    sampled_entities, entity_mods = self._modify_entities_with_mapping(
        sample['entities'],
        reference_mapping  # ← Use the mapping from text!
    )
```

### 2. New Helper Methods

#### `_apply_reference_mapping(text, mapping)`
Applies a reference mapping to any text, replacing all occurrences of original references with their modified versions.

#### `_modify_triplets_with_mapping(triplets, mapping)`
Applies the reference mapping to all triplet fields (`legal_reference`, `source`, `relation`, `target`).

#### `_modify_entities_with_mapping(entities, mapping)`
Applies the reference mapping to entity text fields.

## Test Results

All consistency tests pass:

```bash
$ poetry run python test_consistency_fix.py

Testing text-triplet consistency...
📝 Original Document:
   Text: Підозрюваний обвинувачується за ч.2 ст.185 КК України
   Triplet legal_reference: ч.2 ст.185 КК України

📝 Modified Document:
   Text: Підозрюваний обвинувачується за ч.2 ст.183 КК України
   Triplet legal_reference: ч.2 ст.183 КК України

🔍 Consistency Check:
   Text reference:           ч.2 ст.183 КК України
   Triplet legal_reference:  ч.2 ст.183 КК України
✅ CONSISTENCY VERIFIED!

Testing multiple references consistency...
📝 Original: ч.2 ст.185 КК України та ч.1 ст.190 КК України
📝 Modified: ч.2 ст.183 КК України та ч.1 ст.191 КК України
   ✅ 'ч.2 ст.183 КК України' found in text
   ✅ 'ч.1 ст.191 КК України' found in text
✅ CONSISTENCY VERIFIED for all references!

✅ ALL CONSISTENCY TESTS PASSED!
```

## Examples

### Example 1: Single Reference

**Input**:
```json
{
  "text": "Підозрюваний обвинувачується за ч.2 ст.185 КК України",
  "triplets": [{
    "legal_reference": "ч.2 ст.185 КК України",
    "relation": "обвинувачується_за"
  }]
}
```

**Output** (Consistent):
```json
{
  "text": "Підозрюваний обвинувачується за ч.2 ст.187 КК України",
  "triplets": [{
    "legal_reference": "ч.2 ст.187 КК України",  ← Same modification!
    "relation": "обвинувачується_за"
  }]
}
```

### Example 2: Multiple References

**Input**:
```json
{
  "text": "Обвинувачується за ч.2 ст.185 та ч.1 ст.190 КК України",
  "triplets": [
    {"legal_reference": "ч.2 ст.185 КК України"},
    {"legal_reference": "ч.1 ст.190 КК України"}
  ]
}
```

**Output** (Consistent):
```json
{
  "text": "Обвинувачується за ч.2 ст.183 та ч.1 ст.191 КК України",
  "triplets": [
    {"legal_reference": "ч.2 ст.183 КК України"},  ← Matches text!
    {"legal_reference": "ч.1 ст.191 КК України"}   ← Matches text!
  ]
}
```

## Benefits

1. **Data Integrity**: Text and triplets always match
2. **Model Quality**: Training data is internally consistent
3. **Debuggability**: Easy to verify consistency by comparing text and triplets
4. **Auditability**: Reference mapping is tracked in modification details

## Impact on Existing Data

This fix affects **all negative sampling operations** going forward:
- CSV input: ✅ Consistency enforced
- JSON input: ✅ Consistency enforced
- BigQuery input: ✅ Consistency enforced

### Action Required

If you have **already generated negative samples** with the old inconsistent method:
1. **Regenerate them** using the fixed implementation
2. The old data may have text-triplet mismatches that will confuse your model

## Related Documentation

- `NEGATIVE_SAMPLING_FIX_SUMMARY.md` - Overall summary of all fixes
- `NUMPY_ARRAY_FIX.md` - BigQuery numpy array handling
- `BUGFIX_MULTIPLE_REFS.md` - Multiple reference handling
- `test_consistency_fix.py` - Comprehensive test suite

## Technical Details

### Reference Mapping Format

The mapping tracks all modifications:

```python
{
  "ч.2 ст.185 КК України": "ч.2 ст.187 КК України",
  "ч.1 ст.190 КК України": "ч.1 ст.191 КК України"
}
```

This mapping is:
- Built during text modification
- Applied to triplets
- Applied to entities
- Stored in corruption_details for debugging

### Edge Cases Handled

1. **No modifications**: If text has no legal references, triplets remain unchanged
2. **Multiple occurrences**: All occurrences of a reference are replaced consistently
3. **Overlapping references**: Longer references are matched first to avoid partial replacements
4. **Different formats**: Handles both string and list formats for triplets/entities

## Verification

To verify consistency in your generated data:

```python
# Check a generated sample
sample = negative_dataset[0]

# Extract references from text
text_refs = extract_references(sample['text'])

# Extract references from triplets
triplet_refs = [t['legal_reference'] for t in sample['triplets']]

# Verify they match
assert all(ref in sample['text'] for ref in triplet_refs), \
    "Triplet references must appear in text!"
```

