# Negative Sampling Module Implementation Summary

## Overview

Successfully implemented a comprehensive negative sampling module for the TopoLegal project that creates "hard negative" samples by intelligently modifying legal reference components in Ukrainian legal documents.

**Implementation Date:** 2025-10-12

## What Was Implemented

### 1. Module Structure

Created new module at: `src/legal_graph/processors/negative_sampling/`

```
negative_sampling/
├── __init__.py          (20KB) - Main module with NegativeSampler class
├── __main__.py          (19KB) - CLI interface for command-line usage
├── strategies.py        (13KB) - Strategy pattern implementations
└── README.md            (13KB) - Comprehensive documentation
```

### 2. Core Components

#### NegativeSampler Class (`__init__.py`)

Main class for creating negative samples with fine-grained control:

- **Base Class**: Extends `DataCorruptor` from balance module for consistency
- **Strategy Support**: Accepts string names or custom `SamplingStrategy` instances
- **Component-level Modifications**: Modifies article numbers (ст.#) and/or part numbers (ч.#)
- **Flexible Configuration**: Configurable sampling probability, article ranges, and format preservation
- **Statistics Tracking**: Detailed tracking of modifications by type

**Key Methods:**
- `create_negative_samples(dataset, label_column)` - Create negative samples
- `get_sampling_stats()` - Get detailed statistics
- `corrupt_sample(sample)` - Process individual sample (overrides base class)

#### Sampling Strategies (`strategies.py`)

Implemented 8 different sampling strategies:

1. **ArticleOffsetStrategy** - Modify article numbers by small offset (±1-5)
   - Example: `ст.185` → `ст.187`
   - Use case: Near-miss negatives

2. **PartOffsetStrategy** - Modify part numbers by small offset (±1-2)
   - Example: `ч.2` → `ч.3`
   - Use case: Different parts of same article

3. **CombinedOffsetStrategy** - Modify both article and part
   - Example: `ч.2 ст.185` → `ч.3 ст.187`
   - Use case: More diverse negatives

4. **IndexSwapStrategy** - Swap article and part numbers
   - Example: `ч.2 ст.185` → `ч.185 ст.2` (invalid)
   - Use case: Detect structurally invalid references

5. **RandomComponentStrategy** - Random valid components
   - Example: `ч.2 ст.185` → `ч.7 ст.341`
   - Use case: Maximum diversity

6. **MixedStrategy** - Combines multiple strategies
   - Default strategy with balanced weights
   - Use case: General-purpose sampling

7. **ConservativeStrategy** - Small offsets only
   - 70% article (±1-3), 30% part (±1)
   - Use case: Subtle modifications

8. **AggressiveStrategy** - All strategies with equal weights
   - Maximum diversity and complexity
   - Use case: Training robust models

#### CLI Interface (`__main__.py`)

Complete command-line interface supporting:

**Commands:**
- `sample-csv` - Process CSV files
- `sample-json` - Process JSON files
- `sample-bigquery` - Process BigQuery tables
- `strategies` - Show available strategies

**Options:**
- `--strategy` - Choose sampling strategy
- `--probability` - Set sampling probability (0.0-1.0)
- `--label-column` - Specify label column name
- `--batch-size` - Batch processing size (BigQuery)
- `--project`, `--key-path` - Cloud authentication

**Example Usage:**
```bash
# CSV processing
python -m legal_graph.processors.negative_sampling sample-csv \
    input.csv output.csv --strategy mixed --probability 0.5

# BigQuery processing
python -m legal_graph.processors.negative_sampling sample-bigquery \
    project.dataset.input_table project.dataset.output_table \
    --strategy aggressive --probability 0.7
```

### 3. Integration with Existing Code

#### Updated `processors/__init__.py`

Added complete integration:

1. **Import Section** (lines 84-122):
   - Import all negative sampling classes and functions
   - Graceful fallback if module unavailable
   - Set `NEGATIVE_SAMPLING_AVAILABLE` flag

2. **DocumentProcessor Class** (lines 138-215):
   - Added `enable_negative_sampling` parameter
   - Added `sampling_strategy` parameter
   - Added `sampling_probability` parameter
   - Initialize `negative_sampler` attribute
   - Added `create_negative_samples()` method

3. **Module Status** (lines 420-428):
   - Added `negative_sampling_available` to status dict
   - Updated `all_modules_available` check

4. **Exports** (lines 516-564):
   - Added all 13 negative sampling exports
   - Properly documented each section

### 4. Documentation

#### README.md (13KB)

Comprehensive documentation including:

- **Overview**: Purpose and key differences from balance module
- **Quick Start**: Python API and CLI examples
- **Sampling Strategies**: Detailed description of all 8 strategies
- **API Reference**: Complete parameter documentation
- **Supported Legal Codes**: All Ukrainian legal codes with ranges
- **Examples**: 4 detailed usage examples:
  1. Basic CSV usage
  2. Custom strategy creation
  3. PyTorch integration
  4. BigQuery integration
- **Statistics and Monitoring**: How to track modifications
- **Advanced Topics**: Custom strategies, filtering
- **Troubleshooting**: Common issues and solutions

## Technical Details

### Regex Patterns

Three main patterns for legal reference detection:

1. **Part + Article Pattern**: `ч.X ст.Y КК України`
   ```python
   r'(ч\.\s*)(\d+)(\s+ст\.\s*)(\d+)(\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)'
   ```

2. **Article Pattern**: `ст.Y КК України` (with optional part in suffix)
   ```python
   r'(ст\.\s*)(\d+)(\s*(?:ч\.\s*\d+)?\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)'
   ```

3. **Part Only Pattern**: `ч.X` or `п.X`
   ```python
   r'(ч\.|п\.)(\s*)(\d+)'
   ```

### Article Ranges

Configured for all major Ukrainian legal codes:

- **КК України** (Criminal Code): 1-446
- **КПК України** (Criminal Procedure Code): 1-615
- **ЦК України** (Civil Code): 1-1308
- **КоАП України** (Administrative Code): 1-328
- **СК України** (Family Code): 1-270
- **КЗпП України** (Labor Code): 1-265

### Processing Flow

1. **Input**: Dataset with documents containing legal references
2. **Detection**: Regex patterns identify modifiable references
3. **Strategy Selection**: Choose how to modify (offset, swap, random, etc.)
4. **Validation**: Ensure modified values are within valid ranges
5. **Application**: Apply modifications to text, triplets, and entities
6. **Labeling**: Mark samples as valid (original) or invalid (negative)
7. **Output**: Return dataset with labels and statistics

## Key Features

### 1. Component-Level Granularity

Unlike the balance module which replaces entire article numbers randomly:

```python
# Balance module (coarse-grained)
"ст.537 КПК" → "ст.191 КПК"  # Completely different article

# Negative sampling (fine-grained)
"ч.2 ст.185 КК" → "ч.3 ст.185 КК"  # Same article, different part
"ч.2 ст.185 КК" → "ч.2 ст.187 КК"  # Nearby article
"ч.2 ст.185 КК" → "ч.3 ст.187 КК"  # Both modified
```

### 2. Strategy Pattern

Extensible design allows easy addition of new strategies:

```python
class CustomStrategy(SamplingStrategy):
    def modify_reference(self, article, part, code_name,
                         min_article, max_article, max_part=10):
        # Your custom logic
        return new_article, new_part, "modification_type"

    def get_strategy_name(self):
        return "CustomStrategy"
```

### 3. Hard Negative Mining

Creates challenging samples for training:

- **Near-miss samples**: Similar but incorrect references
- **Syntactically invalid**: Swapped components for structure testing
- **Controlled difficulty**: Adjustable offset ranges

### 4. Comprehensive Statistics

Track everything:

```python
stats = sampler.get_sampling_stats()
# {
#     'total_processed': 1000,
#     'corrupted': 502,
#     'skipped': 498,
#     'errors': 0,
#     'modifications_by_type': {
#         'article_offset_+2': 145,
#         'part_offset_+1': 67,
#         'combined_article_offset_+3_part_offset_-1': 23,
#         'random_component': 41
#     },
#     'strategy': 'MixedStrategy'
# }
```

## Usage Examples

### Example 1: Simple Python API

```python
from legal_graph.processors.negative_sampling import create_negative_samples

dataset = [
    {'doc_id': '001', 'text': 'згідно з ч.2 ст.185 КК України', ...}
]

negative_samples = create_negative_samples(
    dataset,
    strategy='article',
    sampling_probability=0.5
)

for sample in negative_samples:
    print(f"Valid: {sample['is_valid']}, Text: {sample['text']}")
```

### Example 2: DocumentProcessor Integration

```python
from legal_graph.processors import DocumentProcessor

processor = DocumentProcessor(
    enable_negative_sampling=True,
    sampling_strategy='aggressive',
    sampling_probability=0.7
)

dataset = load_legal_documents()
negative_samples = processor.create_negative_samples(dataset)
```

### Example 3: Custom Strategy

```python
from legal_graph.processors.negative_sampling import (
    NegativeSampler,
    MixedStrategy,
    ArticleOffsetStrategy,
    PartOffsetStrategy
)

custom_strategy = MixedStrategy(
    strategies=[
        ArticleOffsetStrategy(offset_range=(1, 3)),
        PartOffsetStrategy(offset_range=(1, 1))
    ],
    weights=[0.8, 0.2]
)

sampler = NegativeSampler(strategy=custom_strategy)
```

### Example 4: CLI Usage

```bash
# Show available strategies
python -m legal_graph.processors.negative_sampling strategies

# Process CSV with article strategy
python -m legal_graph.processors.negative_sampling sample-csv \
    input.csv output.csv \
    --strategy article \
    --probability 0.5

# Process BigQuery with aggressive strategy
python -m legal_graph.processors.negative_sampling sample-bigquery \
    project.dataset.input_table \
    project.dataset.output_table \
    --strategy aggressive \
    --probability 0.7 \
    --batch-size 100
```

## Testing

Created comprehensive test script: `test_negative_sampling.py`

Tests include:
1. **Import Test**: Verify all imports work
2. **Basic Sampling Test**: Test negative sample creation
3. **Strategy Test**: Test all strategy types
4. **CLI Test**: Test command-line interface

Run tests:
```bash
python test_negative_sampling.py
```

## Comparison with Balance Module

| Aspect | Balance Module | Negative Sampling Module |
|--------|---------------|-------------------------|
| **Granularity** | Full article replacement | Component-level (article, part, or both) |
| **Strategy** | Random from range | 8 different strategies (offset, swap, random, mixed) |
| **Control** | Low (random only) | High (configurable strategies and ranges) |
| **Validity** | Always valid | Can create invalid (for structure testing) |
| **Use Case** | Binary classification | Hard negative mining for GNN training |
| **Difficulty** | Easy negatives | Configurable difficulty (conservative to aggressive) |
| **Example** | `ст.537` → `ст.191` | `ч.2 ст.185` → `ч.3 ст.187` |
| **Code Size** | ~400 lines | ~1200 lines (including strategies) |
| **Strategies** | 1 (random) | 8 (extensible) |

## Benefits

1. **Improved Model Training**: Hard negatives improve GNN discrimination ability
2. **Flexibility**: Multiple strategies for different training scenarios
3. **Granularity**: Fine-grained control over modifications
4. **Extensibility**: Easy to add new strategies
5. **Integration**: Seamlessly integrated with existing processor infrastructure
6. **Documentation**: Comprehensive docs with examples
7. **CLI**: Full command-line interface for batch processing
8. **Statistics**: Detailed tracking for analysis

## Future Enhancements

Potential improvements for future development:

1. **Machine Learning-Based Strategy**: Use model confidence to guide sampling
2. **Cross-Code Swapping**: Swap references between different legal codes
3. **Temporal Modifications**: Modify date-based references
4. **Multi-Level Modifications**: Handle nested legal structures (ст. X ч. Y п. Z)
5. **Adversarial Sampling**: Generate samples specifically to fool model
6. **Adaptive Difficulty**: Automatically adjust difficulty based on model performance
7. **Reference Graph Analysis**: Use graph structure to guide modifications

## Files Created

1. `src/legal_graph/processors/negative_sampling/__init__.py` (20KB)
2. `src/legal_graph/processors/negative_sampling/__main__.py` (19KB)
3. `src/legal_graph/processors/negative_sampling/strategies.py` (13KB)
4. `src/legal_graph/processors/negative_sampling/README.md` (13KB)
5. `test_negative_sampling.py` (4KB)
6. `NEGATIVE_SAMPLING_IMPLEMENTATION.md` (this file)

**Total Code**: ~65KB of implementation + documentation

## Files Modified

1. `src/legal_graph/processors/__init__.py`
   - Added negative sampling imports (lines 84-122)
   - Updated DocumentProcessor class (lines 138-215)
   - Added create_negative_samples() method (lines 336-355)
   - Updated get_capabilities() (lines 357-365)
   - Updated get_module_status() (lines 420-428)
   - Updated __all__ exports (lines 516-564)

## Conclusion

Successfully implemented a production-ready negative sampling module that:

✅ Creates component-level modifications of legal references
✅ Provides 8 different sampling strategies
✅ Integrates seamlessly with existing processor infrastructure
✅ Includes comprehensive CLI interface
✅ Has detailed documentation and examples
✅ Tracks detailed statistics
✅ Supports CSV, JSON, and BigQuery data sources
✅ Follows existing code patterns and conventions

The module is ready for use in training graph neural networks for legal document validation tasks, providing "hard negative" samples that will improve model discrimination ability.
