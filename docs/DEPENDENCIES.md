# Dependencies for Negative Sampling Module

## Required Dependencies

The negative sampling module has minimal dependencies and can work with or without the full TopoLegal installation.

### Core Dependencies (Required)

These are needed for the negative sampling module to function:

```bash
pip install pydantic  # For data models (BaseModel, CorruptionResult)
```

### Optional Dependencies

#### For BigQuery Integration:
```bash
pip install google-cloud-bigquery  # BigQuery client
pip install pandas                  # Data processing
```

#### For Full TopoLegal Integration:
```bash
pip install torch torch-geometric  # GNN models
pip install transformers           # Text processing
pip install spacy                  # NER (optional)
```

## Installation Options

### Option 1: Minimal (Negative Sampling Only)

```bash
pip install pydantic
```

This allows you to use:
- `NegativeSampler` class
- All 8 sampling strategies
- CSV/JSON processing
- Python API

### Option 2: With BigQuery

```bash
pip install pydantic google-cloud-bigquery pandas
```

This adds:
- BigQuery table processing
- CLI `sample-bigquery` command
- Large-scale batch processing

### Option 3: Full Installation (Poetry)

If you need the full TopoLegal system:

```bash
# Fix poetry lock file issue
rm poetry.lock
poetry lock
poetry install
```

**Note:** There's currently a poetry lock file issue. If `poetry install` fails with "Could not parse version constraint: <empty>", this is a known issue with the existing lock file.

**Workaround:**
```bash
# Option A: Use pip directly
pip install -e .

# Option B: Install specific dependencies
pip install pydantic google-cloud-bigquery pandas torch torch-geometric

# Option C: Recreate lock file
rm poetry.lock
poetry lock
poetry install
```

## Standalone Usage (No Installation Required)

The negative sampling module can work standalone without installation:

```python
import sys
sys.path.insert(0, 'src')

from legal_graph.processors.negative_sampling import NegativeSampler

# Use the module...
```

## Testing Without Dependencies

The standalone test suite works without any dependencies:

```bash
python test_negative_sampling_simple.py
```

This test validates:
- Regex patterns
- Modification logic
- Text replacement
- Strategy simulation
- BigQuery data format

**Status:** ✅ All 5 tests passed

## Dependency Status

| Component | Dependencies | Status |
|-----------|-------------|--------|
| **Negative Sampling Core** | `pydantic` | ✅ Minimal |
| **Strategies** | None | ✅ No dependencies |
| **CLI (CSV/JSON)** | `pydantic`, `pandas` | ✅ Light |
| **CLI (BigQuery)** | `pydantic`, `google-cloud-bigquery`, `pandas` | ✅ Moderate |
| **Full Integration** | All TopoLegal deps | ⚠️ Poetry issue |

## Known Issues

### Poetry Lock File Issue

**Error:** `Could not parse version constraint: <empty>`

**Cause:** Corrupted or outdated poetry.lock file

**Solutions:**

1. **Remove and regenerate lock:**
   ```bash
   rm poetry.lock
   poetry lock
   ```

2. **Use pip instead:**
   ```bash
   pip install pydantic google-cloud-bigquery pandas
   ```

3. **Manual dependency installation:**
   ```bash
   # Core
   pip install pydantic

   # BigQuery
   pip install google-cloud-bigquery pandas

   # Optional: Full system
   pip install torch torch-geometric transformers spacy
   ```

## Recommendations

### For Development
```bash
# Minimal setup for negative sampling only
pip install pydantic google-cloud-bigquery pandas
```

### For Production
```bash
# Use requirements.txt approach
cat > requirements.txt << EOF
pydantic>=2.0.0
google-cloud-bigquery>=3.24.0
pandas>=2.0.3
EOF

pip install -r requirements.txt
```

### For Testing
```bash
# No installation needed
python test_negative_sampling_simple.py
```

## Verification

After installation, verify the module works:

```bash
# Test imports
python -c "from legal_graph.processors.negative_sampling import NegativeSampler; print('✅ Import successful')"

# Test CLI
python -m legal_graph.processors.negative_sampling strategies

# Run tests
python test_negative_sampling_simple.py
```

## Summary

The negative sampling module is designed to work with minimal dependencies:

- ✅ **Core functionality**: Only needs `pydantic`
- ✅ **BigQuery support**: Add `google-cloud-bigquery` and `pandas`
- ✅ **Standalone tests**: No dependencies required
- ⚠️ **Poetry installation**: Has known lock file issue

**Recommended:** Install minimal dependencies directly with pip rather than using poetry until the lock file issue is resolved.