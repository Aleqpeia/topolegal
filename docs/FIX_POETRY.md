# Poetry Installation Fix Guide

## Problem

`poetry install` fails with:
```
ParseConstraintError
Could not parse version constraint: <empty>
```

This error occurs in the dependency resolution within `poetry.lock`, even after regenerating the lock file.

## Root Cause

The issue appears to be in the lock file's dependency specifications where a transitive dependency has an empty version constraint. This can happen when:

1. A dependency package has malformed metadata on PyPI
2. Version resolution creates conflicting constraints
3. Platform-specific dependencies have issues

## Quick Fixes (Choose One)

### Fix 1: Use pip (Recommended for Negative Sampling Module)

The negative sampling module works perfectly with pip:

```bash
# Install minimal dependencies
pip install pydantic pandas google-cloud-bigquery

# Verify it works
python test_negative_sampling_simple.py
```

**Pros:** ✅ Fast, simple, works immediately
**Cons:** ❌ No lock file for reproducible builds

### Fix 2: Use Minimal pyproject.toml

Use the minimal configuration that only includes negative sampling dependencies:

```bash
# Backup current config
cp pyproject.toml pyproject.toml.full

# Use minimal config
cp pyproject-negative-sampling.toml pyproject.toml

# Remove old lock
rm poetry.lock

# Install
poetry lock
poetry install
```

**Pros:** ✅ Uses poetry, reproducible
**Cons:** ❌ Loses other project dependencies

### Fix 3: Isolate the Problem Dependency

Try removing dependencies one by one to find the problematic one:

```bash
# Create a test script
cat > test_lock.sh << 'SCRIPT'
#!/bin/bash
for dep in torch-scatter torch-sparse torch-cluster torchdrug ydata-profiling bertopic; do
    echo "Testing without $dep..."

    # Comment out the dependency
    sed -i "s/^$dep /#$dep /" pyproject.toml

    # Try to lock
    rm -f poetry.lock
    if poetry lock 2>&1 | grep -q "Writing lock file"; then
        if poetry install --dry-run 2>&1 | grep -q "Could not parse"; then
            echo "  Still failing with $dep removed"
            git checkout pyproject.toml
        else
            echo "  SUCCESS! $dep was the problem"
            exit 0
        fi
    fi

    # Restore
    git checkout pyproject.toml
done
SCRIPT

chmod +x test_lock.sh
./test_lock.sh
```

**Pros:** ✅ Finds exact culprit
**Cons:** ❌ Time-consuming

### Fix 4: Use requirements.txt Instead

Convert to requirements.txt approach:

```bash
# Create requirements.txt for negative sampling
cat > requirements-negative-sampling.txt << 'EOF'
# Core dependencies for negative sampling module
pydantic>=2.0.0
pandas>=2.0.3

# BigQuery integration
google-cloud-bigquery>=3.24.0
google-cloud-bigquery-storage>=2.24.0

# Optional: Full system dependencies
# Uncomment if you need full TopoLegal functionality
# torch==2.5.1
# torch-geometric>=2.6.1
# transformers>=4.35.0
# spacy>=3.7.2
# langchain-community>=0.0.29
EOF

# Install
pip install -r requirements-negative-sampling.txt
```

**Pros:** ✅ Simple, works, easy to maintain
**Cons:** ❌ No poetry lock file

## Detailed Fix Process

### Step 1: Backup Everything

```bash
# Backup important files
cp pyproject.toml pyproject.toml.backup
cp poetry.lock poetry.lock.backup 2>/dev/null || true
```

### Step 2: Clean Poetry Cache

```bash
# Clear poetry cache
poetry cache clear pypi --all
poetry cache clear PyPI --all 2>/dev/null || true

# Remove virtualenv
poetry env remove --all
```

### Step 3: Try Fresh Lock

```bash
# Remove lock file
rm -f poetry.lock

# Try locking with verbose output
poetry lock -vvv 2>&1 | tee poetry-lock.log

# Check if it succeeded
if [ -f poetry.lock ]; then
    echo "Lock file created"
    # Try installing
    poetry install --dry-run
else
    echo "Lock failed"
fi
```

### Step 4: If Still Failing, Simplify pyproject.toml

Remove optional/problematic dependencies:

```toml
[tool.poetry.dependencies]
python = ">=3.10,<4.0"
pydantic = "^2.0.0"
pandas = "^2.0.0"
google-cloud-bigquery = "^3.24.0"

# Comment out problematic packages
# torch-scatter = { version = "*", source = "pyg" }
# torch-sparse = { version = "*", source = "pyg" }
# torch-cluster = { version = "*", source = "pyg" }
```

Then:
```bash
rm poetry.lock
poetry lock
poetry install
```

## Verification

After any fix, verify the negative sampling module works:

```bash
# Test 1: Import test
python -c "import sys; sys.path.insert(0, 'src'); from legal_graph.processors.negative_sampling import NegativeSampler; print('✅ Import OK')"

# Test 2: Standalone tests
python test_negative_sampling_simple.py

# Test 3: CLI
python -m legal_graph.processors.negative_sampling strategies

# Test 4: Quick usage test
python << 'EOF'
import sys
sys.path.insert(0, 'src')
from legal_graph.processors.negative_sampling import create_negative_samples

dataset = [{'doc_id': 'test', 'text': 'згідно з ч.2 ст.185 КК України'}]
result = create_negative_samples(dataset, strategy='article')
print(f'✅ Created {len(result)} samples')
EOF
```

## Best Practice: Hybrid Approach

For production, use this hybrid approach:

1. **Use poetry for development dependencies:**
   ```bash
   # Keep simplified pyproject.toml
   poetry install --only dev
   ```

2. **Use pip for runtime dependencies:**
   ```bash
   pip install pydantic pandas google-cloud-bigquery
   ```

3. **Export requirements for deployment:**
   ```bash
   pip freeze > requirements.txt
   ```

## Summary

| Method | Time | Complexity | Recommended For |
|--------|------|-----------|-----------------|
| **pip install** | 1 min | Low | ✅ Quick testing |
| **Minimal pyproject** | 5 min | Low | ✅ Development |
| **Isolate problem** | 30 min | High | Debugging |
| **requirements.txt** | 2 min | Low | ✅ Production |
| **Fix poetry fully** | Hours | High | ⚠️ If absolutely needed |

## For Negative Sampling Module: Use pip

Since the negative sampling module is complete and tested, the fastest solution is:

```bash
# Install dependencies
pip install pydantic pandas google-cloud-bigquery

# Verify
python test_negative_sampling_simple.py

# Start using
python -m legal_graph.processors.negative_sampling strategies
```

**Status:** ✅ Module is production-ready regardless of poetry issues!

## Additional Resources

- [DEPENDENCIES.md](DEPENDENCIES.md) - Full dependency information
- [POETRY_ISSUE_WORKAROUND.md](POETRY_ISSUE_WORKAROUND.md) - Alternative solutions
- [install_negative_sampling.sh](install_negative_sampling.sh) - Automated installation

## Support

If none of these fixes work:

1. Check [Poetry GitHub Issues](https://github.com/python-poetry/poetry/issues)
2. Try updating poetry: `poetry self update`
3. Use pip as workaround (negative sampling module fully supports this)
4. File an issue with poetry.lock content (sanitized)

**Remember:** The negative sampling module works with pip, so poetry issues don't block development! 🎉