# Next Steps After Refactoring

## Current State

✅ **Branch:** `processors-refactor`
✅ **Status:** Refactoring complete, ready for adoption
✅ **Commit:** 40485a5 - "Refactor processors module: centralize BigQuery utilities"

## What Was Accomplished

1. **Centralized BigQuery Utilities** - Created `src/legal_graph/processors/bigquery_utils.py` with:
   - `BigQueryClient` class for unified operations
   - `BigQueryBatchProcessor` for batch processing
   - Utility functions for schema management and JSON parsing

2. **Organized Documentation** - Moved 30+ files to `docs/` folder:
   - All markdown guides and summaries
   - BigQuery scripts and examples
   - Test scripts and installation scripts
   - Backup configuration files

3. **Updated Module Structure** - Enhanced `processors/__init__.py`:
   - Added BigQuery utilities exports
   - Updated module status tracking
   - Maintained backward compatibility

## Immediate Next Steps

### 1. Review and Test

```bash
# Check out the branch
git checkout processors-refactor

# Test the new utilities
python -c "
from legal_graph.processors import get_module_status
print(get_module_status())
"

# Test BigQuery client
python -c "
from legal_graph.processors import BigQueryClient
client = BigQueryClient()
print(f'Connected to: {client.client.project}')
"
```

### 2. Optional: Refactor Child Modules

Each child module can now be refactored to use the centralized utilities. This is **optional** and can be done incrementally.

#### Example: Refactoring `negative_sampling/__main__.py`

**Before (current approach):**
```python
try:
    from google.cloud import bigquery
except ImportError:
    logger.error("Install google-cloud-bigquery")
    return

if key_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
if project:
    os.environ["GOOGLE_CLOUD_PROJECT"] = project

client = bigquery.Client(
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location="us-central1"
)

# Batch processing code...
while True:
    sql = f"SELECT ... LIMIT {batch_size} OFFSET {processed}"
    rows = client.query(sql).result().to_dataframe()
    # ... process rows ...
```

**After (using centralized utilities):**
```python
from ..bigquery_utils import (
    BigQueryClient,
    BigQueryBatchProcessor,
    parse_json_column,
    create_legal_document_table
)

# Simple initialization
client = BigQueryClient(project=project, key_path=key_path)

# Easy batch processing
def process_negative_samples(batch_data):
    dataset = []
    for row in batch_data:
        triplets = parse_json_column(row.triplets)
        entities = parse_json_column(row.tags)

        if triplets:
            dataset.append({
                'doc_id': str(row.doc_id),
                'text': str(row.text),
                'entities': entities,
                'triplets': triplets
            })

    return sampler.create_negative_samples(dataset)

# Process table
processor = BigQueryBatchProcessor(client, batch_size=batch_size)
stats = processor.process_table(
    table_id=input_table,
    processor_func=process_negative_samples,
    output_table_id=output_table,
    where_clause="triplets IS NOT NULL AND triplets_count > 0"
)
```

**Benefits:**
- 50-100 lines of code reduction per module
- Consistent error handling
- Better maintainability
- Unified authentication

#### Modules to Refactor (Optional)

1. **`negative_sampling/__main__.py`**
   - Function: `sample_bigquery_data()`
   - Lines to reduce: ~150 → ~50

2. **`graph_extraction/__main__.py`**
   - Functions: `run_bigquery()`, `check_bigquery_schema()`, `get_bigquery_stats()`
   - Lines to reduce: ~200 → ~80

3. **`balance/__main__.py`**
   - Functions: `run_bigquery()`, `corrupt_bigquery_data()`, `check_bigquery_schema()`, `get_bigquery_stats()`
   - Lines to reduce: ~250 → ~100

4. **`entities/__main__.py`**
   - Function: `run_bigquery()`
   - Lines to reduce: ~80 → ~30

### 3. Merge to Main Branch

Once tested and reviewed:

```bash
# Option 1: Merge to refactor branch first
git checkout refactor
git merge processors-refactor

# Option 2: Merge directly to main
git checkout main
git merge processors-refactor

# Push changes
git push origin processors-refactor  # For PR/review
# or
git push origin refactor  # If merging to refactor
```

### 4. Update Documentation

If merging to main, update:
- README.md to mention the new utilities
- Add usage examples
- Update architecture diagrams if needed

## Future Enhancements

### Phase 2: Additional Utilities

Consider creating more shared utilities:

1. **`sql_templates.py`** - Common SQL query templates
   ```python
   def get_documents_with_triplets_sql(table_id, limit, offset):
       return f"""
       SELECT doc_id, text, tags, triplets, triplets_count
       FROM `{table_id}`
       WHERE triplets IS NOT NULL
         AND triplets_count > 0
       LIMIT {limit} OFFSET {offset}
       """
   ```

2. **`validation.py`** - Data validation utilities
   ```python
   def validate_legal_document(doc: Dict) -> tuple[bool, List[str]]:
       """Validate legal document structure"""
       errors = []
       if 'doc_id' not in doc:
           errors.append("Missing doc_id")
       # ... more validation
       return len(errors) == 0, errors
   ```

3. **`config.py`** - Shared configuration management
   ```python
   class ProcessorsConfig:
       """Centralized configuration for all processors"""
       DEFAULT_BATCH_SIZE = 100
       DEFAULT_LOCATION = "us-central1"
       # ...
   ```

### Phase 3: Testing

Add comprehensive tests:

```bash
# Create tests directory
mkdir -p src/legal_graph/processors/tests

# Add test files
# - test_bigquery_utils.py
# - test_integration.py
# - test_batch_processor.py
```

### Phase 4: Performance Optimization

1. Add caching for frequently accessed data
2. Implement connection pooling
3. Add query optimization utilities
4. Create performance monitoring

## Documentation Locations

- **Main Guide:** `REFACTORING_SUMMARY.md` - Complete refactoring details
- **User Guide:** `CLAUDE.md` - Updated with usage instructions
- **Examples:** `docs/` - All example scripts and guides
- **This File:** `NEXT_STEPS.md` - You are here!

## Questions?

### Q: Do I need to refactor child modules immediately?
**A:** No, it's optional. The current modules continue to work. Refactoring provides benefits but isn't required.

### Q: Will this break existing code?
**A:** No, all changes are backward compatible. Child modules can use the new utilities at their own pace.

### Q: How do I test the new utilities?
**A:** See "Review and Test" section above. Simple import tests confirm availability.

### Q: Can I merge this to main now?
**A:** Yes, after testing. The refactoring is complete and safe to merge.

### Q: What if I find issues?
**A:** The utilities are extensible. You can add more methods or create issues for enhancement.

## Summary

This refactoring provides a solid foundation for the processors module. The centralized BigQuery utilities reduce code duplication and improve maintainability. Child modules can gradually adopt the new utilities as needed, or continue using their existing approaches.

**The refactoring is complete and ready for use! 🎉**
