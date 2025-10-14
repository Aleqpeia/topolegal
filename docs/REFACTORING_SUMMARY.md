# Processors Module Refactoring Summary

## Branch: `processors-refactor`

This document summarizes the refactoring work done to improve the processors module structure and reduce boilerplate code.

## Changes Made

### 1. Root Folder Cleanup

**Moved to `docs/` folder:**
- All markdown documentation files (except README.md and CLAUDE.md)
- BigQuery utility scripts (bigquery_dataset_reformatter.py, etc.)
- Test scripts (test_*.py)
- Installation scripts (install_negative_sampling.sh)
- Backup config files (pyproject-negative-sampling.toml, poetry.lock.backup)

**Result:** Cleaner root directory with only essential files (notebooks, core scripts, config).

### 2. BigQuery Utilities Abstraction

**Created:** `src/legal_graph/processors/bigquery_utils.py`

This new module provides centralized BigQuery utilities to eliminate code duplication across child modules.

#### Key Classes and Functions:

**BigQueryClient**
- Wrapper around `google.cloud.bigquery.Client`
- Handles authentication and project configuration
- Provides common methods:
  - `query()` - Execute SQL queries
  - `query_to_dataframe()` - Query to pandas DataFrame
  - `insert_rows_json()` - Insert data
  - `create_table()` - Create tables
  - `get_table_schema()` - Get schema information
  - `check_required_columns()` - Validate schema
  - `get_table_stats()` - Get table statistics

**BigQueryBatchProcessor**
- Batch processing with progress tracking
- Automatic error handling
- Configurable batch size and logging
- Process entire tables with custom processor functions

**Utility Functions:**
- `create_legal_document_table()` - Create standard legal doc tables
- `check_legal_document_schema()` - Validate table schemas
- `get_triplet_statistics()` - Get triplet statistics from tables
- `parse_json_column()` - Safely parse JSON columns (handles multiple formats)
- `format_table_stats()` - Format statistics for logging

### 3. Module Integration

**Updated:** `src/legal_graph/processors/__init__.py`

- Added imports for BigQuery utilities
- Added `BIGQUERY_UTILS_AVAILABLE` flag
- Exported all BigQuery utilities in `__all__`
- Updated `get_module_status()` to include BigQuery utils status

## Benefits

### Code Reduction
- Eliminates duplicate BigQuery client initialization code
- Centralized schema management
- Shared batch processing logic
- Common JSON parsing utilities

### Consistency
- Uniform error handling across modules
- Consistent logging format
- Standard table schemas
- Unified authentication approach

### Maintainability
- Single source of truth for BigQuery operations
- Easier to update and test
- Better documentation
- Clear module boundaries

## Usage Examples

### Before Refactoring (in each child module):

```python
# Duplicate code in graph_extraction, negative_sampling, balance, etc.
try:
    from google.cloud import bigquery
except ImportError:
    print("Install google-cloud-bigquery")
    return

if key_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
if project:
    os.environ["GOOGLE_CLOUD_PROJECT"] = project

client = bigquery.Client(
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location="us-central1"
)

# Then implement batch processing, error handling, etc...
```

### After Refactoring:

```python
from legal_graph.processors import BigQueryClient, BigQueryBatchProcessor

# Simple initialization
client = BigQueryClient(
    project="my-project",
    key_path="/path/to/key.json"
)

# Easy batch processing
processor = BigQueryBatchProcessor(client, batch_size=100)

def process_batch(data):
    # Your processing logic
    return processed_data

stats = processor.process_table(
    table_id="project.dataset.table",
    processor_func=process_batch,
    output_table_id="project.dataset.output"
)
```

### Parsing JSON Columns:

```python
from legal_graph.processors import parse_json_column

# Handles all formats automatically
triplets = parse_json_column(row.triplets)  # Works with str, list, np.ndarray
entities = parse_json_column(row.tags)
```

### Creating Standard Tables:

```python
from legal_graph.processors import create_legal_document_table

table = create_legal_document_table(
    client=client,
    table_id="project.dataset.my_table",
    include_entities=True,
    include_triplets=True,
    include_labels=True,
    label_column="is_valid"
)
```

## Module Structure

```
src/legal_graph/processors/
├── __init__.py                 # Updated with BigQuery exports
├── bigquery_utils.py          # NEW: Centralized BigQuery utilities
├── entities/
│   └── __main__.py            # Can now use parent BigQuery utilities
├── graph_extraction/
│   └── __main__.py            # Can now use parent BigQuery utilities
├── balance/
│   └── __main__.py            # Can now use parent BigQuery utilities
└── negative_sampling/
    └── __main__.py            # Can now use parent BigQuery utilities
```

## Next Steps for Child Modules

Child modules (graph_extraction, balance, negative_sampling, entities) can now refactor their `__main__.py` files to use the centralized utilities:

1. Replace direct `bigquery.Client()` initialization with `BigQueryClient`
2. Use `BigQueryBatchProcessor` for batch operations
3. Use `parse_json_column()` for JSON parsing
4. Use `create_legal_document_table()` for table creation
5. Use `check_legal_document_schema()` for validation
6. Use `format_table_stats()` for logging

## Import Statement

```python
# From child modules
from ..bigquery_utils import (
    BigQueryClient,
    BigQueryBatchProcessor,
    create_legal_document_table,
    check_legal_document_schema,
    parse_json_column,
    format_table_stats
)

# Or from parent level
from legal_graph.processors import (
    BigQueryClient,
    BigQueryBatchProcessor,
    # ... etc
)
```

## Testing

To test the refactored module:

```python
from legal_graph.processors import get_module_status

# Check module availability
status = get_module_status()
print(status['bigquery_utils_available'])  # Should be True

# Test BigQuery client
from legal_graph.processors import BigQueryClient

client = BigQueryClient()
print(f"Connected to project: {client.client.project}")
```

## Documentation Updates

- Updated `CLAUDE.md` to include BigQuery utilities section
- Created this refactoring summary
- All markdown docs moved to `docs/` folder for better organization

## Files Modified

1. Created: `src/legal_graph/processors/bigquery_utils.py`
2. Modified: `src/legal_graph/processors/__init__.py`
3. Created: `REFACTORING_SUMMARY.md` (this file)
4. Organized: Moved 16+ files to `docs/` folder

## Branch Information

- **Branch:** `processors-refactor`
- **Base:** `refactor`
- **Purpose:** Refactor processors module with centralized BigQuery utilities
- **Status:** Ready for review and testing

## Checklist

- [x] Create new branch
- [x] Move documentation to docs folder
- [x] Create bigquery_utils.py module
- [x] Update processors __init__.py
- [x] Add BigQuery utils to exports
- [x] Update module status tracking
- [x] Write refactoring documentation
- [ ] Refactor child modules to use utilities (future work)
- [ ] Add unit tests for BigQuery utilities (future work)
- [ ] Create examples for child modules (future work)

## Future Improvements

1. **Child Module Refactoring:** Update graph_extraction, balance, negative_sampling, and entities __main__.py files to use the new utilities

2. **Additional Utilities:** Consider adding more shared utilities:
   - Common SQL query templates
   - Standard data validation functions
   - Shared configuration management
   - Common error types and handling

3. **Testing:** Add comprehensive tests for BigQuery utilities

4. **Documentation:** Add more usage examples and best practices

5. **Performance:** Add caching mechanisms for frequently accessed data

## Notes

- The refactoring maintains backward compatibility
- Child modules can gradually migrate to using the centralized utilities
- No breaking changes to existing APIs
- All modules remain independently functional
