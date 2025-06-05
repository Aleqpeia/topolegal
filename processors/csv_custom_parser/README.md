# Generic CSV Parser

This module processes any CSV file structure and automatically ingests data into BigQuery with auto-generated schemas.

## Overview

The Generic CSV Parser is designed to:
1. Parse CSV files with any column structure 
2. Automatically infer BigQuery schema from the data
3. Clean column names for BigQuery compatibility
4. Generate table names from CSV filenames
5. Handle various data types (integers, floats, strings)
6. Provide batch processing for large files

## Key Features

- **Automatic Schema Detection**: Infers column types from data
- **Flexible Input**: Works with any CSV structure
- **BigQuery Integration**: Creates tables and ingests data automatically
- **Table Naming**: Uses CSV filename as table name (customizable)
- **Data Cleaning**: Handles quotes, special characters, and type conversion
- **Batch Processing**: Configurable batch sizes for large datasets
- **Error Handling**: Robust error handling with detailed logging

## Input Format

The module accepts any CSV file format. Common delimiters supported:
- Tab-separated (default)
- Comma-separated
- Auto-detection available

Examples:
```csv
category_code	name
2036	"Захоплення заручників"
3017	"звільнення з публічної служби"
```

```csv
id,title,amount,date
1,"Purchase Order",150.50,"2024-01-01"
2,"Invoice",2750.00,"2024-01-02"
```

## Output Schema

BigQuery tables are created automatically with:
- Column names cleaned for BigQuery compatibility
- Types inferred as INTEGER, FLOAT, or STRING
- NULLABLE or REQUIRED modes based on data presence
- Table name derived from CSV filename

## Usage

### Command Line Interface

List available CSV files:
```bash
python -m processors.csv_custom_parser
```

Process a specific CSV file:
```bash
python -m processors.csv_custom_parser --csv-file cause_categories.csv
```

With custom options:
```bash
python -m processors.csv_custom_parser \
    --csv-file my_data.csv \
    --table-name custom_table_name \
    --batch-size 500 \
    --delimiter "," \
    --dataset-id my_dataset
```

Dry run (parse only, no BigQuery ingestion):
```bash
python -m processors.csv_custom_parser --csv-file data.csv --dry-run --verbose
```

### Programmatic Usage

```python
from pathlib import Path
from processors.csv_custom_parser import GenericCsvParser

# Initialize parser
parser = GenericCsvParser(
    project_id="your-project-id",
    dataset_id="your_dataset",
    csv_delimiter="\t"  # or "," for comma-separated
)

# Process CSV file (table name auto-generated from filename)
csv_path = Path("processors/resources/my_data.csv")
result = parser.process_csv_file(csv_path)

# Or with custom table name
result = parser.process_csv_file(csv_path, table_name="custom_table")

print(f"Created table: {result['table_ref']}")
print(f"Processed: {result['total_processed']} records")
print(f"Errors: {result['total_errors']}")
```

### Schema Inference Example

```python
from processors.csv_custom_parser import GenericCsvParser, SchemaInference

parser = GenericCsvParser()
records = parser.parse_csv_file(Path("data.csv"))

# Get inferred schema
schema_info = SchemaInference.infer_schema(records)
for column in schema_info:
    print(f"{column['name']}: {column['type']} ({column['mode']})")
```

## Configuration

### Environment Variables

- `GOOGLE_CLOUD_PROJECT`: GCP project ID
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON file

### Command Line Options

- `--csv-file`: CSV filename (relative to resources directory)
- `--table-name`: Override table name (default: filename without extension)
- `--batch-size`: Records per BigQuery insert batch (default: 1000)
- `--delimiter`: CSV delimiter ('\t', ',', ';', or 'auto') (default: tab)
- `--dry-run`: Parse CSV without BigQuery ingestion
- `--project-id`: Override GCP project ID
- `--dataset-id`: BigQuery dataset ID (default: court_data_2022)
- `--verbose`: Enable debug logging

## Data Processing Pipeline

1. **File Discovery**: Lists available CSV files if none specified
2. **Table Naming**: Generates BigQuery-compatible table name from filename
3. **CSV Parsing**: Reads file with specified/detected delimiter
4. **Data Cleaning**: 
   - Cleans column names (removes special chars, ensures valid format)
   - Converts values to appropriate types (int, float, string)
   - Handles quoted strings and empty values
5. **Schema Inference**: Analyzes data to determine column types and nullability
6. **Table Creation**: Creates BigQuery table with inferred schema
7. **Batch Ingestion**: Inserts data in configurable batches with error handling

## Column Name Cleaning

Column names are automatically cleaned for BigQuery compatibility:
- Special characters replaced with underscores
- Quotes and whitespace removed
- Ensures names start with letter or underscore
- Converts to lowercase
- Handles empty/invalid names

Examples:
- `"Category Code"` → `category_code`
- `Amount ($)` → `amount___`
- `123field` → `col_123field`

## Type Inference

Data types are inferred by analyzing all values in each column:
- **INTEGER**: All non-null values are integers
- **FLOAT**: All non-null values are numbers (int or float)
- **STRING**: Default for all other cases

## Error Handling

- **File Not Found**: Lists available CSV files
- **Parse Errors**: Logs and skips problematic rows
- **BigQuery Errors**: Detailed error reporting per batch
- **Schema Errors**: Validates schema before table creation
- **Empty Files**: Graceful handling with informative messages

## File Structure

```
processors/csv_custom_parser/
├── __init__.py          # Module exports  
├── __main__.py          # CLI entry point
├── parser.py            # Main GenericCsvParser class
├── models.py            # CsvRecord and SchemaInference classes
└── README.md            # This documentation
```

## Examples

### Processing Legal Categories
```bash
python -m processors.csv_custom_parser --csv-file cause_categories.csv
# Creates table: court_data_2022.cause_categories
```

### Processing Financial Data
```bash
python -m processors.csv_custom_parser \
    --csv-file transactions.csv \
    --delimiter "," \
    --table-name financial_transactions
# Creates table: court_data_2022.financial_transactions
```

### Dry Run Analysis
```bash
python -m processors.csv_custom_parser --csv-file data.csv --dry-run --verbose
# Shows schema inference and sample data without BigQuery ingestion
``` 