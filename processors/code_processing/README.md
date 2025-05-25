# Legal Code Parser

A modular parser for extracting and storing legal code entries from HTML documents in BigQuery.

## Overview

This parser is designed to extract structured legal code information from HTML documents (specifically Ukrainian legal codes) and store them in a BigQuery database for analysis. The parser handles hierarchical structure including sections (розділ), headings (глава), articles (стаття), paragraphs (частина), and code entries (пункт).

## Architecture

The parser follows a modular architecture with clear separation of concerns:

### 1. Resources Module (`resources/`)
- **Abstract Base**: `ResourceConnector` - defines interface for resource fetching
- **Implementation**: `LocalResourceConnector` - fetches HTML files from local filesystem
- **Future**: Can be extended with `RemoteResourceConnector` for web resources

### 2. Repositories Module (`repositories/`)
- **Abstract Base**: `LegalCodeRepository` - defines interface for data storage
- **Implementation**: `BigQueryLegalCodeRepository` - stores entries in Google BigQuery
- **Repository Pattern**: Allows easy switching between different storage backends

### 3. Services Module (`services/`)
- **Code Extraction Service**: Main service that orchestrates the parsing process
- **Features**:
  - Extracts hierarchical legal code structure
  - Identifies and preserves article numbers with complex formats (e.g., "4-1")
  - Separates main content from additional information (italic text)
  - Placeholder for spaCy pipeline integration

### 4. Models (`models.py`)
- **LegalCodeEntry**: Data model for a single code entry
- **ParsedElement**: Represents parsed HTML elements
- **HtmlParseResult**: Contains the full parsed structure

## Dependencies

All dependencies are managed through Poetry at the project root level. The legal code parser uses:
- **beautifulsoup4**: HTML parsing and element extraction
- **google-cloud-bigquery**: BigQuery integration for data storage
- **python-dotenv**: Environment variable management
- **spacy** (optional): NLP processing for Ukrainian text

## Installation

This module is part of the `topolegal` project. All dependencies are managed through Poetry at the project root.

1. From the project root, ensure Poetry is installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install all project dependencies:
```bash
poetry install
```

3. Activate the Poetry shell:
```bash
poetry shell
```

4. Set up Google Cloud credentials (if using BigQuery):
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

5. Create a `.env` file in the code_processing directory:
```bash
cd processors/code_processing
cp env.example .env
# Edit .env with your configuration
```

## Configuration

Configure the parser using environment variables in `.env`:

```env
BIGQUERY_PROJECT_ID=your-gcp-project-id
BIGQUERY_DATASET_ID=legal_codes
BIGQUERY_TABLE_ID=criminal_process_code
LOG_LEVEL=INFO
SPACY_MODEL=uk_core_news_sm
```

## Usage

### Command Line Interface

From the `processors/code_processing` directory:

Basic usage:
```bash
python main.py --resource criminal_process_code.htm
```

Dry run (extract without storing):
```bash
python main.py --resource criminal_process_code.htm --dry-run
```

Export to JSON:
```bash
python main.py --resource criminal_process_code.htm --dry-run --output-json extracted_entries.json
```

With spaCy pipeline:
```bash
python main.py --resource criminal_process_code.htm --use-spacy
```

### Testing the Parser

Run the test script to verify the parser works correctly:
```bash
cd processors/code_processing
python test_parser.py
```

This will extract entries and save a sample to `test_output_sample.json`.

### Interactive Usage with Jupyter

Use the provided notebook for interactive exploration:
```bash
cd processors/code_processing
jupyter notebook code_parser_example.ipynb
```

### Programmatic Usage

```python
import sys
from pathlib import Path

# Add project to path if running standalone
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from processors.code_processing.resources import LocalResourceConnector
from processors.code_processing.repositories import BigQueryLegalCodeRepository
from processors.code_processing.services import CodeExtractionService

# Initialize components
resource_connector = LocalResourceConnector()
repository = BigQueryLegalCodeRepository(
    project_id="your-project",
    dataset_id="legal_codes",
    table_id="criminal_code"
)
extraction_service = CodeExtractionService(
    resource_connector=resource_connector,
    repository=repository
)

# Extract and store
result = extraction_service.extract_and_store("criminal_process_code.htm")
print(f"Processed {result['total_entries']} entries")

# Query data
article_entries = repository.get_entry_by_article(42)
search_results = repository.search_entries("слідчий")
```

## BigQuery Schema

The parser creates a BigQuery table with the following schema:

| Field | Type | Description |
|-------|------|-------------|
| code_entry | STRING | Entry number (e.g., "1", "2-1") |
| code_entry_text | STRING | Main text content |
| additional_info | STRING | Additional information (notes, amendments) |
| article_number | INTEGER | Article number |
| article_name | STRING | Article title |
| paragraph | STRING | Paragraph number |
| paragraph_name | STRING | Paragraph title |
| heading | STRING | Chapter number |
| heading_name | STRING | Chapter title |
| section | STRING | Section number (Roman numerals) |
| section_name | STRING | Section title |
| created_at | TIMESTAMP | Entry creation timestamp |

## Extending the Parser

### Adding New Resource Types

1. Create a new resource connector:
```python
from processors.code_processing.resources.base import ResourceConnector

class RemoteResourceConnector(ResourceConnector):
    def fetch(self, resource_path: str) -> str:
        # Implement fetching from URL
        pass
    
    def parse_to_json(self, content: str) -> Dict[str, Any]:
        # Use the same parsing logic
        pass
```

### Adding New Storage Backends

1. Create a new repository implementation:
```python
from processors.code_processing.repositories.base import LegalCodeRepository

class PostgreSQLRepository(LegalCodeRepository):
    # Implement all abstract methods
    pass
```

### Integrating spaCy

The service includes a placeholder for spaCy integration:
```python
import spacy

nlp = spacy.load("uk_core_news_sm")
extraction_service.set_spacy_pipeline(nlp)
```

## Notes

- The parser is specifically designed for Ukrainian legal documents
- Content is preserved in Ukrainian while code/structure uses English
- The parser handles complex HTML structures with nested CSS classes
- Italic text is automatically identified as additional information
- Changes/amendments at the top of documents are skipped

## Troubleshooting

1. **Import errors**: The code uses absolute imports. If running standalone scripts, they add the project root to the Python path automatically.

2. **Large file error**: The HTML files can be very large (>2MB). The parser handles this by streaming the content.

3. **BigQuery authentication**: Ensure your Google Cloud credentials are properly configured:
   ```bash
   gcloud auth application-default login
   ```

4. **Poetry environment**: Make sure you're in the Poetry shell from the project root:
   ```bash
   cd /path/to/topolegal
   poetry shell
   ```

## Future Enhancements

- [ ] Remote resource fetching (URLs)
- [ ] Additional storage backends (PostgreSQL, Elasticsearch)
- [ ] Full spaCy integration for NER and text analysis
- [ ] Batch processing of multiple documents
- [ ] Export to other formats (CSV, Parquet)
- [ ] Web interface for browsing extracted data 