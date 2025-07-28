# Legal Knowledge Graph Extraction and Visualization

This module provides comprehensive legal knowledge graph extraction and visualization capabilities using LangChain and NetworkX.

## Features

- **Knowledge Graph Extraction**: Extract legal triplets (source, relation, target) from text using pre-extracted entities
- **Graph Visualization**: Create interactive visualizations of legal knowledge graphs
- **Multiple Output Formats**: Export graphs as PNG images, GraphML, GML, and CSV
- **Statistics Generation**: Comprehensive graph analysis and statistics
- **Flexible Processing**: Support for individual documents, combined graphs, and person-specific subgraphs
- **BigQuery Integration**: Process documents directly from BigQuery tables and update with triplet results

## Installation

### Dependencies

```bash
# Core dependencies
pip install langchain-community openai pandas

# Visualization dependencies
pip install matplotlib seaborn networkx

# BigQuery dependencies
pip install google-cloud-bigquery
```

### NixOS Setup

If you're on NixOS and encounter shared library issues:

```bash
# Install required system libraries
nix-env -iA nixpkgs.libstdcxx5 nixpkgs.gcc-unwrapped

# Or add to your shell.nix
{
  buildInputs = with pkgs; [
    libstdcxx5
    gcc-unwrapped
  ];
}
```

## Usage

### 1. Basic Knowledge Graph Extraction

```python
import asyncio
from src.legal_graph.processors import LegalGraphExtractor


async def extract_kg():
   # Sample text and entities
   text = "Суд вирішує питання про запобіжний захід згідно з ч.2 ст.331 КПК України."
   entities = [
      {"text": "суд", "label": "ORG", "start": 0, "end": 3},
      {"text": "ч.2 ст.331 КПК України", "label": "LEGAL_REF", "start": 45, "end": 64}
   ]

   # Extract knowledge graph
   extractor = LegalGraphExtractor()
   kg = await extractor.extract_triplets(text, entities)

   print(f"Extracted {len(kg.triplets)} triplets")
   for triplet in kg.triplets:
      print(f"{triplet.source} --{triplet.relation}--> {triplet.target}")


asyncio.run(extract_kg())
```

### 2. Graph Visualization

```python
from src.legal_graph.processors import visualize_graphs

# Visualize results from JSON file
visualize_graphs(
   input_file="results.json",
   output_dir="graphs/",
   min_confidence=0.6,
   layout='spring',
   combined=True,
   individual=True,
   export=True,
   stats=True
)
```

### 3. Command Line Interface

#### Process CSV and Extract Knowledge Graphs

```bash
# Basic processing
python -m processors.graph_extraction.__main__ data.csv --max-docs 10

# With visualization
python -m processors.graph_extraction.__main__ data.csv --max-docs 10 --visualize

# Advanced options
python -m processors.graph_extraction.__main__ data.csv \
  --max-docs 10 \
  --visualize \
  --graph-output custom_graphs/ \
  --min-confidence 0.7 \
  --layout hierarchical \
  --combined \
  --individual \
  --export \
  --stats
```

#### Process BigQuery Tables

```bash
# Basic BigQuery processing
python processors/graph_extraction/bigquery_triplets.py \
  --table lab-test-project-1-305710.court_data_2022.processing_doc_links

# With custom settings
python processors/graph_extraction/bigquery_triplets.py \
  --table lab-test-project-1-305710.court_data_2022.processing_doc_links \
  --gcp-project your-project-id \
  --gcp-key /path/to/service-account.json \
  --batch 500 \
  --debug-output debug_logs/

# Update existing records with zero triplets
python processors/graph_extraction/bigquery_triplets.py \
  --table lab-test-project-1-305710.court_data_2022.processing_doc_links \
  --update-existing

# Check table schema and show statistics
python processors/graph_extraction/bigquery_triplets.py \
  --table lab-test-project-1-305710.court_data_2022.processing_doc_links \
  --check-schema --stats
```

#### Visualize Existing Results

```bash
# Using the CLI function
python -c "from processors.graph_extraction import run_visualizer_cli; run_visualizer_cli()" \
  results.json --combined --export --stats

# Or run the graph visualizer directly
python processors/graph_extraction/graph.py results.json --combined --export
```

### 4. Integration with Main Workflow

```python
from src.legal_graph.processors import LegalGraphExtractor, visualize_graphs
from src.legal_graph.processors import EntityExtractor  # Your entity extraction module


async def process_document(text):
   # Step 1: Extract entities
   entity_extractor = EntityExtractor()
   entities = await entity_extractor.extract_entities(text)

   # Step 2: Extract knowledge graph
   kg_extractor = LegalGraphExtractor()
   kg = await kg_extractor.extract_triplets(text, entities)

   return kg


# Process multiple documents and visualize
results = []
for doc in documents:
   kg = await process_document(doc['text'])
   results.append({
      'doc_id': doc['id'],
      'knowledge_graph': kg.dict()
   })

# Save results
import json

with open('results.json', 'w') as f:
   json.dump(results, f, indent=2)

# Visualize
visualize_graphs('results.json', 'graphs/', combined=True, export=True)
```

## Output Formats

### 1. Graph Images (PNG)
- Combined graph from all documents
- Individual document graphs
- Person-specific subgraphs
- Color-coded by entity type and confidence levels

### 2. Graph Data Files
- **GraphML**: Preserves all attributes for network analysis tools
- **GML**: Simple graph format
- **CSV**: Edge and node lists for spreadsheet analysis

### 3. Statistics (JSON)
```json
{
  "basic_stats": {
    "nodes": 25,
    "edges": 45,
    "density": 0.15,
    "is_connected": false
  },
  "node_types": {
    "PERSON": 10,
    "COURT": 3,
    "LEGAL_REF": 8,
    "CRIME": 4
  },
  "confidence_distribution": {
    "high (0.8+)": 20,
    "medium (0.6-0.8)": 15,
    "low (0.4-0.6)": 10
  }
}
```

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### BigQuery Table Schema

The BigQuery table should have the following structure:

```sql
-- Required columns
doc_id STRING NOT NULL,
text STRING,
tags STRING,  -- JSON string of entities

-- Optional columns (will be updated by the script)
triplets STRING,  -- JSON string of extracted triplets
triplets_count INT64,
processing_timestamp TIMESTAMP
```

#### BigQuery Processing Features

- **Idempotent Processing**: Only processes documents without triplets or with zero triplets
- **Batch Processing**: Processes documents in configurable batches
- **Error Handling**: Continues processing even if individual documents fail
- **Statistics**: Provides processing statistics and table schema validation
- **Debug Output**: Saves LLM responses for debugging and analysis

### Model Configuration

```python
# Custom model settings
extractor = LegalGraphExtractor(
    model_name="gpt-4",  # or "gpt-3.5-turbo"
    temperature=0.1      # Lower for more consistent results
)
```

### Visualization Options

- **Layouts**: `spring`, `circular`, `hierarchical`
- **Confidence Thresholds**: Filter edges by confidence level
- **Node Types**: Color-coded by entity type (PERSON, COURT, CRIME, etc.)
- **Edge Styling**: Thickness based on confidence levels

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install matplotlib seaborn networkx google-cloud-bigquery
   ```

2. **NixOS Library Issues**
   ```bash
   # Install missing libraries
   nix-env -iA nixpkgs.libstdcxx5
   ```

3. **OpenAI API Errors**
   - Check your API key is set: `echo $OPENAI_API_KEY`
   - Verify API key has sufficient credits
   - Check rate limits

4. **BigQuery Authentication Issues**
   - Verify service account key: `echo $GOOGLE_APPLICATION_CREDENTIALS`
   - Check project ID: `echo $GOOGLE_CLOUD_PROJECT`
   - Ensure service account has BigQuery permissions

5. **Memory Issues with Large Graphs**
   - Reduce `max-docs` parameter
   - Increase `min-confidence` threshold
   - Use `--individual` instead of `--combined`

### Performance Tips

- Use `--max-docs` to limit processing for testing
- Set higher `min-confidence` to reduce noise
- Use `--delay` to respect API rate limits
- Process in batches for large datasets
- Use `--batch` parameter for BigQuery processing

## Example Workflow

1. **Prepare Data**: CSV with `text` and `entities` columns
2. **Extract Knowledge Graphs**: Run the extraction script
3. **Visualize Results**: Generate graphs and statistics
4. **Analyze**: Use exported data for further analysis

```bash
# Complete workflow example
python -m processors.graph_extraction.__main__ legal_docs.csv \
  --max-docs 50 \
  --visualize \
  --combined \
  --export \
  --stats \
  --graph-output analysis_results/
```

This will create:
- `results.json`: Extracted knowledge graphs
- `analysis_results/combined_graph_conf_0.6.png`: Main visualization
- `analysis_results/combined_graph_conf_0.6.graphml`: Network analysis file
- `analysis_results/combined_graph_stats_conf_0.6.json`: Statistics
- `analysis_results/combined_graph_conf_0.6_edges.csv`: Edge list
- `analysis_results/combined_graph_conf_0.6_nodes.csv`: Node list

### BigQuery Workflow

```bash
# Check table schema and statistics
python processors/graph_extraction/bigquery_triplets.py \
  --table lab-test-project-1-305710.court_data_2022.processing_doc_links \
  --check-schema --stats

# Process documents and update table
python processors/graph_extraction/bigquery_triplets.py \
  --table lab-test-project-1-305710.court_data_2022.processing_doc_links \
  --batch 1000 \
  --debug-output debug_logs/

# Query results from BigQuery
bq query --use_legacy_sql=false "
SELECT doc_id, triplets_count, entities_count
FROM \`lab-test-project-1-305710.court_data_2022.processing_doc_links\`
WHERE triplets_count > 0
ORDER BY triplets_count DESC
LIMIT 10
"
``` 