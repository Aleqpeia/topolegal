# Knowledge Graph Extraction for Legal Documents

This module provides functionality to extract knowledge graphs from legal documents using LangChain and OpenAI's GPT models. **This module focuses solely on knowledge graph generation from pre-extracted entities - it does not perform entity extraction itself.**

## Architecture

This module follows the principle of **separation of concerns**:
- **Entity Extraction**: Handled by the separate `processors.entities` module
- **Knowledge Graph Extraction**: This module - generates triplets from text and pre-extracted entities

## Features

- **Knowledge Graph Generation**: Create knowledge graph triplets (source, relation, target) from legal text and pre-extracted entities
- **Legal Reference Tracking**: Track legal references and their relationships
- **Multiple Input Formats**: Support for CSV files with pre-extracted entities
- **LangChain Integration**: Uses LangChain for LLM interactions with OpenAI GPT models

## Installation

1. Install the required dependencies:
```bash
pip install langchain langchain-community openai pandas pydantic
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
import asyncio
from processors.graph_extraction import LegalGraphExtractor

async def extract_knowledge_graph():
    # Initialize the extractor
    extractor = LegalGraphExtractor(model_name="gpt-4", temperature=0.0)
    
    # Legal text in Ukrainian
    text = """
    Відповідно до положень ч.2 ст.331 КПК України, суд вирішує питання про запобіжний захід. 
    Підозрюваному може бути призначено домашній арешт згідно з ч.1 ст.181 КПК України.
    """
    
    # Pre-extracted entities (from the entities module)
    entities = [
        {"text": "суд", "label": "COURT", "start": 0, "end": 3, "confidence": 0.9},
        {"text": "підозрюваному", "label": "LEGAL_ROLE", "start": 0, "end": 12, "confidence": 0.8},
        {"text": "ч.2 ст.331 КПК України", "label": "LEGAL_REF", "start": 0, "end": 20, "confidence": 0.9},
    ]
    
    # Extract knowledge graph
    knowledge_graph = await extractor.extract_triplets(text, entities)
    
    # Print results
    for triplet in knowledge_graph.triplets:
        print(f"({triplet.source}, {triplet.relation}, {triplet.target})")

# Run the extraction
asyncio.run(extract_knowledge_graph())
```

### Processing CSV Files with Pre-extracted Entities

```bash
python -m processors.graph_extraction processing_doc_links.csv --max-docs 10 --output results.json
```

The CSV file should have:
- `text` column: The legal document text
- `entities` column (optional): Pre-extracted entities as JSON string or list

Example CSV format:
```csv
text,entities,doc_id
"Відповідно до положень ч.2 ст.331 КПК України...",'[{"text":"суд","label":"COURT","start":0,"end":3}]',doc_001
```

### Integration with Entity Extraction Module

```python
# First, extract entities using the entities module
from processors.entities import EntityExtractor

entity_extractor = EntityExtractor()
entities = entity_extractor.extract_entities(legal_text)

# Then, extract knowledge graph using this module
from processors.graph_extraction import LegalGraphExtractor

graph_extractor = LegalGraphExtractor()
knowledge_graph = await graph_extractor.extract_triplets(legal_text, entities)
```

## Data Models

### KnowledgeTriplet
Represents a knowledge graph triplet:
- `source`: The source entity
- `relation`: The relationship between entities
- `target`: The target entity
- `legal_reference`: Legal basis for the relationship
- `confidence`: Confidence score (0.0-1.0)

### LegalEntity
Represents an entity with legal context:
- `text`: The entity text
- `label`: NER label (COURT, LEGAL_ROLE, etc.)
- `start`: Start position in text
- `end`: End position in text
- `legal_role`: Role in legal context

### LegalKnowledgeGraph
Contains the complete knowledge graph:
- `triplets`: List of KnowledgeTriplet objects
- `entities`: List of LegalEntity objects
- `legal_references`: List of legal references found

## Example Output

```json
{
  "triplets": [
    {
      "source": "суд",
      "relation": "вирішує_питання_про",
      "target": "запобіжний захід",
      "legal_reference": "ч.2 ст.331 КПК України",
      "confidence": 0.9
    },
    {
      "source": "підозрюваному",
      "relation": "може_бути_призначено",
      "target": "домашній арешт",
      "legal_reference": "ч.1 ст.181 КПК України",
      "confidence": 0.8
    }
  ],
  "legal_references": [
    "ч.2 ст.331 КПК України",
    "ч.1 ст.181 КПК України"
  ]
}
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Parameters
- `model_name`: GPT model to use (default: "gpt-4")
- `temperature`: Model temperature (default: 0.0 for deterministic output)

## Workflow

1. **Entity Extraction**: Use `processors.entities` module to extract entities from legal documents
2. **Knowledge Graph Extraction**: Use this module to generate knowledge graph triplets from text and entities
3. **Analysis**: Analyze the resulting knowledge graph for legal insights

## Dependencies

- `langchain>=0.3`
- `langchain-community>=0.3`
- `openai>=1.0.0`
- `pandas>=2.0.0`
- `pydantic>=2.0.0`

## Related Modules

- `processors.entities`: Entity extraction from legal documents
- `processors.graph_extraction`: Knowledge graph generation (this module) 