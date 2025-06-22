#!/usr/bin/env python3
"""
Standalone script to process legal documents from CSV
Run with: python main.py processing_doc_links.csv --max-docs 10
"""

import asyncio
import json
import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging
from pathlib import Path
import argparse
from datetime import datetime
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for LangChain
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain community available")
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        LANGCHAIN_AVAILABLE = True
        logger.warning("Using deprecated langchain imports. Consider upgrading to langchain-community.")
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logger.error("LangChain not available. Please install: pip install langchain-community")
        sys.exit(1)

# Pydantic models for knowledge graph
class KnowledgeTriplet(BaseModel):
    """Represents a knowledge graph triplet (source, relation, target)"""
    source: str
    relation: str
    target: str
    legal_reference: Optional[str] = None
    confidence: float = 0.0

    def __str__(self) -> str:
        return f"({self.source}, {self.relation}, {self.target})"

class LegalEntity(BaseModel):
    """Enhanced entity with legal context"""
    text: str
    label: str
    start: int
    end: int
    legal_role: Optional[str] = None

class LegalKnowledgeGraph(BaseModel):
    """Knowledge graph represented as triplets with legal metadata"""
    triplets: List[KnowledgeTriplet]
    entities: List[LegalEntity]
    legal_references: List[str]

class LegalGraphExtractor:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for graph extraction")

        # Ensure ChatOpenAI is available
        if 'ChatOpenAI' not in globals():
            raise ImportError("ChatOpenAI not available. Please check LangChain installation.")

        try:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        except Exception as e:
            raise ImportError(f"Failed to initialize ChatOpenAI: {str(e)}")
            
        self.prompt = PromptTemplate(
            input_variables=["text", "entities"],
            template="""На основі наданого тексту та виявлених сутностей, створіть граф знань у форматі триплетів (джерело, відношення, ціль).

Текст: {text}

Виявлені сутності: {entities}

Приклад аналізу:
Текст: "Відповідно до положень ч.2 ст.331 КПК України, суд вирішує питання про запобіжний захід. Підозрюваному може бути призначено домашній арешт згідно з ч.1 ст.181 КПК України."

Очікувані триплети:
1. (суд, вирішує_питання_про, запобіжний захід) - ч.2 ст.331 КПК України
2. (підозрюваному, може_бути_призначено, домашній арешт) - ч.1 ст.181 КПК України
3. (запобіжний захід, регулюється, ч.2 ст.331 КПК України)

Формат виводу як JSON:
{{
    "triplets": [
        {{
            "source": "string",
            "relation": "string", 
            "target": "string",
            "legal_reference": "string",
            "confidence": float
        }}
    ],
    "legal_references": ["string"]
}}

Правила створення триплетів:
1. Використовуйте конкретні сутності з тексту
2. Відношення повинні відображати правові зв'язки
3. Обов'язково вказуйте правове посилання
4. Використовуйте українські терміни у відношеннях"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    async def extract_triplets(self, text: str, entities: List[Dict]) -> LegalKnowledgeGraph:
        """Extract knowledge graph triplets from legal text and entities"""
        # Convert entities to LegalEntity objects
        legal_entities = [
            LegalEntity(
                text=e.get("text", ""),
                label=e.get("label", ""),
                start=e.get("start", 0),
                end=e.get("end", 0),
                legal_role=self._determine_legal_role(e.get("text", ""), e.get("label", ""))
            ) for e in entities
        ]

        try:
            # Extract triplets using LLM
            result = await self.chain.arun(text=text, entities=str(entities))

            parsed_result = json.loads(result)

            triplets = [
                KnowledgeTriplet(**triplet_data)
                for triplet_data in parsed_result.get("triplets", [])
            ]

            legal_references = parsed_result.get("legal_references", [])

            return LegalKnowledgeGraph(
                triplets=triplets,
                entities=legal_entities,
                legal_references=legal_references
            )

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return LegalKnowledgeGraph(
                triplets=[],
                entities=legal_entities,
                legal_references=[]
            )

    def _determine_legal_role(self, text: str, label: str) -> Optional[str]:
        """Determine legal role based on entity text and NER label"""
        legal_roles_map = {
            "суд": "judicial_authority",
            "підозрюваний": "suspect",
            "обвинувачений": "accused",
            "прокурор": "prosecutor",
            "адвокат": "defense_attorney",
            "слідчий": "investigator",
            "потерпілий": "victim"
        }

        text_lower = text.lower()
        for term, role in legal_roles_map.items():
            if term in text_lower:
                return role

        return None

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor for knowledge graph extraction"""
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set!")
            logger.info("Please set your OpenAI API key:")
            logger.info("export OPENAI_API_KEY='your-api-key-here'")
            sys.exit(1)

        self.extractor = LegalGraphExtractor()

    def preprocess_legal_text(self, text: str) -> str:
        """Preprocess legal text for better knowledge graph extraction"""
        if not text or pd.isna(text):
            return ""

        # Convert to string if it's a pandas Series or other type
        if not isinstance(text, str):
            text = str(text)

        text = text.strip()
        return text

async def process_document(text: str, entities: List[Dict], doc_metadata: Dict, extractor: LegalGraphExtractor) -> Dict:
    """Process a document to extract knowledge graph with metadata"""
    try:
        graph = await extractor.extract_triplets(text, entities)

        result = {
            "doc_id": doc_metadata.get("doc_id"),
            "court_code": doc_metadata.get("court_code"),
            "judgment_code": doc_metadata.get("judgment_code"),
            "processing_timestamp": datetime.now().isoformat(),
            "entities_count": len(entities),
            "triplets_count": len(graph.triplets),
            "knowledge_graph": {
                "triplets": [
                    {
                        "source": t.source,
                        "relation": t.relation,
                        "target": t.target,
                        "legal_reference": t.legal_reference,
                        "confidence": t.confidence
                    } for t in graph.triplets
                ],
                "legal_references": graph.legal_references,
                "entities": [
                    {
                        "text": e.text,
                        "label": e.label,
                        "legal_role": e.legal_role,
                        "start": e.start,
                        "end": e.end
                    } for e in graph.entities
                ]
            }
        }

        return result

    except Exception as e:
        logger.error(f"Error processing document {doc_metadata.get('doc_id')}: {str(e)}")
        return {
            "doc_id": doc_metadata.get("doc_id"),
            "error": str(e),
            "processing_timestamp": datetime.now().isoformat()
        }

async def main():
    parser = argparse.ArgumentParser(description="Process legal documents from CSV for knowledge graph extraction")
    parser.add_argument("csv_file", help="Path to CSV file with legal documents and entities")
    parser.add_argument("--output", "-o", default="knowledge_graph_results.json",
                        help="Output JSON file path")
    parser.add_argument("--max-docs", "-m", type=int, default=None,
                        help="Maximum number of documents to process")
    parser.add_argument("--delay", "-d", type=float, default=2.0,
                        help="Delay between API calls (seconds)")

    args = parser.parse_args()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Please set your OpenAI API key:")
        logger.info("export OPENAI_API_KEY='your-api-key-here'")
        return

    # Validate input file
    if not Path(args.csv_file).exists():
        logger.error(f"CSV file not found: {args.csv_file}")
        return

    # Load CSV
    logger.info(f"Loading CSV file: {args.csv_file}")
    try:
        df = pd.read_csv(args.csv_file)
        logger.info(f"Loaded {len(df)} documents")
        logger.info(f"Available columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return

    # Check if we have required columns
    required_columns = ['text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info("Expected columns: text, entities (optional)")
        return

    # Remove documents with empty text
    df = df[df['text'].notna() & (df['text'].str.strip() != '')]
    logger.info(f"Removed empty texts: {len(df)} documents remaining")

    # Limit number of documents
    if args.max_docs:
        df = df.head(args.max_docs)
        logger.info(f"Limited to {args.max_docs} documents")

    if len(df) == 0:
        logger.warning("No documents to process")
        return

    # Initialize processor
    processor = DocumentProcessor()

    # Process documents one by one
    all_results = []

    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            logger.info(f"Processing document {idx + 1}/{len(df)}")

            # Preprocess text
            text = processor.preprocess_legal_text(row['text'])

            if not text:
                logger.warning(f"Empty text for document {idx}")
                continue

            # Get entities from the row if available, otherwise empty list
            entities = []
            if 'tags' in df.columns:
                entities_value = row['tags']
                if pd.notna(entities_value) and entities_value is not None:
                    try:
                        if isinstance(entities_value, str):
                            entities = json.loads(entities_value)
                        elif isinstance(entities_value, list):
                            entities = entities_value
                        else:
                            # Convert pandas Series or other types to string first
                            entities = json.loads(str(entities_value))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Could not parse entities for document {idx}: {e}")
                        entities = []
            
            logger.info(f"Using {len(entities)} entities for knowledge graph extraction")

            # Prepare metadata
            doc_metadata = {
                "doc_id": row.get('doc_id', f"doc_{idx}"),
                "court_code": row.get('court_code', ''),
                "judgment_code": row.get('judgment_code', ''),
            }

            # Process document
            result = await process_document(text, entities, doc_metadata, processor.extractor)
            all_results.append(result)

            logger.info(f"Document {idx + 1} processed successfully")

            # Add delay between API calls
            if idx + 1 < len(df):
                logger.info(f"Waiting {args.delay} seconds...")
                await asyncio.sleep(args.delay)

        except Exception as e:
            logger.error(f"Error processing document {idx}: {str(e)}")
            continue

    # Save results
    logger.info(f"Saving {len(all_results)} processed documents to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    successful_docs = [r for r in all_results if 'error' not in r]
    failed_docs = [r for r in all_results if 'error' in r]

    logger.info(f"""
    Processing Complete:
    - Total documents processed: {len(all_results)}
    - Successful: {len(successful_docs)}
    - Failed: {len(failed_docs)}
    """)

if __name__ == "__main__":
    asyncio.run(main())