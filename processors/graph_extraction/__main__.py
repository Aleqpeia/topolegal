#!/usr/bin/env python3
"""
Simplified script to process legal documents from CSV
Run with: python __main__.py processing_doc_links.csv --max-docs 10
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for LangChain
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
    except ImportError:
        logger.error("LangChain not available. Install: pip install langchain-community")
        sys.exit(1)


# Simple models
class KnowledgeTriplet(BaseModel):
    source: str
    relation: str
    target: str
    legal_reference: Optional[str] = None
    confidence: float = 0.0


class LegalEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class LegalKnowledgeGraph(BaseModel):
    triplets: List[KnowledgeTriplet]
    entities: List[LegalEntity]
    legal_references: List[str]


class LegalGraphExtractor:
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0.1):
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not set!")
            sys.exit(1)

        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = PromptTemplate(
            input_variables=["text", "entity_list"],
            template="""Створіть правові триплети (джерело, відношення, ціль) ТІЛЬКИ з цих сутностей:
{entity_list}

Текст: {text}

ПРАВИЛА:
1. Джерело та ціль ТІЛЬКИ зі списку сутностей
2. Обов'язково правове посилання 
3. Confidence: 0.4-0.6 низька, 0.7-0.8 середня, 0.9+ висока

ФОРМАТ JSON:
{{
    "triplets": [
        {{
            "source": "з_списку_сутностей",
            "relation": "правовий_зв'язок", 
            "target": "з_списку_сутностей",
            "legal_reference": "обов'язково",
            "confidence": 0.8
        }}
    ],
    "legal_references": ["унікальні_посилання"]
}}""")

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    async def extract_triplets(self, text: str, entities: List[Dict]) -> LegalKnowledgeGraph:
        # Keep ALL entities
        all_entities = [
            LegalEntity(
                text=e.get("text", ""),
                label=e.get("label", ""),
                start=e.get("start", 0),
                end=e.get("end", 0)
            ) for e in entities
        ]

        # Create entity list for prompt
        entity_texts = [e.get("text", "") for e in entities if e.get("text", "").strip()]
        entity_list_str = ", ".join(f'"{entity}"' for entity in entity_texts)

        try:
            result = await self.chain.arun(text=text, entity_list=entity_list_str)
            parsed = json.loads(result)

            # Simple validation: only keep triplets with valid entities
            valid_triplets = []
            for t in parsed.get("triplets", []):
                source = t.get("source", "")
                target = t.get("target", "")
                legal_ref = t.get("legal_reference", "")

                # Check if source and target are in entity list
                if (source in entity_texts and
                        target in entity_texts and
                        legal_ref and legal_ref != "null"):
                    valid_triplets.append(KnowledgeTriplet(**t))

            return LegalKnowledgeGraph(
                triplets=valid_triplets,
                entities=all_entities,
                legal_references=parsed.get("legal_references", [])
            )

        except Exception as e:
            logger.error(f"Error: {e}")
            return LegalKnowledgeGraph(triplets=[], entities=all_entities, legal_references=[])


async def process_document(text: str, entities: List[Dict], doc_id: str, extractor: LegalGraphExtractor) -> Dict:
    try:
        graph = await extractor.extract_triplets(text, entities)

        return {
            "doc_id": doc_id,
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
                "entities": [
                    {
                        "text": e.text,
                        "label": e.label,
                        "start": e.start,
                        "end": e.end
                    } for e in graph.entities
                ]
            }
        }
    except Exception as e:
        return {"doc_id": doc_id, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Simple legal knowledge graph extraction")
    parser.add_argument("csv_file", help="CSV file with legal documents")
    parser.add_argument("--output", "-o", default="results.json", help="Output file")
    parser.add_argument("--max-docs", "-m", type=int, default=None, help="Max documents")
    parser.add_argument("--delay", "-d", type=float, default=2.0, help="API delay")

    args = parser.parse_args()

    if not Path(args.csv_file).exists():
        logger.error(f"File not found: {args.csv_file}")
        return

    # Load data
    df = pd.read_csv(args.csv_file)
    if 'text' not in df.columns:
        logger.error("Missing 'text' column")
        return

    # Remove empty texts
    df = df[df['text'].notna() & (df['text'].str.strip() != '')]

    if args.max_docs:
        df = df.head(args.max_docs)

    logger.info(f"Processing {len(df)} documents")

    extractor = LegalGraphExtractor()
    results = []

    for idx, row in df.iterrows():
        logger.info(f"Processing {idx + 1}/{len(df)}")

        text = str(row['text']).strip()
        doc_id = row.get('doc_id', f"doc_{idx}")

        # Get entities from 'tags' column if available
        entities = []
        if 'tags' in df.columns and pd.notna(row['tags']):
            try:
                entities = json.loads(str(row['tags']))
            except:
                logger.warning(f"Could not parse entities for doc {doc_id}")

        result = await process_document(text, entities, doc_id, extractor)
        results.append(result)

        if idx + 1 < len(df):
            await asyncio.sleep(args.delay)

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    successful = len([r for r in results if 'error' not in r])
    total_triplets = sum(r.get('triplets_count', 0) for r in results if 'error' not in r)

    logger.info(f"""
    Complete:
    - Successful: {successful}/{len(results)}
    - Total triplets: {total_triplets}
    - Saved to: {args.output}
    """)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python __main__.py data.csv --max-docs 10")
    else:
        asyncio.run(main())