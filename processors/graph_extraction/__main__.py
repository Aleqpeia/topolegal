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
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0.8, debug_output: str = None):
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not set!")
            sys.exit(1)

        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.debug_output = debug_output
        self.prompt = PromptTemplate(
            input_variables=["text", "entity_list"],
            template="""Return ONLY valid JSON. Do not include any other text before or after the JSON.

Створіть правові триплети (джерело, відношення, ціль) ТІЛЬКИ з цих сутностей:
{entity_list}

Текст: {text}

ПРАВИЛА:
1. Джерело та ціль ТІЛЬКИ зі списку сутностей
2. Обов'язково правове посилання 
3. Confidence: 0.4-0.6 низька, 0.7-0.8 середня, 0.9+ висока
4. Повертайте ТІЛЬКИ валідний JSON без додаткового тексту

ПРИКЛАД ВАЛІДНОГО JSON:
{{
    "triplets": [
        {{
            "source": "суд",
            "relation": "вирішує_питання_про",
            "target": "запобіжний захід",
            "legal_reference": "ч.2 ст.331 КПК України",
            "confidence": 0.8
        }},
        {{
            "source": "підозрюваний",
            "relation": "може_бути_призначено",
            "target": "домашній арешт",
            "legal_reference": "ч.1 ст.181 КПК України",
            "confidence": 0.7
        }}
    ],
    "legal_references": ["ч.2 ст.331 КПК України", "ч.1 ст.181 КПК України"]
}}

ВАЖЛИВО: Якщо не можете створити триплети, поверніть порожній JSON:
{{
    "triplets": [],
    "legal_references": []
}}

Повертайте ТІЛЬКИ JSON без додаткового тексту.""")

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

        # Prepare debug info
        debug_info = {
            "text": text,
            "entities": entities,
            "entity_list_str": entity_list_str,
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Get raw LLM output
            result = await self.chain.arun(text=text, entity_list=entity_list_str)
            
            # Log raw output for debugging
            debug_info["raw_llm_output"] = result
            debug_info["success"] = True
            
            if self.debug_output:
                self._save_debug_output(debug_info)
            
            # Try to parse JSON
            try:
                parsed = json.loads(result)
                debug_info["parsed_json"] = parsed
            except json.JSONDecodeError as e:
                debug_info["json_error"] = str(e)
                debug_info["success"] = False
                if self.debug_output:
                    self._save_debug_output(debug_info)
                logger.error(f"JSON parse error: {e}")
                logger.error(f"Raw output: {result[:500]}...")
                return LegalKnowledgeGraph(triplets=[], entities=all_entities, legal_references=[])

            # Simple validation: only keep triplets with valid entities
            valid_triplets = []
            validation_errors = []
            
            for i, t in enumerate(parsed.get("triplets", [])):
                source = t.get("source", "")
                target = t.get("target", "")
                legal_ref = t.get("legal_reference", "")

                # Check if source and target are in entity list
                if source not in entity_texts:
                    validation_errors.append(f"Triplet {i}: source '{source}' not in entity list")
                elif target not in entity_texts:
                    validation_errors.append(f"Triplet {i}: target '{target}' not in entity list")
                elif not legal_ref or legal_ref == "null":
                    validation_errors.append(f"Triplet {i}: missing or null legal_reference")
                else:
                    valid_triplets.append(KnowledgeTriplet(**t))
            
            debug_info["validation_errors"] = validation_errors
            debug_info["valid_triplets_count"] = len(valid_triplets)
            debug_info["total_triplets_in_output"] = len(parsed.get("triplets", []))
            
            if self.debug_output:
                self._save_debug_output(debug_info)

            return LegalKnowledgeGraph(
                triplets=valid_triplets,
                entities=all_entities,
                legal_references=parsed.get("legal_references", [])
            )

        except Exception as e:
            debug_info["error"] = str(e)
            debug_info["success"] = False
            if self.debug_output:
                self._save_debug_output(debug_info)
            logger.error(f"Error: {e}")
            return LegalKnowledgeGraph(triplets=[], entities=all_entities, legal_references=[])

    def _save_debug_output(self, debug_info):
        """Save debug information to file"""
        try:
            import json
            from pathlib import Path
            
            debug_dir = Path(self.debug_output)
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename based on timestamp and success status
            timestamp = debug_info["timestamp"].replace(":", "-").replace(".", "-")
            success_str = "success" if debug_info.get("success", False) else "failed"
            filename = f"debug_{timestamp}_{success_str}.json"
            
            debug_file = debug_dir / filename
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save debug output: {e}")


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


def visualize_results(results_file: str, output_dir: str = "graphs/", **kwargs):
    """Visualize the knowledge graph results"""
    try:
        from . import visualize_graphs
        visualize_graphs(results_file, output_dir, **kwargs)
        logger.info(f"Graph visualization complete. Results saved to {output_dir}")
    except ImportError as e:
        logger.error(f"Missing visualization dependencies: {e}")
        logger.info("Install required packages: pip install matplotlib seaborn networkx")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Simple legal knowledge graph extraction")
    parser.add_argument("csv_file", help="CSV file with legal documents")
    parser.add_argument("--output", "-o", default="results.json", help="Output file")
    parser.add_argument("--max-docs", "-m", type=int, default=None, help="Max documents")
    parser.add_argument("--delay", "-d", type=float, default=2.0, help="API delay")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize graphs after processing")
    parser.add_argument("--graph-output", default="graphs/", help="Graph output directory")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence for graph edges")
    parser.add_argument("--layout", choices=['spring', 'circular', 'hierarchical'], 
                       default='spring', help="Graph layout")
    parser.add_argument("--combined", action="store_true", help="Create combined graph")
    parser.add_argument("--individual", action="store_true", help="Create individual graphs")
    parser.add_argument("--subgraphs", action="store_true", help="Create person subgraphs")
    parser.add_argument("--export", action="store_true", help="Export graph data")
    parser.add_argument("--stats", action="store_true", help="Generate graph statistics")
    parser.add_argument("--filter-zero-triplets", action="store_true", default=True, 
                       help="Filter out documents with zero triplets (default: True)")
    parser.add_argument("--keep-zero-triplets", action="store_true", 
                       help="Keep documents with zero triplets (overrides --filter-zero-triplets)")
    parser.add_argument("--debug-output", help="Directory to save debug LLM outputs")

    args = parser.parse_args()

    # Handle filtering options
    if args.keep_zero_triplets:
        filter_zero_triplets = False
    else:
        filter_zero_triplets = args.filter_zero_triplets

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

    extractor = LegalGraphExtractor(debug_output=args.debug_output)
    results = []

    for idx, row in df.iterrows():
        logger.info(f"Processing {int(idx) + 1}/{len(df)}")

        text = str(row['text']).strip()
        doc_id = str(row.get('doc_id', f"doc_{int(idx)}"))

        # Get entities from 'tags' column if available
        entities = []
        if 'tags' in df.columns and pd.notna(row['tags']):
            try:
                entities = json.loads(str(row['tags']))
            except:
                logger.warning(f"Could not parse entities for doc {doc_id}")

        result = await process_document(text, entities, doc_id, extractor)
        
        # Filter out zero triplets if requested
        if filter_zero_triplets and result.get('triplets_count', 0) == 0:
            logger.info(f"Skipping doc {doc_id} - no triplets extracted")
            continue
            
        results.append(result)

        if int(idx) + 1 < len(df):
            await asyncio.sleep(args.delay)

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    successful = len([r for r in results if 'error' not in r])
    total_triplets = sum(r.get('triplets_count', 0) for r in results if 'error' not in r)
    zero_triplet_docs = len([r for r in results if r.get('triplets_count', 0) == 0])

    logger.info(f"""
    Complete:
    - Successful: {successful}/{len(results)}
    - Total triplets: {total_triplets}
    - Zero triplet docs: {zero_triplet_docs}
    - Saved to: {args.output}
    """)

    # Visualize graphs if requested
    if args.visualize:
        logger.info("Starting graph visualization...")
        visualize_results(
            results_file=args.output,
            output_dir=args.graph_output,
            min_confidence=args.min_confidence,
            layout=args.layout,
            combined=args.combined,
            individual=args.individual,
            subgraphs=args.subgraphs,
            export=args.export,
            stats=args.stats
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python __main__.py data.csv --max-docs 10 --visualize")
    else:
        asyncio.run(main())