from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class KnowledgeTriplet(BaseModel):
    """Represents a knowledge graph triplet (source, relation, target)"""
    source: str
    relation: str
    target: str
    legal_reference: Optional[str] = None  # Legal basis for this relationship
    confidence: float = 0.0

    def __str__(self) -> str
        return f"({self.source}, {self.relation}, {self.target})"

class LegalEntity(BaseModel):
    """Enhanced entity with legal context"""
    text: str
    label: str  # NER label (PERSON, ORG, ROLE, etc.)
    start: int
    end: int
    legal_role: Optional[str] = None  # Role in legal context (підозрюваний, суд, etc.)

class LegalKnowledgeGraph(BaseModel):
    """Knowledge graph represented as triplets with legal metadata"""
    triplets: List[KnowledgeTriplet]
    entities: List[LegalEntity]
    legal_references: List[str]  # All legal references found in text

    def get_triplets_by_legal_ref(self, legal_ref: str) -> List[KnowledgeTriplet]:
        """Get all triplets associated with a specific legal reference"""
        return [t for t in self.triplets if t.legal_reference == legal_ref]

    def get_entity_relations(self, entity: str) -> List[KnowledgeTriplet]:
        """Get all relations where entity appears as source or target"""
        return [t for t in self.triplets if t.source == entity or t.target == entity]

class LegalGraphExtractor:
    def __init__(self, model_name: str = "gpt-4.1-nano", temperature: float = 5.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
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
4. (домашній арешт, регулюється, ч.1 ст.181 КПК України)
5. (суд, має_повноваження, призначення запобіжного заходу)

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
1. Використовуйте конкретні сутності з тексту як джерела та цілі
2. Відношення повинні відображати правові зв'язки (призначає, регулюється, підлягає, застосовується)
3. Обов'язково вказуйте правове посилання для кожного триплету
4. Створюйте додаткові триплети для зв'язків між правовими нормами та процедурами
5. Використовуйте українські терміни у відношеннях

Зосередьтеся на:
- Процедурних відношеннях (хто що робить)
- Правових основах (що чим регулюється)
- Суб'єктно-об'єктних відношеннях у правовому процесі"""
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

        # Extract triplets using LLM
        result = await self.chain.arun(text=text, entities=str(entities))

        try:
            import json
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

        except json.JSONDecodeError:
            # Fallback: create empty graph
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

    def validate_triplets(self, kg: LegalKnowledgeGraph) -> List[str]:
        """Validate extracted triplets for legal consistency"""
        issues = []

        for triplet in kg.triplets:
            # Check if legal reference is provided
            if not triplet.legal_reference:
                issues.append(f"Missing legal reference for: {triplet}")

            # Check confidence threshold
            if triplet.confidence < 0.5:
                issues.append(f"Low confidence for: {triplet}")

            # Check for empty relations
            if not triplet.relation.strip():
                issues.append(f"Empty relation in: {triplet}")

        return issues

    def export_to_networkx(self, kg: LegalKnowledgeGraph):
        """Export knowledge graph to NetworkX format for analysis"""
        try:
            import networkx as nx

            G = nx.DiGraph()

            for triplet in kg.triplets:
                G.add_edge(
                    triplet.source,
                    triplet.target,
                    relation=triplet.relation,
                    legal_reference=triplet.legal_reference,
                    confidence=triplet.confidence
                )

            return G

        except ImportError:
            raise ImportError("NetworkX is required for graph export. Install with: pip install networkx")

# Usage example
async def process_legal_document(text: str, entities: List[Dict]) -> LegalKnowledgeGraph:
    """Process a legal document and extract knowledge graph from pre-extracted entities"""
    extractor = LegalGraphExtractor()
    kg = await extractor.extract_triplets(text, entities)

    # Validate results
    issues = extractor.validate_triplets(kg)
    if issues:
        print("Validation issues found:")
        for issue in issues:
            print(f"- {issue}")

    # Print extracted triplets
    print(f"\nExtracted {len(kg.triplets)} triplets:")
    for triplet in kg.triplets:
        legal_ref = f" [{triplet.legal_reference}]" if triplet.legal_reference else ""
        print(f"  {triplet}{legal_ref}")

    return kg

def visualize_graphs(input_file: str, output_dir: str = "graphs/", 
                    min_confidence: float = 0.6,
                    layout: str = 'spring',
                    combined: bool = True,
                    individual: bool = False,
                    subgraphs: bool = False,
                    export: bool = False,
                    stats: bool = False):
    """
    Visualize knowledge graphs from JSON results
    
    Args:
        input_file: JSON file with knowledge graph results
        output_dir: Output directory for graphs
        min_confidence: Minimum confidence for edges
        layout: Graph layout ('spring', 'circular', 'hierarchical')
        combined: Create combined graph from all documents
        individual: Create individual graphs for each document
        subgraphs: Create subgraphs for each person
        export: Export graph data in multiple formats
        stats: Generate graph statistics
    """
    try:
        # Import the visualizer
        from .graph import LegalGraphVisualizer
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and process results
        visualizer = LegalGraphVisualizer()
        results = visualizer.load_results(input_file)
        
        if not results:
            logger.error("No valid results to process")
            return
        
        logger.info(f"Loaded {len(results)} documents")
        
        # Create combined graph
        if combined or not (individual or subgraphs):
            logger.info("Creating combined graph...")
            combined_graph = visualizer.create_combined_graph(results, min_confidence)
            
            if combined_graph.number_of_nodes() > 0:
                # Visualize
                output_file = output_path / f"combined_graph_conf_{min_confidence}.png"
                visualizer.visualize_graph(combined_graph, str(output_file), layout)
                
                # Export data
                if export:
                    export_file = output_path / f"combined_graph_conf_{min_confidence}"
                    visualizer.export_graph_data(combined_graph, str(export_file))
                
                # Generate stats
                if stats:
                    stats_data = visualizer.generate_graph_statistics(combined_graph)
                    stats_file = output_path / f"combined_graph_stats_conf_{min_confidence}.json"
                    with open(stats_file, 'w', encoding='utf-8') as f:
                        json.dump(stats_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Statistics saved to {stats_file}")
                
                # Create person subgraphs
                if subgraphs:
                    subgraph_dir = output_path / "person_subgraphs"
                    subgraph_dir.mkdir(exist_ok=True)
                    visualizer.create_subgraphs_by_person(combined_graph, str(subgraph_dir))
        
        # Create individual graphs
        if individual:
            logger.info("Creating individual document graphs...")
            individual_dir = output_path / "individual_docs"
            individual_dir.mkdir(exist_ok=True)
            
            for doc_data in results:
                if 'error' not in doc_data:
                    doc_id = doc_data.get('doc_id', 'unknown')
                    doc_graph = visualizer.create_graph_from_document(doc_data, min_confidence)
                    
                    if doc_graph.number_of_edges() > 0:
                        output_file = individual_dir / f"doc_{doc_id}_graph.png"
                        visualizer.visualize_graph(doc_graph, str(output_file), layout, figsize=(12, 8))
        
        logger.info(f"Graph visualization complete. Results saved to {output_dir}")
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Install required packages: pip install matplotlib seaborn networkx")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")

def run_visualizer_cli():
    """Run the graph visualizer from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize legal knowledge graphs")
    parser.add_argument("input_file", help="JSON file with knowledge graph results")
    parser.add_argument("--output", "-o", default="graphs/", help="Output directory")
    parser.add_argument("--min-confidence", "-c", type=float, default=0.6, 
                       help="Minimum confidence for edges")
    parser.add_argument("--layout", choices=['spring', 'circular', 'hierarchical'], 
                       default='spring', help="Graph layout")
    parser.add_argument("--combined", action="store_true", 
                       help="Create combined graph from all documents")
    parser.add_argument("--individual", action="store_true", 
                       help="Create individual graphs for each document")
    parser.add_argument("--subgraphs", action="store_true", 
                       help="Create subgraphs for each person")
    parser.add_argument("--export", action="store_true", 
                       help="Export graph data in multiple formats")
    parser.add_argument("--stats", action="store_true", 
                       help="Generate graph statistics")
    
    args = parser.parse_args()
    
    # Set defaults if no specific options chosen
    if not (args.combined or args.individual or args.subgraphs):
        args.combined = True
    
    visualize_graphs(
        input_file=args.input_file,
        output_dir=args.output,
        min_confidence=args.min_confidence,
        layout=args.layout,
        combined=args.combined,
        individual=args.individual,
        subgraphs=args.subgraphs,
        export=args.export,
        stats=args.stats
    )

def filter_zero_triplets(results_file: str, output_file: Optional[str] = None, min_triplets: int = 1):
    """
    Filter out documents with zero or fewer triplets than specified threshold
    
    Args:
        results_file: Path to JSON file with results
        output_file: Path to save filtered results (if None, overwrites original)
        min_triplets: Minimum number of triplets required (default: 1)
    
    Returns:
        Number of documents removed
    """
    import json
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    original_count = len(results)
    
    # Filter results
    filtered_results = [
        result for result in results 
        if result.get('triplets_count', 0) >= min_triplets
    ]
    
    removed_count = original_count - len(filtered_results)
    
    # Save filtered results
    output_path = output_file if output_file is not None else results_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Filtered {removed_count} documents with < {min_triplets} triplets")
    logger.info(f"Remaining: {len(filtered_results)} documents")
    logger.info(f"Saved to: {output_path}")
    
    return removed_count

def analyze_results(results_file: str):
    """
    Analyze results and show statistics about triplet distribution
    
    Args:
        results_file: Path to JSON file with results
    """
    import json
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    total_docs = len(results)
    docs_with_triplets = len([r for r in results if r.get('triplets_count', 0) > 0])
    docs_without_triplets = total_docs - docs_with_triplets
    
    triplet_counts = [r.get('triplets_count', 0) for r in results]
    total_triplets = sum(triplet_counts)
    avg_triplets = total_triplets / total_docs if total_docs > 0 else 0
    
    print(f"""
Results Analysis:
================
Total documents: {total_docs}
Documents with triplets: {docs_with_triplets} ({docs_with_triplets/total_docs*100:.1f}%)
Documents without triplets: {docs_without_triplets} ({docs_without_triplets/total_docs*100:.1f}%)
Total triplets: {total_triplets}
Average triplets per document: {avg_triplets:.2f}

Triplet distribution:
- 0 triplets: {docs_without_triplets} docs
- 1-5 triplets: {len([c for c in triplet_counts if 1 <= c <= 5])} docs
- 6-10 triplets: {len([c for c in triplet_counts if 6 <= c <= 10])} docs
- 11+ triplets: {len([c for c in triplet_counts if c > 10])} docs
    """)