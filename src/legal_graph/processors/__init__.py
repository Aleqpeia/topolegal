"""
TopoLegal Processors Module
==========================

This module provides comprehensive legal document processing capabilities including:
- Entity extraction and Named Entity Recognition (NER)
- Legal knowledge graph extraction and triplet generation
- Data corruption for binary classification training

Main Components:
- Entity Processing: spaCy-based NER for legal documents
- Graph Extraction: LLM-powered knowledge graph generation
- Data Corruption: Legal reference corruption for dataset balancing
- Document Processing: Unified interface for legal document analysis
"""

import logging
from typing import Dict, List, Optional, Callable, Union
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Core Entity Processing
# ============================================================================

try:
    from .entities import BaseMatcherComponent
    ENTITIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Entities module not fully available: {e}")
    BaseMatcherComponent = None
    ENTITIES_AVAILABLE = False

# ============================================================================
# Knowledge Graph Extraction
# ============================================================================

try:
    from .graph_extraction import (
        LegalGraphExtractor,
        LegalKnowledgeGraph,
        KnowledgeTriplet,
        LegalEntity,
        process_legal_document,
        visualize_graphs,
        filter_zero_triplets,
        analyze_results
    )
    GRAPH_EXTRACTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Graph extraction module not fully available: {e}")
    LegalGraphExtractor = None
    LegalKnowledgeGraph = None
    KnowledgeTriplet = None
    LegalEntity = None
    process_legal_document = None
    visualize_graphs = None
    filter_zero_triplets = None
    analyze_results = None
    GRAPH_EXTRACTION_AVAILABLE = False

# ============================================================================
# Data Corruption and Balance
# ============================================================================

try:
    from .balance import (
        DataCorruptor,
        LegalReferenceCorruptor,
        CorruptionResult,
        BinaryClassificationCorruptor  # Backward compatibility
    )
    BALANCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Balance module not fully available: {e}")
    DataCorruptor = None
    LegalReferenceCorruptor = None
    CorruptionResult = None
    BinaryClassificationCorruptor = None
    BALANCE_AVAILABLE = False



# ============================================================================
# Unified Document Processor
# ============================================================================

class DocumentProcessor:
    """
    Unified interface for legal document processing.
    
    Combines entity extraction, knowledge graph generation, and optional corruption
    for comprehensive legal document analysis.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        use_spacy: bool = False,
        spacy_model: str = "uk_core_news_trf",
        custom_entity_extractor: Optional[Callable[[str], List[Dict]]] = None,
        enable_corruption: bool = False,
        corruption_probability: float = 0.5,
        temperature: float = 0.8
    ):
        """
        Initialize the unified document processor.
        
        Args:
            model_name: LLM model name for graph extraction
            use_spacy: Whether to use spaCy for entity extraction
            spacy_model: spaCy model to use
            custom_entity_extractor: Custom entity extraction function
            enable_corruption: Whether to enable data corruption capabilities
            corruption_probability: Probability for data corruption
            temperature: LLM temperature for graph extraction
        """
        self.use_spacy = use_spacy
        self.custom_entity_extractor = custom_entity_extractor
        self.enable_corruption = enable_corruption
        
        # Initialize spaCy if requested
        self.nlp = None
        if use_spacy and ENTITIES_AVAILABLE:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model {spacy_model}: {e}")
                self.use_spacy = False
        
        # Initialize graph extractor
        self.graph_extractor = None
        if GRAPH_EXTRACTION_AVAILABLE:
            try:
                self.graph_extractor = LegalGraphExtractor(
                    model_name=model_name,
                    temperature=temperature
                )
                logger.info(f"Initialized graph extractor with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize graph extractor: {e}")
        
        # Initialize corruptor if requested
        self.corruptor = None
        if enable_corruption and BALANCE_AVAILABLE:
            try:
                self.corruptor = LegalReferenceCorruptor(
                    corruption_probability=corruption_probability
                )
                logger.info(f"Initialized data corruptor with probability: {corruption_probability}")
            except Exception as e:
                logger.warning(f"Failed to initialize corruptor: {e}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using the configured method.
        
        Args:
            text: Input text to process
            
        Returns:
            List of entity dictionaries with keys: text, label, start, end
        """
        entities = []
        
        if self.custom_entity_extractor:
            entities = self.custom_entity_extractor(text)
        elif self.use_spacy and self.nlp:
            # Process with spaCy
            doc = self.nlp(text)
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]
        else:
            logger.warning("No entity extraction method configured")
        
        return entities
    
    async def extract_knowledge_graph(self, text: str, entities: Optional[List[Dict]] = None) -> Optional[LegalKnowledgeGraph]:
        """
        Extract knowledge graph from text and entities.
        
        Args:
            text: Input text
            entities: Optional pre-extracted entities (will extract if not provided)
            
        Returns:
            LegalKnowledgeGraph object or None if extraction fails
        """
        if not self.graph_extractor:
            logger.error("Graph extractor not available")
            return None
        
        if entities is None:
            entities = self.extract_entities(text)
        
        try:
            return await self.graph_extractor.extract_triplets(text, entities)
        except Exception as e:
            logger.error(f"Failed to extract knowledge graph: {e}")
            return None
    
    async def process_document(self, text: str, extract_entities: bool = True, 
                              extract_graph: bool = True) -> Dict:
        """
        Comprehensive document processing.
        
        Args:
            text: Input text to process
            extract_entities: Whether to extract entities
            extract_graph: Whether to extract knowledge graph
            
        Returns:
            Dictionary containing processing results
        """
        result = {
            "text": text,
            "entities": [],
            "knowledge_graph": None,
            "processing_stats": {}
        }
        
        # Extract entities
        if extract_entities:
            try:
                result["entities"] = self.extract_entities(text)
                result["processing_stats"]["entities_count"] = len(result["entities"])
                logger.info(f"Extracted {len(result['entities'])} entities")
            except Exception as e:
                logger.error(f"Entity extraction failed: {e}")
        
        # Extract knowledge graph
        if extract_graph and self.graph_extractor:
            try:
                kg = await self.extract_knowledge_graph(text, result["entities"])
                if kg:
                    result["knowledge_graph"] = kg
                    result["processing_stats"]["triplets_count"] = len(kg.triplets)
                    result["processing_stats"]["legal_references"] = len(kg.legal_references)
                    logger.info(f"Extracted {len(kg.triplets)} triplets")
            except Exception as e:
                logger.error(f"Knowledge graph extraction failed: {e}")
        
        return result
    
    def corrupt_data(self, dataset: List[Dict], label_column: str = "is_valid") -> List[Dict]:
        """
        Apply data corruption to a dataset.
        
        Args:
            dataset: List of data samples
            label_column: Column name for corruption labels
            
        Returns:
            Corrupted dataset with labels
        """
        if not self.corruptor:
            logger.error("Data corruptor not available")
            return dataset
        
        try:
            return self.corruptor.corrupt_dataset(dataset, label_column=label_column)
        except Exception as e:
            logger.error(f"Data corruption failed: {e}")
            return dataset
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get information about available capabilities."""
        return {
            "entity_extraction": ENTITIES_AVAILABLE and (self.use_spacy or self.custom_entity_extractor is not None),
            "knowledge_graph_extraction": GRAPH_EXTRACTION_AVAILABLE and self.graph_extractor is not None,
            "data_corruption": BALANCE_AVAILABLE and self.corruptor is not None,
            "spacy_available": self.use_spacy and self.nlp is not None
        }

# ============================================================================
# Entity Processing Utilities
# ============================================================================

def build_entity_pipeline(use_pretrained: bool = True, model_name: str = "uk_core_news_trf"):
    """
    Build a complete spaCy NER pipeline for legal documents.
    
    Args:
        use_pretrained: Whether to use pretrained model
        model_name: spaCy model name
        
    Returns:
        spaCy Language object with legal NER components
    """
    if not ENTITIES_AVAILABLE:
        raise ImportError("Entities module not available. Install required dependencies.")
    
    try:
        import spacy
        from .entities.__main__ import build_nlp
        return build_nlp(use_pretrained)
    except Exception as e:
        logger.error(f"Failed to build entity pipeline: {e}")
        raise

def analyze_entities(text: str, nlp=None) -> str:
    """
    Analyze entities in text and return JSON result.
    
    Args:
        text: Text to analyze
        nlp: Optional spaCy pipeline (will create if not provided)
        
    Returns:
        JSON string with entity analysis
    """
    if not ENTITIES_AVAILABLE:
        raise ImportError("Entities module not available")
    
    try:
        from .entities.__main__ import analyse
        if nlp is None:
            nlp = build_entity_pipeline()
        return analyse(text, nlp)
    except Exception as e:
        logger.error(f"Entity analysis failed: {e}")
        raise

# ============================================================================
# Module Status and Diagnostics
# ============================================================================

def get_module_status() -> Dict[str, Union[bool, str]]:
    """Get status of all processor modules."""
    return {
        "entities_available": ENTITIES_AVAILABLE,
        "graph_extraction_available": GRAPH_EXTRACTION_AVAILABLE,
        "balance_available": BALANCE_AVAILABLE,
        "all_modules_available": all([ENTITIES_AVAILABLE, GRAPH_EXTRACTION_AVAILABLE, BALANCE_AVAILABLE])
    }

def check_dependencies() -> Dict[str, Union[bool, str]]:
    """Check if all required dependencies are available."""
    deps = {}
    
    # Check spaCy
    try:
        import spacy
        deps["spacy"] = True
    except ImportError:
        deps["spacy"] = "pip install spacy"
    
    # Check LangChain
    try:
        from langchain_community.chat_models import ChatOpenAI
        deps["langchain"] = True
    except ImportError:
        deps["langchain"] = "pip install langchain-community"
    
    # Check pandas
    try:
        import pandas
        deps["pandas"] = True
    except ImportError:
        deps["pandas"] = "pip install pandas"
    
    # Check pydantic
    try:
        import pydantic
        deps["pydantic"] = True
    except ImportError:
        deps["pydantic"] = "pip install pydantic"
    
    # Check BigQuery
    try:
        from google.cloud import bigquery
        deps["bigquery"] = True
    except ImportError:
        deps["bigquery"] = "pip install google-cloud-bigquery"
    
    return deps

def print_module_info():
    """Print comprehensive module information."""
    print("TopoLegal Processors Module Status")
    print("=" * 40)
    
    status = get_module_status()
    for module, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"{status_icon} {module}: {available}")
    
    print("\nDependency Status")
    print("-" * 20)
    deps = check_dependencies()
    for dep, status in deps.items():
        if status is True:
            print(f"✅ {dep}: Available")
        else:
            print(f"❌ {dep}: {status}")
    
    print(f"\nAvailable Classes and Functions:")
    print("-" * 30)
    available_items = [item for item in __all__ if globals().get(item) is not None]
    for item in available_items:
        print(f"• {item}")

# ============================================================================
# Legacy Compatibility
# ============================================================================

# Backward compatibility for existing code
def extract_events(text: str, entities: List[Dict]):
    """Legacy function - use DocumentProcessor.extract_knowledge_graph instead."""
    warnings.warn(
        "extract_events is deprecated. Use DocumentProcessor.extract_knowledge_graph instead.",
        DeprecationWarning,
        stacklevel=2
    )
    processor = DocumentProcessor()
    import asyncio
    return asyncio.run(processor.extract_knowledge_graph(text, entities))

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core processor
    'DocumentProcessor',
    
    # Entity processing
    'BaseMatcherComponent',
    'build_entity_pipeline',
    'analyze_entities',
    
    # Knowledge graph extraction
    'LegalGraphExtractor',
    'LegalKnowledgeGraph', 
    'KnowledgeTriplet',
    'LegalEntity',
    'process_legal_document',
    'visualize_graphs',
    'filter_zero_triplets',
    'analyze_results',
    
    # Data corruption
    'DataCorruptor',
    'LegalReferenceCorruptor',
    'CorruptionResult',
    'BinaryClassificationCorruptor',
    

    
    # Utilities
    'get_module_status',
    'check_dependencies',
    'print_module_info',
    
    # Legacy (deprecated)
    'extract_events',
]

# Remove None values from exports
__all__ = [item for item in __all__ if globals().get(item) is not None]

# ============================================================================
# Module Initialization
# ============================================================================

logger.info("TopoLegal Processors module initialized")
logger.info(f"Available modules: {[k for k, v in get_module_status().items() if v and k != 'all_modules_available']}")

# Show warnings for missing modules
missing_modules = [k for k, v in get_module_status().items() if not v and k != 'all_modules_available']
if missing_modules:
    logger.warning(f"Some modules are not available: {missing_modules}")
    logger.warning("Run processors.check_dependencies() for installation instructions")
