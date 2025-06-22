from typing import Dict, List, Optional, Callable
import spacy
from .entities import BaseMatcherComponent
from .graph_extraction import LegalGraphExtractor, LegalKnowledgeGraph


class DocumentProcessor:
    def __init__(
        self, 
        model_name: str = "gpt-4",
        use_spacy: bool = False,
        custom_entity_extractor: Optional[Callable[[str], List[Dict]]] = None
    ):
        """
        Initialize the document processor
        
        Args:
            model_name: Name of the LLM model to use
            use_spacy: Whether to use spaCy for entity extraction
            custom_entity_extractor: Optional custom function for entity extraction
                                   Should take text as input and return List[Dict]
        """
        self.use_spacy = use_spacy
        self.custom_entity_extractor = custom_entity_extractor
        
        if use_spacy:
            self.nlp = spacy.load("en_core_web_lg")
        
        self.graph_extractor = LegalGraphExtractor(model_name=model_name)
        
    def process_text(self, text: str) -> Dict:
        """
        Process text to extract entities and events
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing text and extracted entities
        """
        entities = []
        
        if self.custom_entity_extractor:
            entities = self.custom_entity_extractor(text)
        elif self.use_spacy:
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
        
        return {
            "text": text,
            "entities": entities
        }
    
    async def extract_events(self, text: str, entities: List[Dict]) -> LegalKnowledgeGraph:
        """Extract events and their relationships"""
        return await self.graph_extractor.extract_events(text, entities)

# Export commonly used components
__all__ = [
    'DocumentProcessor',
    'BaseMatcherComponent',
    'LegalGraphExtractor',
    'LegalKnowledgeGraph'
]
