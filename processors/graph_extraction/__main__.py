import asyncio
import json
from typing import List, Dict
from . import GraphExtractor

async def process_document(text: str, entities: List[Dict]) -> Dict:
    """Process a document to extract event graph"""
    extractor = GraphExtractor()
    graph = await extractor.extract_events(text, entities)
    return graph.dict()

async def main():
    # Example usage
    text = """
    The defendant was charged with theft on January 15, 2023. 
    The court found the defendant guilty and sentenced them to 2 years in prison.
    The defense attorney filed an appeal on February 1, 2023.
    """
    
    entities = [
        {"text": "defendant", "label": "PERSON", "start": 4, "end": 13},
        {"text": "January 15, 2023", "label": "DATE", "start": 35, "end": 51},
        {"text": "court", "label": "ORG", "start": 57, "end": 62},
        {"text": "2 years", "label": "DURATION", "start": 95, "end": 102},
        {"text": "defense attorney", "label": "PERSON", "start": 120, "end": 135},
        {"text": "February 1, 2023", "label": "DATE", "start": 150, "end": 166}
    ]
    
    result = await process_document(text, entities)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 