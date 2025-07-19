#!/usr/bin/env python3
"""
PyTorch Geometric dataset for graph data with entities and triplets.
Handles the JSON structure with knowledge graphs for reference validation.
"""

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import global_mean_pool
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import re


class ReferenceValidationDataset(Dataset):
    """
    PyTorch Geometric dataset for reference validation using graph data.
    Handles entities as nodes and triplets as edges with legal references.
    """
    
    def __init__(self, documents: List[Dict], tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
        super().__init__()
        self.documents = documents
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Legal entity labels mapping
        self.entity_labels = {
            "ORG": 0,    # Organization
            "PER": 1,    # Person
            "LOC": 2,    # Location
            "ROLE": 3,   # Role
            "INFO": 4,   # Information
            "CRIME": 5,  # Crime
            "DTYPE": 6,  # Document Type
            "NUM": 7     # Number
        }
        
        # Document type mapping
        self.document_types = {
            "court_decision": 0,
            "prosecution_document": 1,
            "civil_case": 2,
            "administrative_case": 3
        }
        
        # Process documents into graph data
        self.process_documents()
    
    def process_documents(self):
        """Process documents into graph data structures."""
        self.graph_data = []
        
        for doc in self.documents:
            # Extract entities and triplets
            entities = doc.get('knowledge_graph', {}).get('entities', [])
            triplets = doc.get('knowledge_graph', {}).get('triplets', [])
            
            # Create entity mapping
            entity_map = {}
            for i, entity in enumerate(entities):
                entity_map[entity['text']] = i
            
            # Create node features
            node_features = []
            node_labels = []
            
            for entity in entities:
                # Create entity embedding (simplified - in practice you'd use proper embeddings)
                entity_text = entity['text']
                entity_label = entity['label']
                
                # Tokenize entity text
                tokens = self.tokenizer(
                    entity_text, 
                    max_length=32, 
                    truncation=True, 
                    padding='max_length',
                    return_tensors='pt'
                )
                
                # Create node feature (token embeddings + label encoding)
                label_encoding = torch.zeros(len(self.entity_labels))
                if entity_label in self.entity_labels:
                    label_encoding[self.entity_labels[entity_label]] = 1.0
                
                # Combine token embeddings with label encoding
                node_feature = torch.cat([
                    tokens['input_ids'].squeeze(),
                    label_encoding
                ], dim=0)
                
                node_features.append(node_feature)
                node_labels.append(self.entity_labels.get(entity_label, 0))
            
            # Create edge indices and features
            edge_index = []
            edge_attr = []
            
            for triplet in triplets:
                source = triplet['source']
                target = triplet['target']
                relation = triplet['relation']
                legal_reference = triplet.get('legal_reference', '')
                confidence = triplet.get('confidence', 0.5)
                
                if source in entity_map and target in entity_map:
                    # Add edge from source to target
                    edge_index.append([entity_map[source], entity_map[target]])
                    
                    # Create edge attribute (relation + confidence + reference validation)
                    edge_attr.append([
                        hash(relation) % 1000,  # Simplified relation encoding
                        confidence,
                        self.validate_reference(legal_reference)
                    ])
            
            # Convert to tensors
            if node_features:
                x = torch.stack(node_features)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                y = torch.tensor(self.get_document_label(doc), dtype=torch.long)
                
                # Create PyTorch Geometric Data object
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    num_nodes=len(node_features)
                )
                
                # Add additional attributes
                data.text = doc.get('text', '')
                data.legal_references = doc.get('knowledge_graph', {}).get('triplets', [])
                data.document_id = doc.get('doc_id', '')
                data.entities = entities
                data.triplets = triplets
                
                self.graph_data.append(data)
    
    def validate_reference(self, reference: str) -> float:
        """Validate legal reference and return confidence score."""
        if not reference:
            return 0.0
        
        # Ukrainian legal code patterns
        legal_patterns = {
            'КК України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*КК\s*України',
            'КПК України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*КПК\s*України',
            'ЦК України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*ЦК\s*України',
            'КоАП України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*КоАП\s*України'
        }
        
        # Check against patterns
        for code_name, pattern in legal_patterns.items():
            if re.search(pattern, reference, re.IGNORECASE):
                return 0.9
        
        # Check for common legal reference patterns
        if re.search(r'ст\.\s*\d+', reference):
            return 0.7
        
        return 0.1
    
    def get_document_label(self, doc: Dict) -> int:
        """Get document label for classification."""
        # For now, use a simple heuristic based on reference validity
        triplets = doc.get('knowledge_graph', {}).get('triplets', [])
        valid_refs = sum(1 for t in triplets if self.validate_reference(t.get('legal_reference', '')) > 0.5)
        total_refs = len(triplets)
        
        if total_refs == 0:
            return 0  # Invalid if no references
        
        validity_ratio = valid_refs / total_refs
        return 1 if validity_ratio > 0.5 else 0  # 1 for valid, 0 for invalid
    
    def len(self):
        return len(self.graph_data)
    
    def get(self, idx):
        return self.graph_data[idx]


class GraphDataProcessor:
    """
    Processor for converting JSON documents to PyTorch Geometric graph data.
    """
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def process_document(self, doc: Dict) -> Data:
        """Process a single document into PyTorch Geometric Data."""
        entities = doc.get('knowledge_graph', {}).get('entities', [])
        triplets = doc.get('knowledge_graph', {}).get('triplets', [])
        
        # Create entity mapping
        entity_map = {entity['text']: i for i, entity in enumerate(entities)}
        
        # Create node features
        node_features = []
        for entity in entities:
            # Tokenize entity text
            tokens = self.tokenizer(
                entity['text'],
                max_length=32,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Create node feature
            node_feature = tokens['input_ids'].squeeze()
            node_features.append(node_feature)
        
        # Create edge indices and attributes
        edge_index = []
        edge_attr = []
        
        for triplet in triplets:
            source = triplet['source']
            target = triplet['target']
            relation = triplet['relation']
            legal_reference = triplet.get('legal_reference', '')
            confidence = triplet.get('confidence', 0.5)
            
            if source in entity_map and target in entity_map:
                edge_index.append([entity_map[source], entity_map[target]])
                
                # Edge attributes: [relation_hash, confidence, reference_validity]
                edge_attr.append([
                    hash(relation) % 1000,
                    confidence,
                    self.validate_reference(legal_reference)
                ])
        
        # Convert to tensors
        if node_features:
            x = torch.stack(node_features)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(node_features)
            )
            
            # Add metadata
            data.text = doc.get('text', '')
            data.legal_references = [t.get('legal_reference', '') for t in triplets]
            data.document_id = doc.get('doc_id', '')
            data.entities = entities
            data.triplets = triplets
            
            return data
        
        return None
    
    def validate_reference(self, reference: str) -> float:
        """Validate legal reference and return confidence score."""
        if not reference:
            return 0.0
        
        # Ukrainian legal code patterns
        legal_patterns = {
            'КК України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*КК\s*України',
            'КПК України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*КПК\s*України',
            'ЦК України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*ЦК\s*України',
            'КоАП України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*КоАП\s*України'
        }
        
        # Check against patterns
        for code_name, pattern in legal_patterns.items():
            if re.search(pattern, reference, re.IGNORECASE):
                return 0.9
        
        # Check for common legal reference patterns
        if re.search(r'ст\.\s*\d+', reference):
            return 0.7
        
        return 0.1


def create_sample_graph_data() -> List[Dict]:
    """Create sample graph data for testing."""
    sample_documents = [
        {
            "doc_id": "sample_1",
            "text": "Приморський районний суд м. Одеси визнав ОСОБА_4 винним у крадіжці згідно з ч.2 ст.185 КК України.",
            "knowledge_graph": {
                "entities": [
                    {"text": "Приморський районний суд м. Одеси", "label": "ORG"},
                    {"text": "ОСОБА_4", "label": "PER"},
                    {"text": "крадіжка", "label": "CRIME"}
                ],
                "triplets": [
                    {
                        "source": "ОСОБА_4",
                        "relation": "визнаний_винним",
                        "target": "крадіжка",
                        "legal_reference": "ч.2 ст.185 КК України",
                        "confidence": 0.95
                    },
                    {
                        "source": "Приморський районний суд м. Одеси",
                        "relation": "призначив_покарання",
                        "target": "ОСОБА_4",
                        "legal_reference": "ст.65 КК України",
                        "confidence": 0.90
                    }
                ]
            }
        },
        {
            "doc_id": "sample_2",
            "text": "Суддя ОСОБА_1 ухвалив ухвалу про клопотання слідчого ОСОБА_3.",
            "knowledge_graph": {
                "entities": [
                    {"text": "ОСОБА_1", "label": "PER"},
                    {"text": "ОСОБА_3", "label": "PER"},
                    {"text": "ухвала", "label": "DTYPE"}
                ],
                "triplets": [
                    {
                        "source": "ОСОБА_1",
                        "relation": "ухвалив",
                        "target": "ухвала",
                        "legal_reference": "ст. 219 КПК України",
                        "confidence": 0.90
                    }
                ]
            }
        }
    ]
    
    return sample_documents


def load_json_documents(file_path: str) -> List[Dict]:
    """Load documents from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    return documents


def create_graph_dataset(documents: List[Dict], tokenizer_name: str = "bert-base-uncased") -> ReferenceValidationDataset:
    """Create a PyTorch Geometric dataset from documents."""
    return ReferenceValidationDataset(documents, tokenizer_name)


if __name__ == "__main__":
    # Test the dataset
    sample_docs = create_sample_graph_data()
    dataset = create_graph_dataset(sample_docs)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first graph
    if len(dataset) > 0:
        graph = dataset[0]
        print(f"Graph nodes: {graph.num_nodes}")
        print(f"Graph edges: {graph.edge_index.shape[1]}")
        print(f"Graph features: {graph.x.shape}")
        print(f"Graph label: {graph.y}")
        print(f"Document ID: {graph.document_id}") 