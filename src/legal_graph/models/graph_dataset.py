#!/usr/bin/env python3
"""
BigQuery Legal Knowledge Graph Dataset for PyTorch Geometric

This module provides a dataset class that loads legal document data from BigQuery
and creates PyTorch Geometric graph structures for training the vision-compliant
GraphCheck model.

Usage:
    dataset = GraphDataset(
        table_id="your-project.dataset.table",
        max_nodes=100,
        max_edges=200
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import logging
from transformers import AutoTokenizer
import re

logger = logging.getLogger(__name__)


class GraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for Legal Knowledge Graph training using BigQuery data.
    
    This dataset loads Ukrainian legal documents from BigQuery and converts them
    into PyTorch Geometric graph structures compatible with the vision-compliant
    GraphCheck model.
    """
    
    def __init__(
        self, 
        table_id: str,
        tokenizer_name: str = "bert-base-uncased",
        max_nodes: int = 100,
        max_edges: int = 200,
        max_text_length: int = 512,
        include_legal_references: bool = True,
        node_features_dim: int = 768,  # BERT embedding dimension
        edge_features_dim: int = 128,
        min_triplets: int = 1,
        transform=None,
        pre_transform=None
    ):
        """
        Initialize the BigQuery Legal Graph Dataset
        
        Args:
            table_id: BigQuery table ID (format: "project.dataset.table")
            tokenizer_name: HuggingFace tokenizer for text encoding
            max_nodes: Maximum number of nodes per graph
            max_edges: Maximum number of edges per graph
            max_text_length: Maximum text length for tokenization
            include_legal_references: Whether to include legal reference features
            node_features_dim: Dimension of node features (should match frozen transformer)
            edge_features_dim: Dimension of edge features
            min_triplets: Minimum number of triplets required per document
            transform: Optional transform to apply to each data object
            pre_transform: Optional pre-transform to apply during processing
        """
        self.table_id = table_id
        self.tokenizer_name = tokenizer_name
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_text_length = max_text_length
        self.include_legal_references = include_legal_references
        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.min_triplets = min_triplets
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Legal entity labels mapping (consistent with vision-compliant model)
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
        
        # Ukrainian legal code patterns
        self.legal_code_patterns = {
            r'КК України': 'Criminal Code',
            r'КПК України': 'Criminal Procedure Code',
            r'ЦК України': 'Civil Code',
            r'КоАП України': 'Administrative Code',
            r'СК України': 'Family Code',
            r'КЗпП України': 'Labor Code'
        }
        
        super().__init__(transform=transform, pre_transform=pre_transform)
        
        # Load and process data
        self.raw_data = self._load_from_bigquery()
        self.processed_data = self._process_documents()
        
        # Build vocabularies
        self.node_vocab, self.relation_vocab, self.legal_ref_vocab = self._build_vocabularies()
        
        logger.info(f"Dataset initialized with {len(self.processed_data)} samples")
        logger.info(f"Node vocabulary size: {len(self.node_vocab)}")
        logger.info(f"Relation vocabulary size: {len(self.relation_vocab)}")
        logger.info(f"Legal reference vocabulary size: {len(self.legal_ref_vocab)}")
    
    def _load_from_bigquery(self) -> List[Dict]:
        """Load legal documents from BigQuery"""
        try:
            from google.cloud import bigquery
            
            client = bigquery.Client()
            
            # Query documents with triplets and tags
            sql = f"""
                SELECT 
                    doc_id,
                    text,
                    tags,
                    triplets,
                    triplets_count,
                    CASE 
                        WHEN triplets_count >= {self.min_triplets} THEN 'valid'
                        ELSE 'invalid'
                    END as label
                FROM `{self.table_id}`
                WHERE triplets IS NOT NULL 
                  AND tags IS NOT NULL
                  AND text IS NOT NULL
                  AND LENGTH(text) > 50
                  AND triplets_count >= {self.min_triplets}
                ORDER BY triplets_count DESC
                LIMIT 10000
            """
            
            logger.info(f"Executing BigQuery: {sql}")
            job = client.query(sql)
            results = job.result().to_dataframe()
            
            logger.info(f"Loaded {len(results)} documents from BigQuery")
            
            data = []
            for _, row in results.iterrows():
                try:
                    # Parse triplets
                    triplets = json.loads(row.triplets) if isinstance(row.triplets, str) else row.triplets
                    if not isinstance(triplets, list):
                        continue
                        
                    # Parse entities (tags)
                    entities = json.loads(row.tags) if isinstance(row.tags, str) else row.tags
                    if not isinstance(entities, list):
                        continue
                    
                    # Extract legal references from text
                    legal_references = self._extract_legal_references(str(row.text))
                    
                    # Determine document type from text
                    document_type = self._classify_document_type(str(row.text))
                    
                    data.append({
                        'doc_id': str(row.doc_id),
                        'text': str(row.text),
                        'entities': entities,
                        'triplets': triplets,
                        'triplets_count': int(row.triplets_count),
                        'label': str(row.label),
                        'legal_references': legal_references,
                        'document_type': document_type
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing document {row.get('doc_id', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(data)} documents")
            return data
            
        except Exception as e:
            logger.error(f"Error loading from BigQuery: {e}")
            logger.info("💡 To use BigQuery data:")
            logger.info("   1. Install: pip install google-cloud-bigquery")
            logger.info("   2. Authenticate: gcloud auth application-default login")
            logger.info("   3. Or set: export GOOGLE_APPLICATION_CREDENTIALS='path/to/key.json'")
            logger.info("   4. Or use: dataset.load_from_csv('your_data.csv')")
            logger.warning("Falling back to sample data for demonstration")
            return self._create_sample_data()
    
    def _extract_legal_references(self, text: str) -> List[str]:
        """Extract legal references from text using regex patterns"""
        references = []
        
        # Pattern for Ukrainian legal references
        patterns = [
            r'ч\.\s*\d+\s*ст\.\s*\d+\s*КК\s*України',
            r'ст\.\s*\d+\s*КК\s*України',
            r'ст\.\s*\d+\s*КПК\s*України',
            r'ст\.\s*\d+\s*ЦК\s*України',
            r'ст\.\s*\d+\s*КоАП\s*України',
            r'ст\.\s*\d+\s*СК\s*України',
            r'ст\.\s*\d+\s*КЗпП\s*України'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    def _classify_document_type(self, text: str) -> str:
        """Classify document type based on text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['суд', 'ухвала', 'рішення']):
            return 'court_decision'
        elif any(word in text_lower for word in ['слідчий', 'прокурор', 'розслідування']):
            return 'prosecution_document'
        elif any(word in text_lower for word in ['позов', 'договір', 'цивільна']):
            return 'civil_case'
        elif any(word in text_lower for word in ['адміністративн', 'штраф', 'правопорушення']):
            return 'administrative_case'
        else:
            return 'other'
    
    def _create_sample_data(self) -> List[Dict]:
        """Create sample data when BigQuery is not available"""
        return [
            {
                "doc_id": "sample_001",
                "text": "Приморський районний суд м. Одеси визнав ОСОБА_4 винним у крадіжці згідно з ч.2 ст.185 КК України.",
                "label": "valid",
                "document_type": "court_decision",
                "legal_references": ["ч.2 ст.185 КК України"],
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
                        "legal_reference": "ч.2 ст.185 КК України"
                    }
                ],
                "triplets_count": 1
            }
        ]
    
    def load_from_csv(self, csv_path: str) -> List[Dict]:
        """
        Load data from CSV file as alternative to BigQuery
        
        Args:
            csv_path: Path to CSV file with columns: doc_id, text, tags, triplets, triplets_count
        
        Returns:
            List of processed documents
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} documents from CSV: {csv_path}")
            
            data = []
            for _, row in df.iterrows():
                try:
                    # Parse JSON strings
                    triplets = json.loads(row.triplets) if isinstance(row.triplets, str) else row.triplets
                    entities = json.loads(row.tags) if isinstance(row.tags, str) else row.tags
                    
                    if not isinstance(triplets, list) or not isinstance(entities, list):
                        continue
                    
                    # Extract legal references and classify document
                    legal_references = self._extract_legal_references(str(row.text))
                    document_type = self._classify_document_type(str(row.text))
                    
                    data.append({
                        'doc_id': str(row.doc_id),
                        'text': str(row.text),
                        'entities': entities,
                        'triplets': triplets,
                        'triplets_count': int(row.triplets_count),
                        'label': 'valid' if int(row.triplets_count) >= self.min_triplets else 'invalid',
                        'legal_references': legal_references,
                        'document_type': document_type
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing CSV row {row.get('doc_id', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(data)} documents from CSV")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return self._create_sample_data()
    
    def _process_documents(self) -> List[Dict]:
        """Process raw documents into graph-ready format"""
        processed = []
        
        for doc in self.raw_data:
            try:
                # Create knowledge graph structure
                knowledge_graph = {
                    "entities": doc.get('entities', []),
                    "triplets": doc.get('triplets', [])
                }
                
                processed_doc = {
                    "id": doc['doc_id'],
                    "text": doc['text'],
                    "label": doc['label'],
                    "document_type": doc['document_type'],
                    "legal_references": doc['legal_references'],
                    "knowledge_graph": knowledge_graph
                }
                
                processed.append(processed_doc)
                
            except Exception as e:
                logger.warning(f"Error processing document {doc.get('doc_id', 'unknown')}: {e}")
                continue
        
        return processed
    
    def _build_vocabularies(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """Build vocabularies for nodes, relations, and legal references"""
        node_vocab = defaultdict(int)
        relation_vocab = defaultdict(int)
        legal_ref_vocab = defaultdict(int)
        
        for doc in self.processed_data:
            kg = doc['knowledge_graph']
            
            # Collect nodes (entities)
            for entity in kg.get('entities', []):
                if isinstance(entity, dict) and 'text' in entity:
                    node_vocab[entity['text']] += 1
            
            # Collect relations and legal references
            for triplet in kg.get('triplets', []):
                if isinstance(triplet, dict):
                    if 'relation' in triplet:
                        relation_vocab[triplet['relation']] += 1
                    if self.include_legal_references and 'legal_reference' in triplet:
                        legal_ref_vocab[triplet['legal_reference']] += 1
        
        # Convert to indexed vocabularies
        node_vocab = {term: idx for idx, term in enumerate(node_vocab.keys())}
        relation_vocab = {term: idx for idx, term in enumerate(relation_vocab.keys())}
        legal_ref_vocab = {term: idx for idx, term in enumerate(legal_ref_vocab.keys())}
        
        return node_vocab, relation_vocab, legal_ref_vocab
    
    def len(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.processed_data)
    
    def get(self, idx: int) -> Data:
        """Get a single PyTorch Geometric Data object"""
        doc = self.processed_data[idx]
        
        # Extract entities and create node mappings
        entities = doc['knowledge_graph'].get('entities', [])
        node_mapping = {}
        node_texts = []
        node_labels = []
        
        for i, entity in enumerate(entities[:self.max_nodes]):
            if isinstance(entity, dict) and 'text' in entity:
                node_text = entity['text']
                node_texts.append(node_text)
                node_mapping[node_text] = i
                
                # Get entity label
                entity_label = entity.get('label', 'INFO')
                node_labels.append(self.entity_labels.get(entity_label, 4))  # Default to INFO
        
        if not node_texts:
            # Create dummy node if no entities
            node_texts = ['unknown']
            node_mapping = {'unknown': 0}
            node_labels = [4]  # INFO label
        
        # Create node features using tokenizer
        node_features = []
        for node_text in node_texts:
            # Tokenize node text
            tokens = self.tokenizer(
                node_text,
                max_length=min(32, self.max_text_length // 4),
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Use token embeddings as features (simplified)
            # In practice, you'd use the frozen transformer to get embeddings
            token_ids = tokens['input_ids'].squeeze()
            
            # Create feature vector
            node_feat = torch.zeros(self.node_features_dim)
            if len(token_ids) > 0:
                # Simple encoding: use token IDs modulo feature dimension
                for i, token_id in enumerate(token_ids[:self.node_features_dim]):
                    if token_id != self.tokenizer.pad_token_id:
                        node_feat[i % self.node_features_dim] += float(token_id) / 30000.0
            
            node_features.append(node_feat)
        
        # Stack node features
        x = torch.stack(node_features)
        
        # Create edges from triplets
        edges = []
        edge_attrs = []
        
        triplets = doc['knowledge_graph'].get('triplets', [])
        for triplet in triplets[:self.max_edges]:
            if isinstance(triplet, dict):
                source = triplet.get('source', '')
                target = triplet.get('target', '')
                relation = triplet.get('relation', '')
                
                if source in node_mapping and target in node_mapping:
                    source_idx = node_mapping[source]
                    target_idx = node_mapping[target]
                    
                    edges.append([source_idx, target_idx])
                    
                    # Create edge attributes
                    edge_attr = torch.zeros(self.edge_features_dim)
                    
                    # Encode relation
                    if relation in self.relation_vocab:
                        rel_idx = self.relation_vocab[relation]
                        edge_attr[rel_idx % self.edge_features_dim] = 1.0
                    
                    # Encode legal reference if available
                    if self.include_legal_references and 'legal_reference' in triplet:
                        legal_ref = triplet['legal_reference']
                        if legal_ref in self.legal_ref_vocab:
                            ref_idx = self.legal_ref_vocab[legal_ref]
                            # Use second half of edge features for legal references
                            edge_attr[(ref_idx % (self.edge_features_dim // 2)) + (self.edge_features_dim // 2)] = 1.0
                    
                    edge_attrs.append(edge_attr)
        
        # Handle case with no edges
        if not edges:
            edges = [[0, 0]]  # Self-loop on first node
            edge_attrs = [torch.zeros(self.edge_features_dim)]
        
        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs)
        
        # Create labels
        y = torch.tensor([1 if doc['label'] == 'valid' else 0], dtype=torch.long)
        
        # Additional attributes
        num_nodes = len(node_texts)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes,
            doc_id=doc['id'],
            text=doc['text'],
            legal_references=doc['legal_references'],
            document_type=doc['document_type']
        )
        
        return data
    
    @staticmethod
    def test_bigquery_connection(table_id: str = None) -> bool:
        """
        Test BigQuery connection and authentication
        
        Args:
            table_id: Optional table ID to test access to specific table
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            from google.cloud import bigquery
            
            print("🔄 Testing BigQuery connection...")
            client = bigquery.Client()
            
            # Test basic connection
            print(f"✅ BigQuery client created successfully")
            print(f"📊 Project: {client.project}")
            
            if table_id:
                # Test table access
                print(f"🔍 Testing access to table: {table_id}")
                query = f"SELECT COUNT(*) as count FROM `{table_id}` LIMIT 1"
                job = client.query(query)
                result = job.result()
                count = list(result)[0].count
                print(f"✅ Table accessible, found {count} rows")
            
            print("🎉 BigQuery connection successful!")
            return True
            
        except ImportError:
            print("❌ google-cloud-bigquery not installed")
            print("💡 Install with: pip install google-cloud-bigquery")
            return False
            
        except Exception as e:
            print(f"❌ BigQuery connection failed: {e}")
            print("💡 Authentication options:")
            print("   1. gcloud auth application-default login")
            print("   2. export GOOGLE_APPLICATION_CREDENTIALS='path/to/key.json'")
            print("   3. Use service account key file")
            return False

    def get_vocabulary_info(self) -> Dict:
        """Get vocabulary information for model initialization"""
        return {
            'node_vocab_size': len(self.node_vocab),
            'relation_vocab_size': len(self.relation_vocab),
            'legal_ref_vocab_size': len(self.legal_ref_vocab),
            'node_features_dim': self.node_features_dim,
            'edge_features_dim': self.edge_features_dim,
            'num_classes': 2,  # valid/invalid
            'entity_labels': self.entity_labels
        }


class ReferenceValidationDataset(GraphDataset):
    """Alias for backward compatibility"""
    pass


def create_graph_dataset(table_id: str, **kwargs) -> GraphDataset:
    """Create a graph dataset instance"""
    return GraphDataset(table_id=table_id, **kwargs)


def create_dataloader(
    table_id: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a PyTorch Geometric DataLoader for BigQuery legal data
    
    Args:
        table_id: BigQuery table ID
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for GraphDataset
    
    Returns:
        DataLoader: PyTorch Geometric DataLoader
    """
    dataset = GraphDataset(table_id=table_id, **dataset_kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def load_json_documents(file_path: str) -> List[Dict]:
    """Load documents from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def demonstrate_dataset(table_id: str):
    """Demonstrate the BigQuery dataset functionality"""
    print("🚀 BigQuery Legal Graph Dataset Demo")
    print("=" * 50)
    
    try:
        # Create dataset
        print(f"📊 Loading data from BigQuery table: {table_id}")
        dataset = GraphDataset(
            table_id=table_id,
            max_nodes=50,
            max_edges=100
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Show vocabulary info
        vocab_info = dataset.get_vocabulary_info()
        print(f"\n📚 Vocabulary Information:")
        for key, value in vocab_info.items():
            print(f"   {key}: {value}")
        
        # Get sample data
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n📋 Sample Data Object:")
            print(f"   Node features shape: {sample.x.shape}")
            print(f"   Edge index shape: {sample.edge_index.shape}")
            print(f"   Edge attributes shape: {sample.edge_attr.shape}")
            print(f"   Label: {sample.y.item()}")
            print(f"   Document ID: {sample.doc_id}")
            print(f"   Text: {sample.text[:100]}...")
            print(f"   Legal references: {sample.legal_references}")
        
        # Create dataloader
        dataloader = create_dataloader(
            table_id=table_id,
            batch_size=4,
            shuffle=True
        )
        
        print(f"\n🔄 DataLoader created with batch size 4")
        print(f"   Number of batches: {len(dataloader)}")
        
        # Show batch sample
        for batch in dataloader:
            print(f"\n📦 Sample batch:")
            print(f"   Batch size: {batch.batch.max().item() + 1}")
            print(f"   Total nodes: {batch.x.shape[0]}")
            print(f"   Total edges: {batch.edge_index.shape[1]}")
            print(f"   Labels: {batch.y.tolist()}")
            break
        
        print(f"\n✅ BigQuery dataset demo completed!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        print("💡 Make sure BigQuery credentials are configured and table exists")


if __name__ == "__main__":
    # Example usage
    table_id = "lab-test-project-1-305710.court_data_2022.processing_doc_links"
    demonstrate_dataset(table_id) 