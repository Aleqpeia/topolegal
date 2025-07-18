import torch
from torch.utils.data import Dataset
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class LegalKnowledgeGraphDataset(Dataset):
    """
    PyTorch Dataset for Legal Knowledge Graph training
    Handles triplet data from BigQuery or CSV sources
    """
    
    def __init__(
        self, 
        data_source: str,  # 'bigquery' or 'csv'
        table_id: Optional[str] = None,
        csv_file: Optional[str] = None,
        max_nodes: int = 100,
        max_edges: int = 200,
        include_legal_references: bool = True,
        node_features_dim: int = 128,
        edge_features_dim: int = 64,
        transform=None
    ):
        """
        Initialize the dataset
        
        Args:
            data_source: 'bigquery' or 'csv'
            table_id: BigQuery table ID (for bigquery mode)
            csv_file: CSV file path (for csv mode)
            max_nodes: Maximum number of nodes per graph
            max_edges: Maximum number of edges per graph
            include_legal_references: Whether to include legal reference features
            node_features_dim: Dimension of node features
            edge_features_dim: Dimension of edge features
            transform: Optional transform to apply
        """
        self.data_source = data_source
        self.table_id = table_id
        self.csv_file = csv_file
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.include_legal_references = include_legal_references
        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
        # Build vocabulary and feature encoders
        self.node_vocab, self.edge_vocab, self.legal_ref_vocab = self._build_vocabularies()
        
        # Create feature encoders
        self.node_encoder = self._create_node_encoder()
        self.edge_encoder = self._create_edge_encoder()
        self.legal_ref_encoder = self._create_legal_ref_encoder()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
        logger.info(f"Node vocabulary size: {len(self.node_vocab)}")
        logger.info(f"Edge vocabulary size: {len(self.edge_vocab)}")
        logger.info(f"Legal reference vocabulary size: {len(self.legal_ref_vocab)}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from BigQuery or CSV"""
        if self.data_source == 'bigquery':
            return self._load_from_bigquery()
        elif self.data_source == 'csv':
            return self._load_from_csv()
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
    
    def _load_from_bigquery(self) -> List[Dict]:
        """Load data from BigQuery"""
        try:
            from google.cloud import bigquery
            
            client = bigquery.Client()
            
            # Query documents with triplets
            sql = f"""
                SELECT doc_id, text, tags, triplets, triplets_count
                FROM `{self.table_id}`
                WHERE triplets IS NOT NULL 
                  AND triplets_count > 0
                  AND text IS NOT NULL
                  AND tags IS NOT NULL
                ORDER BY triplets_count DESC
            """
            
            job = client.query(sql)
            results = job.result().to_dataframe()
            
            data = []
            for _, row in results.iterrows():
                try:
                    # Parse triplets
                    triplets = json.loads(row.triplets) if isinstance(row.triplets, str) else row.triplets
                    
                    # Parse entities
                    entities = json.loads(row.tags) if isinstance(row.tags, str) else row.tags
                    
                    data.append({
                        'doc_id': str(row.doc_id),
                        'text': str(row.text),
                        'entities': entities,
                        'triplets': triplets,
                        'triplets_count': int(row.triplets_count)
                    })
                except Exception as e:
                    logger.warning(f"Error parsing row {row.doc_id}: {e}")
                    continue
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading from BigQuery: {e}")
            return []
    
    def _load_from_csv(self) -> List[Dict]:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(self.csv_file)
            
            data = []
            for _, row in df.iterrows():
                try:
                    # Parse triplets
                    triplets = json.loads(row.triplets) if pd.notna(row.triplets) else []
                    
                    # Parse entities
                    entities = json.loads(row.tags) if pd.notna(row.tags) else []
                    
                    data.append({
                        'doc_id': str(row.get('doc_id', f"doc_{len(data)}")),
                        'text': str(row.text),
                        'entities': entities,
                        'triplets': triplets,
                        'triplets_count': len(triplets)
                    })
                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading from CSV: {e}")
            return []
    
    def _build_vocabularies(self) -> Tuple[Dict, Dict, Dict]:
        """Build vocabularies for nodes, edges, and legal references"""
        node_vocab = defaultdict(int)
        edge_vocab = defaultdict(int)
        legal_ref_vocab = defaultdict(int)
        
        for sample in self.data:
            # Collect nodes (entities)
            for entity in sample['entities']:
                node_vocab[entity['text']] += 1
            
            # Collect edges (relations)
            for triplet in sample['triplets']:
                edge_vocab[triplet['relation']] += 1
                if self.include_legal_references and triplet.get('legal_reference'):
                    legal_ref_vocab[triplet['legal_reference']] += 1
        
        # Convert to dictionaries with indices
        node_vocab = {k: i for i, k in enumerate(node_vocab.keys())}
        edge_vocab = {k: i for i, k in enumerate(edge_vocab.keys())}
        legal_ref_vocab = {k: i for i, k in enumerate(legal_ref_vocab.keys())}
        
        return dict(node_vocab), dict(edge_vocab), dict(legal_ref_vocab)
    
    def _create_node_encoder(self):
        """Create node feature encoder"""
        def encode_node(node_text: str) -> torch.Tensor:
            # Simple one-hot encoding based on vocabulary
            if node_text in self.node_vocab:
                idx = self.node_vocab[node_text]
                features = torch.zeros(self.node_features_dim)
                features[idx % self.node_features_dim] = 1.0
                return features
            else:
                # Unknown node - use zero vector
                return torch.zeros(self.node_features_dim)
        
        return encode_node
    
    def _create_edge_encoder(self):
        """Create edge feature encoder"""
        def encode_edge(relation: str) -> torch.Tensor:
            # Simple one-hot encoding based on vocabulary
            if relation in self.edge_vocab:
                idx = self.edge_vocab[relation]
                features = torch.zeros(self.edge_features_dim)
                features[idx % self.edge_features_dim] = 1.0
                return features
            else:
                # Unknown relation - use zero vector
                return torch.zeros(self.edge_features_dim)
        
        return encode_edge
    
    def _create_legal_ref_encoder(self):
        """Create legal reference feature encoder"""
        def encode_legal_ref(legal_ref: str) -> torch.Tensor:
            # Simple one-hot encoding based on vocabulary
            if legal_ref in self.legal_ref_vocab:
                idx = self.legal_ref_vocab[legal_ref]
                features = torch.zeros(self.edge_features_dim)  # Use same dim as edge features
                features[idx % self.edge_features_dim] = 1.0
                return features
            else:
                # Unknown legal reference - use zero vector
                return torch.zeros(self.edge_features_dim)
        
        return encode_legal_ref
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single graph sample"""
        sample = self.data[idx]
        
        # Extract nodes from entities
        nodes = []
        node_mapping = {}  # Map entity text to node index
        
        for i, entity in enumerate(sample['entities'][:self.max_nodes]):
            node_text = entity['text']
            nodes.append(node_text)
            node_mapping[node_text] = i
        
        # Create node features
        node_features = torch.stack([
            self.node_encoder(node_text) for node_text in nodes
        ])
        
        # Extract edges from triplets
        edges = []
        edge_features = []
        
        for triplet in sample['triplets'][:self.max_edges]:
            source = triplet['source']
            target = triplet['target']
            relation = triplet['relation']
            
            # Only include edges where both nodes exist
            if source in node_mapping and target in node_mapping:
                source_idx = node_mapping[source]
                target_idx = node_mapping[target]
                
                edges.append([source_idx, target_idx])
                
                # Create edge features
                edge_feat = self.edge_encoder(relation)
                if self.include_legal_references and triplet.get('legal_reference'):
                    legal_ref_feat = self.legal_ref_encoder(triplet['legal_reference'])
                    edge_feat = torch.cat([edge_feat, legal_ref_feat])
                
                edge_features.append(edge_feat)
        
        if not edges:
            # Create dummy edge if no edges exist
            edges = [[0, 0]]
            edge_features = [torch.zeros(self.edge_features_dim * (2 if self.include_legal_references else 1))]
        
        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_features = torch.stack(edge_features)
        
        # Pad to max dimensions
        node_features = self._pad_tensor(node_features, self.max_nodes, self.node_features_dim)
        edge_features = self._pad_tensor(edge_features, self.max_edges, edge_features.size(-1))
        
        # Create graph data
        graph_data = {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'doc_id': sample['doc_id'],
            'triplets_count': sample['triplets_count']
        }
        
        if self.transform:
            graph_data = self.transform(graph_data)
        
        return graph_data
    
    def _pad_tensor(self, tensor: torch.Tensor, max_size: int, feature_dim: int) -> torch.Tensor:
        """Pad tensor to max_size"""
        if tensor.size(0) < max_size:
            padding = torch.zeros(max_size - tensor.size(0), feature_dim)
            tensor = torch.cat([tensor, padding], dim=0)
        elif tensor.size(0) > max_size:
            tensor = tensor[:max_size]
        return tensor
    
    def get_vocabulary_info(self) -> Dict:
        """Get vocabulary information for model initialization"""
        return {
            'node_vocab_size': len(self.node_vocab),
            'edge_vocab_size': len(self.edge_vocab),
            'legal_ref_vocab_size': len(self.legal_ref_vocab),
            'node_features_dim': self.node_features_dim,
            'edge_features_dim': self.edge_features_dim
        }


class LegalGraphCollate:
    """Custom collate function for batching graphs"""
    
    def __init__(self, max_nodes: int = 100, max_edges: int = 200):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of graphs"""
        # Stack node features
        node_features = torch.stack([item['node_features'] for item in batch])
        
        # Stack edge features
        edge_features = torch.stack([item['edge_features'] for item in batch])
        
        # Create batch index for edge_index
        edge_indices = []
        node_counts = []
        
        for i, item in enumerate(batch):
            edge_index = item['edge_index']
            num_nodes = item['num_nodes']
            
            # Adjust edge indices for batch
            edge_index[0] += i * self.max_nodes
            edge_index[1] += i * self.max_nodes
            
            edge_indices.append(edge_index)
            node_counts.append(num_nodes)
        
        # Concatenate edge indices
        edge_index = torch.cat(edge_indices, dim=1)
        
        # Create batch index for nodes
        batch_index = torch.cat([
            torch.full((item['num_nodes'],), i, dtype=torch.long)
            for i, item in enumerate(batch)
        ])
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'batch_index': batch_index,
            'num_nodes': torch.tensor(node_counts),
            'doc_ids': [item['doc_id'] for item in batch],
            'triplets_counts': torch.tensor([item['triplets_count'] for item in batch])
        }


def create_dataloader(
    dataset: LegalKnowledgeGraphDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the legal knowledge graph dataset"""
    collate_fn = LegalGraphCollate(
        max_nodes=dataset.max_nodes,
        max_edges=dataset.max_edges
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    ) 