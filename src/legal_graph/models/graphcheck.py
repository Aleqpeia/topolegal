#!/usr/bin/env python3
"""
Vision-Compliant GraphCheck Implementation
Matches the user's diagram vision with proper synthetic data step and data flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModelForCausalLM
import contextlib
from typing import List, Dict, Tuple, Optional
import json
import re


class EntityExtractor(nn.Module):
    """Trainable NER model for legal entity extraction."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_legal_labels: int = 8):
        super().__init__()
        from transformers import AutoModelForTokenClassification
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=num_legal_labels
        )
        
        # Legal entity labels
        self.label_map = {
            "ORG": 0,    # Organization
            "PER": 1,    # Person
            "LOC": 2,    # Location
            "ROLE": 3,   # Role
            "INFO": 4,   # Information
            "CRIME": 5,  # Crime
            "DTYPE": 6,  # Document Type
            "NUM": 7     # Number
        }
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def extract_legal_entities(self, text: str) -> List[Dict]:
        """Extract legal entities from text using trainable NER."""
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            return_offsets_mapping=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Convert predictions to entities
        entities = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        offset_mapping = inputs["offset_mapping"][0]
        
        current_entity = None
        
        for i, (token, pred, offset) in enumerate(zip(tokens, predictions[0], offset_mapping)):
            if pred != 0:  # Not O (Outside)
                label = list(self.label_map.keys())[pred.item()]
                
                if current_entity is None:
                    current_entity = {
                        "text": token,
                        "label": label,
                        "start": offset[0],
                        "end": offset[1],
                        "confidence": 0.8
                    }
                else:
                    # Extend current entity
                    current_entity["text"] += " " + token
                    current_entity["end"] = offset[1]
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities


class SyntheticDataProcessor(nn.Module):
    """Process extracted entities into synthetic data (JSON/Graph Nodes)."""
    
    def __init__(self):
        super().__init__()
    
    def process_entities_to_synthetic(self, entities: List[Dict], text: str) -> Dict:
        """Convert extracted entities to synthetic data structure."""
        # Create synthetic data structure
        synthetic_data = {
            "entities": entities,
            "entity_count": len(entities),
            "text": text,
            "graph_nodes": [],
            "json_structure": {}
        }
        
        # Create graph nodes from entities
        for i, entity in enumerate(entities):
            node = {
                "id": i,
                "text": entity["text"],
                "label": entity["label"],
                "start": entity["start"],
                "end": entity["end"],
                "confidence": entity["confidence"],
                "node_type": self._classify_node_type(entity["text"], entity["label"])
            }
            synthetic_data["graph_nodes"].append(node)
        
        # Create JSON structure
        synthetic_data["json_structure"] = {
            "entities": entities,
            "graph_nodes": synthetic_data["graph_nodes"],
            "metadata": {
                "text_length": len(text),
                "entity_count": len(entities),
                "processing_timestamp": "2024-01-01T00:00:00Z"
            }
        }
        
        return synthetic_data
    
    def _classify_node_type(self, text: str, label: str) -> str:
        """Classify node type based on text and label."""
        if label == "ORG":
            return "organization"
        elif label == "PER":
            return "person"
        elif label == "LOC":
            return "location"
        elif label == "CRIME":
            return "crime"
        else:
            return "other"


class GraphEncoder(nn.Module):
    """Trainable GNN for encoding legal knowledge graphs."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout
        self.attn_weights = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters
        for bn in self.bns:
            bn.reset_parameters

    def forward(self, x, edge_index, edge_attr=None):
        attn_weights_list = []
        for i, conv in enumerate(self.convs[:-1]):
            x, attn_weights = conv(x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=True)
            attn_weights_list.append(attn_weights[1])
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x, attn_weights = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=True)
        attn_weights_list.append(attn_weights[1])
        self.attn_weights = attn_weights_list[-1]
        
        return x, edge_attr


class Projector(nn.Module):
    """Trainable projector to map between embedding spaces."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.projector(x)


class AttentionFusion(nn.Module):
    """Trainable attention fusion layer."""
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        # Multi-head attention
        attn_output, _ = self.attention(query, key, value)
        attn_output = self.dropout(attn_output)
        
        # Residual connection and normalization
        output = self.norm1(query + attn_output)
        
        return output


class GraphCheck(nn.Module):
    """
    Vision-compliant GraphCheck that matches the user's diagram.
    Implements the exact data flow: INPUT → SYNTHETIC → GNN → PROJECTOR → FUSION
    """
    
    def __init__(self, args):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        # Setup device and memory management
        num_devices = torch.cuda.device_count()   
        max_memory = {}
        for i in range(num_devices):
            total_memory = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
            max_memory[i] = f"{max(total_memory - 2, 2)}GiB"     
        
        kwargs = {
            "max_memory": max_memory,
            "device_map": "auto",
            "revision": "main",
        }
        
        # 🔒 FROZEN COMPONENTS
        # Load frozen transformer (LLM)
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )
        
        # Freeze LLM parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        model.gradient_checkpointing_enable()
        self.model = model
        print('✅ Finished loading frozen model')

        self.word_embedding = self.model.model.get_input_embeddings()

        # 🔄 TRAINABLE COMPONENTS
        # Trainable NER model for legal entities
        self.ner_model = EntityExtractor(
            model_name=args.ner_model_name,
            num_legal_labels=args.num_legal_labels
        ).to(self.model.device)
        
        # Synthetic data processor
        self.synthetic_processor = SyntheticDataProcessor()
        
        # Trainable GNN for legal graph encoding
        self.graph_encoder = GraphEncoder(
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)
        
        # Trainable projector
        self.projector = Projector(
            input_dim=args.gnn_hidden_dim,
            output_dim=self.word_embedding.weight.shape[1]
        ).to(self.model.device)
        
        # Trainable fusion layer
        self.fusion = AttentionFusion(
            hidden_size=self.word_embedding.weight.shape[1]
        ).to(self.model.device)

        self.embed_dim = self.word_embedding.weight.shape[1]
        self.gnn_output = args.gnn_hidden_dim

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def get_frozen_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings from frozen transformer."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():  # 🔒 NO GRADIENTS
            embedding = self.model.model.embed_tokens(inputs["input_ids"].to(self.device))
            return torch.mean(embedding, dim=1).squeeze(0)
    
    def process_input_to_synthetic(self, text: str) -> Dict:
        """Process input text to synthetic data (matches diagram flow)."""
        # Step 1: Extract entities using trainable NER
        entities = self.ner_model.extract_legal_entities(text)
        
        # Step 2: Process to synthetic data
        synthetic_data = self.synthetic_processor.process_entities_to_synthetic(entities, text)
        
        return synthetic_data
    
    def build_graph_from_synthetic(self, synthetic_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph from synthetic data using frozen embeddings."""
        entities = synthetic_data["entities"]
        
        # Get frozen embeddings for entities
        entity_embeddings = []
        for entity in entities:
            if entity['confidence'] > 0.5:
                # Use frozen transformer to get embeddings
                frozen_emb = self.get_frozen_embeddings(entity['text'])
                entity_embeddings.append(frozen_emb)
        
        if not entity_embeddings:
            # If no entities found, use the entire text
            frozen_emb = self.get_frozen_embeddings(synthetic_data["text"])
            entity_embeddings = [frozen_emb]
        
        # Stack embeddings
        node_features = torch.stack(entity_embeddings)
        
        # Create edges (fully connected graph)
        num_nodes = len(entity_embeddings)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
        
        return node_features, edge_index
    
    def forward(self, data):
        """
        Forward pass matching the diagram flow:
        INPUT → SYNTHETIC → GNN → PROJECTOR → FUSION → OUTPUT
        """
        batch_size = len(data['id'])
        all_graph_embeds = []
        
        for i in range(batch_size):
            text = data['text'][i]
            
            # Step 1: INPUT → SYNTHETIC (Trainable NER)
            synthetic_data = self.process_input_to_synthetic(text)
            
            # Step 2: SYNTHETIC → GNN (with frozen embeddings)
            node_features, edge_index = self.build_graph_from_synthetic(synthetic_data)
            
            # Step 3: GNN processing (Trainable)
            if edge_index.size(1) > 0:
                node_embeds, _ = self.graph_encoder(node_features, edge_index)
                graph_embed = global_mean_pool(node_embeds, torch.zeros(node_embeds.size(0), dtype=torch.long).to(self.device))
            else:
                graph_embed = torch.mean(node_features, dim=0, keepdim=True)
            
            # Step 4: PROJECTOR (Trainable)
            projected_embed = self.projector(graph_embed)
            all_graph_embeds.append(projected_embed)
        
        # Stack all graph embeddings
        graph_embeds = torch.stack(all_graph_embeds)
        
        # Step 5: FUSION (Trainable)
        # Get frozen embeddings for text
        frozen_embeds = []
        for text in data['text']:
            frozen_emb = self.get_frozen_embeddings(text)
            frozen_embeds.append(frozen_emb)
        frozen_embeds = torch.stack(frozen_embeds)
        
        # Fuse projected GNN output with frozen embeddings
        fused_embeds = self.fusion(graph_embeds, frozen_embeds, frozen_embeds)
        
        # Step 6: OUTPUT (classification)
        # Use frozen transformer for final processing
        texts = self.tokenizer(data["text"], add_special_tokens=False)
        labels = self.tokenizer(data["label"], add_special_tokens=False)

        # Encode special tokens
        eos_tokens = self.tokenizer("</s>", add_special_tokens=False)
        eos_user_tokens = self.tokenizer("<|endoftext|>", add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer("<|endoftext|>", add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids   
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            
            # Add fused embeddings
            fused_embedding = fused_embeds[i].unsqueeze(0)
            inputs_embeds = torch.cat([bos_embeds, fused_embedding, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [-100] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # Padding
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [-100] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss
    
    def print_trainable_params(self):
        """Print trainable vs frozen parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable percentage: {trainable_params/total_params*100:.1f}%")


def create_model(args):
    """Create vision-compliant GraphCheck model."""
    return GraphCheck(args) 