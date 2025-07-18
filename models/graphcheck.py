import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification
from torch_scatter import scatter
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool
from torch_geometric.data import Data
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import re


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
IGNORE_INDEX = -100


class EntityExtractor(nn.Module):
    """
    Trainable NER model specialized for legal document entities.
    """
    def __init__(self, model_name: str = "bert-base-uncased", num_legal_labels: int = 8):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=num_legal_labels
        )
        self.hidden_size = self.ner_model.config.hidden_size
        
        # Legal entity labels mapping
        self.legal_labels = {
            0: "O",  # Outside
            1: "ORG",  # Organization
            2: "LOC",  # Location
            3: "ROLE",  # Role
            4: "PER",  # Person
            5: "INFO",  # Information
            6: "CRIME",  # Crime
            7: "DTYPE",  # Document Type
            8: "NUM"  # Number
        }
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.ner_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits, outputs.hidden_states[-1]
    
    def extract_legal_entities(self, text: str) -> List[Dict]:
        """Extract legal entities from text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        logits, hidden_states = self.forward(inputs["input_ids"], inputs["attention_mask"])
        
        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Extract entities with proper NER decoding
        entities = []
        current_entity = None
        
        for i, pred in enumerate(predictions[0]):
            if pred != 0:  # Not 'O'
                label = self.legal_labels.get(pred.item(), "UNK")
                token_text = self.tokenizer.decode(inputs['input_ids'][0][i])
                
                if current_entity is None:
                    current_entity = {
                        'start': i,
                        'label': label,
                        'text': token_text,
                        'confidence': torch.softmax(logits[0][i], dim=0)[pred].item()
                    }
                else:
                    current_entity['text'] += ' ' + token_text
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities


class GraphEncoder(nn.Module):
    """
    Trainable GNN for encoding legal knowledge graphs.
    """
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
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

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


class ReferenceValidator(nn.Module):
    """
    Module to validate legal references based on Ukrainian legal code patterns.
    """
    def __init__(self):
        super().__init__()
        
        # Ukrainian legal code patterns
        self.legal_patterns = {
            'КК України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*КК\s*України',
            'КПК України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*КПК\s*України',
            'ЦК України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*ЦК\s*України',
            'КоАП України': r'ст\.\s*\d+(\s*ч\.\s*\d+)?\s*КоАП\s*України'
        }
        
    def validate_reference(self, reference: str) -> Dict[str, any]:
        """Validate a legal reference"""
        if not reference:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'matched_pattern': None,
                'error': 'Empty reference'
            }
        
        # Check against patterns
        for code_name, pattern in self.legal_patterns.items():
            if re.search(pattern, reference, re.IGNORECASE):
                return {
                    'is_valid': True,
                    'confidence': 0.9,
                    'matched_pattern': code_name,
                    'error': None
                }
        
        # Check for common legal reference patterns
        if re.search(r'ст\.\s*\d+', reference):
            return {
                'is_valid': True,
                'confidence': 0.7,
                'matched_pattern': 'Generic article reference',
                'error': None
            }
        
        return {
            'is_valid': False,
            'confidence': 0.1,
            'matched_pattern': None,
            'error': 'No valid legal reference pattern found'
        }


class GraphCheck(nn.Module):
    """
    Adapted GraphCheck for legal document classification.
    Combines frozen transformer, trainable NER, trainable GNN, and legal reference validation.
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
        print('Finished loading frozen LLM!')

        self.word_embedding = self.model.model.get_input_embeddings()

        # Trainable NER model for legal entities
        self.ner_model = EntityExtractor(
            model_name=args.ner_model_name,
            num_legal_labels=args.num_legal_labels
        ).to(self.model.device)
        
        # Trainable GNN for legal graph encoding
        self.graph_encoder = GraphEncoder(
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)
        
        # Legal reference validator
        self.reference_validator = ReferenceValidator()
        
        # Projector to map GNN output to LLM embedding space
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.word_embedding.weight.shape[1]),
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
    
    def build_legal_graph(self, text: str, legal_references: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Build legal knowledge graph from text using NER and frozen transformer embeddings.
        """
        # Extract legal entities using trainable NER model
        entities = self.ner_model.extract_legal_entities(text)
        
        # Get embeddings for entities using frozen transformer
        entity_embeddings = []
        valid_entities = []
        
        for entity in entities:
            if entity['confidence'] > 0.5:  # Filter by confidence
                # Use frozen transformer to get embeddings
                inputs = self.tokenizer(entity['text'], return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    embedding = self.model.model.embed_tokens(inputs["input_ids"].to(self.device))
                    entity_emb = torch.mean(embedding, dim=1).squeeze(0)
                entity_embeddings.append(entity_emb)
                valid_entities.append(entity)
        
        if not entity_embeddings:
            # If no entities found, use the entire text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                embedding = self.model.model.embed_tokens(inputs["input_ids"].to(self.device))
                entity_embeddings = [torch.mean(embedding, dim=1).squeeze(0)]
                valid_entities = [{'text': text, 'label': 'O', 'confidence': 1.0}]
        
        # Stack embeddings
        node_features = torch.stack(entity_embeddings)
        
        # Create edges (fully connected graph for legal relationship prediction)
        num_nodes = len(valid_entities)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
        
        return node_features, edge_index, valid_entities
    
    def encode_legal_graphs(self, data):
        """
        Encode legal knowledge graphs using trainable GNN.
        """
        # Build graphs from text
        legal_graphs = []
        all_entities = []
        
        for text in data['text']:
            legal_references = data.get('legal_references', [])
            node_features, edge_index, entities = self.build_legal_graph(text, legal_references)
            
            # Create PyTorch Geometric Data object
            if edge_index.size(1) > 0:
                graph_data = Data(
                    x=node_features,
                    edge_index=edge_index,
                    num_nodes=node_features.size(0)
                ).to(self.device)
            else:
                # Empty graph
                graph_data = Data(
                    x=node_features,
                    edge_index=torch.empty((2, 0), dtype=torch.long).to(self.device),
                    num_nodes=node_features.size(0)
                ).to(self.device)
            
            legal_graphs.append(graph_data)
            all_entities.append(entities)
        
        # Encode graphs using trainable GNN
        graph_embeddings = []
        for graph in legal_graphs:
            if graph.edge_index.size(1) > 0:
                node_embeds, _ = self.graph_encoder(graph.x, graph.edge_index)
                # Global pooling
                graph_embed = global_mean_pool(node_embeds, torch.zeros(node_embeds.size(0), dtype=torch.long).to(self.device))
            else:
                graph_embed = torch.mean(graph.x, dim=0, keepdim=True)
            graph_embeddings.append(graph_embed)
        
        return torch.stack(graph_embeddings), all_entities

    def forward(self, data):
        """
        Forward pass for training.
        """
        # Encode legal graphs
        graph_embeds, all_entities = self.encode_legal_graphs(data)
        
        # Project graph embeddings to LLM embedding space
        graph_embeds = self.projector(graph_embeds)
        
        # Prepare text inputs
        texts = self.tokenizer(data["text"], add_special_tokens=False)
        labels = self.tokenizer(data["label"], add_special_tokens=False)

        # Encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        batch_size = len(data['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids   
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            
            # Add graph embeddings
            graph_embedding = graph_embeds[i].unsqueeze(0)
            inputs_embeds = torch.cat([bos_embeds, graph_embedding, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # Padding
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

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

    def inference(self, data):
        """
        Inference pass for prediction.
        """
        # Encode legal graphs
        graph_embeds, all_entities = self.encode_legal_graphs(data)
        
        # Project graph embeddings
        graph_embeds = self.projector(graph_embeds)
        
        # Encode prompt
        texts = self.tokenizer(data["text"], add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)
        
        batch_size = len(data['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        
        for i in range(batch_size):
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            
            # Add graph embeddings
            graph_embedding = graph_embeds[i].unsqueeze(0)
            inputs_embeds = torch.cat([bos_embeds, graph_embedding, inputs_embeds], dim=0)
            
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # Padding
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
            )
        
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return {
            'id': data['id'],
            'pred': pred,
            'label': data['label'],
            'text': data['text'],
            'entities': all_entities
        }

    def predict_legal_document(self, text: str, legal_references: List[str]) -> Dict[str, any]:
        """
        Make prediction for a single legal document.
        """
        self.eval()
        with torch.no_grad():
            # Create data structure
            data = {
                'id': [0],
                'text': [text],
                'label': ['valid'],  # Dummy label for inference
                'legal_references': [legal_references] if legal_references else []
            }
            
            # Run inference
            result = self.inference(data)
            
            # Validate legal references
            reference_validations = []
            if legal_references:
                for ref in legal_references:
                    validation = self.reference_validator.validate_reference(ref)
                    reference_validations.append(validation)
            
            # Calculate validity score
            validity_score = 0.0
            if reference_validations:
                valid_refs = sum(1 for ref in reference_validations if ref['is_valid'])
                validity_score = valid_refs / len(reference_validations)
            
            return {
                'prediction': result['pred'][0],
                'entities': result['entities'][0],
                'reference_validations': reference_validations,
                'validity_score': validity_score,
                'is_valid_document': validity_score > 0.5
            }

    def print_trainable_params(self):
        """Print trainable parameters information."""
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(f"Trainable parameters: {trainable_params:,}")
        print(f"All parameters: {all_param:,}")
        print(f"Trainable percentage: {trainable_params/all_param*100:.1f}%")
        
        return trainable_params, all_param


def create_legal_graphcheck_model(args):
    """
    Factory function to create LegalGraphCheck model.
    """
    return LegalGraphCheck(args) 