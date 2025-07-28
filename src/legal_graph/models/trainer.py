import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import logging
import argparse

from .legal_graphcheck import create_legal_graphcheck_model


class LegalKGDataset(Dataset):
    """
    Dataset for legal document classification with GraphCheck.
    """
    def __init__(self, documents: List[Dict], max_length: int = 512):
        self.documents = documents
        self.max_length = max_length
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        return {
            'id': doc.get('id', idx),
            'text': doc.get('text', ''),
            'label': doc.get('label', 'valid'),
            'legal_references': doc.get('legal_references', [])
        }


class GraphCheckTrainer:
    """
    Trainer for the GraphCheck model.
    """
    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Optimizer - only trainable parameters
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Prepare batch data
            batch_data = {
                'id': [item['id'] for item in batch],
                'text': [item['text'] for item in batch],
                'label': [item['label'] for item in batch],
                'legal_references': [item.get('legal_references', []) for item in batch]
            }
            
            # Forward pass
            loss = self.model(batch_data)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions (simplified for now)
            batch_predictions = ['valid'] * len(batch)  # Placeholder
            batch_labels = [item['label'] for item in batch]
            
            all_predictions.extend(batch_predictions)
            all_labels.extend(batch_labels)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy_score(batch_labels, batch_predictions):.4f}"
            })
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Prepare batch data
                batch_data = {
                    'id': [item['id'] for item in batch],
                    'text': [item['text'] for item in batch],
                    'label': [item['label'] for item in batch],
                    'legal_references': [item.get('legal_references', []) for item in batch]
                }
                
                # Forward pass
                loss = self.model(batch_data)
                total_loss += loss.item()
                
                # Get predictions
                result = self.model.inference(batch_data)
                batch_predictions = result['pred']
                batch_labels = [item['label'] for item in batch]
                
                all_predictions.extend(batch_predictions)
                all_labels.extend(batch_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 10,
        save_path: str = "legal_graphcheck_model.pt",
        early_stopping_patience: int = 5
    ):
        """
        Train the model with early stopping.
        """
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Log metrics
            self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_f1': best_val_f1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, save_path)
                
                self.logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                self.logger.info(f"Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break
        
        self.logger.info(f"Training completed. Best validation F1: {best_val_f1:.4f}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Model loaded from {model_path}")
        self.logger.info(f"Best validation F1: {checkpoint['best_val_f1']:.4f}")


def prepare_legal_data(documents: List[Dict], test_size: float = 0.2, batch_size: int = 8):
    """
    Prepare data loaders for legal document training.
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    train_docs, val_docs, train_labels, val_labels = train_test_split(
        documents, [doc.get('label', 'valid') for doc in documents], 
        test_size=test_size, random_state=42, stratify=[doc.get('label', 'valid') for doc in documents]
    )
    
    # Add labels back to documents
    for doc, label in zip(train_docs, train_labels):
        doc['label'] = label
    for doc, label in zip(val_docs, val_labels):
        doc['label'] = label
    
    # Create datasets
    train_dataset = LegalKGDataset(train_docs)
    val_dataset = LegalKGDataset(val_docs)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader


def create_sample_legal_data() -> List[Dict]:
    """
    Create sample legal document data for testing.
    """
    sample_documents = [
        {
            'id': 1,
            'text': 'Приморський районний суд м. Одеси визнав ОСОБА_4 винним у крадіжці згідно з ч.2 ст.185 КК України.',
            'label': 'valid',
            'legal_references': ['ч.2 ст.185 КК України', 'ст.65 КК України']
        },
        {
            'id': 2,
            'text': 'Суддя ОСОБА_1 ухвалив ухвалу про клопотання слідчого ОСОБА_3.',
            'label': 'valid',
            'legal_references': ['ст. 219 КПК України', 'ч.1 ст. 294 КПК України']
        },
        {
            'id': 3,
            'text': 'Invalid legal document without proper references.',
            'label': 'invalid',
            'legal_references': ['invalid reference']
        }
    ]
    
    return sample_documents


def parse_args():
    """
    Parse command line arguments for training.
    """
    parser = argparse.ArgumentParser(description="GraphCheck Training")
    
    # Model arguments
    parser.add_argument("--llm_model_path", type=str, default="microsoft/DialoGPT-medium", 
                       help="Path to the LLM model")
    parser.add_argument("--ner_model_name", type=str, default="bert-base-uncased",
                       help="NER model name")
    parser.add_argument("--num_legal_labels", type=int, default=8,
                       help="Number of legal entity labels")
    
    # GNN arguments
    parser.add_argument("--gnn_in_dim", type=int, default=768,
                       help="GNN input dimension")
    parser.add_argument("--gnn_hidden_dim", type=int, default=256,
                       help="GNN hidden dimension")
    parser.add_argument("--gnn_num_layers", type=int, default=3,
                       help="Number of GNN layers")
    parser.add_argument("--gnn_dropout", type=float, default=0.1,
                       help="GNN dropout rate")
    parser.add_argument("--gnn_num_heads", type=int, default=4,
                       help="Number of attention heads")
    
    # Training arguments
    parser.add_argument("--max_txt_len", type=int, default=512,
                       help="Maximum text length")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum new tokens for generation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--save_path", type=str, default="legal_graphcheck_model.pt",
                       help="Model save path")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                       help="Early stopping patience")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create sample data
    documents = create_sample_legal_data()
    print(f"Created dataset with {len(documents)} documents")
    
    # Create model
    model = create_legal_graphcheck_model(args)
    
    # Print model information
    model.print_trainable_params()
    
    # Create trainer
    trainer = GraphCheckTrainer(
        model, 
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Prepare data
    train_dataloader, val_dataloader = prepare_legal_data(
        documents, 
        batch_size=args.batch_size
    )
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    
    # Train model
    print("\nStarting training...")
    trainer.train(
        train_dataloader, 
        val_dataloader, 
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        early_stopping_patience=args.early_stopping_patience
    )
    
    print("\nTraining completed!") 