#!/usr/bin/env python3
"""
Vision-Compliant GraphCheck Models CLI

This module provides command-line interface access to the vision-compliant
graph-based legal reference validation system.

Usage:
    python -m models --help
    python -m models demo
    python -m models bigquery-demo --table your-project.dataset.table
    python -m models train --config config.json
    python -m models test --model model.pt --data data.json
    python -m models info
"""

import argparse
import sys
import json
import os
from pathlib import Path

# Add the models directory to the Python path
models_dir = Path(__file__).parent
sys.path.insert(0, str(models_dir.parent))

# Import models components
try:
    from models import (
        GraphCheck,
        create_model,
        create_sample_dataset,
        get_available_models,
        get_available_trainers,
        __version__,
        __description__,
        BigQueryLegalGraphDataset,
        demonstrate_bigquery_dataset,
        create_bigquery_dataset
    )
    from models.train_legal_validation import LegalValidationTrainer
    from models.example_legal_validation import create_sample_legal_data
except ImportError as e:
    print(f"❌ Error importing models: {e}")
    print("Make sure you're running from the repository root directory")
    sys.exit(1)


def show_info():
    """Show package information."""
    print(f"📦 Vision-Compliant GraphCheck v{__version__}")
    print(f"📝 {__description__}")
    print()
    print("🏗️ Architecture: INPUT → SYNTHETIC → GNN → PROJECTOR → FUSION → OUTPUT")
    print("🔒 Frozen: Transformer (red blocks)")
    print("🔄 Trainable: NER, Synthetic, GNN, Projector, Fusion (teal blocks)")
    print()
    print("🇺🇦 Supported Ukrainian Legal Codes:")
    print("   ⚖️  КК України (Criminal Code)")
    print("   🏛️  КПК України (Criminal Procedure Code)")
    print("   📜 ЦК України (Civil Code)")
    print("   🚔 КоАП України (Administrative Code)")
    print("   👨‍👩‍👧‍👦 СК України (Family Code)")
    print("   💼 КЗпП України (Labor Code)")
    print()
    print(f"🤖 Available Models: {', '.join(get_available_models())}")
    print(f"👨‍🏫 Available Trainers: {', '.join(get_available_trainers())}")
    print()
    print("📊 Data Sources:")
    print("   📄 Sample data (built-in Ukrainian legal documents)")
    print("   🔥 BigQuery integration (production legal document datasets)")


def run_demo():
    """Run a demonstration of the system."""
    print("🚀 Running Vision-Compliant GraphCheck Demo")
    print("=" * 50)
    
    # Create sample data
    print("📄 Creating sample Ukrainian legal documents...")
    documents = create_sample_dataset()
    print(f"✅ Created {len(documents)} sample documents")
    
    # Show sample document
    print("\n📋 Sample Document:")
    sample_doc = documents[0]
    print(f"ID: {sample_doc['id']}")
    print(f"Text: {sample_doc['text'][:100]}...")
    print(f"Label: {sample_doc['label']}")
    print(f"References: {sample_doc.get('legal_references', [])}")
    print(f"Entities: {len(sample_doc['knowledge_graph']['entities'])}")
    print(f"Triplets: {len(sample_doc['knowledge_graph']['triplets'])}")
    
    # Show architecture info
    print("\n🏗️ Architecture Components:")
    print("🔒 FROZEN (Red blocks):")
    print("   - Transformer Model (BERT/T5/RoBERTa)")
    print("   - Word Embeddings")
    print()
    print("🔄 TRAINABLE (Teal blocks):")
    print("   - NER Model (Entity extraction)")
    print("   - Synthetic Data Processor")
    print("   - Graph Encoder (GNN)")
    print("   - Projector (GNN → Frozen space)")
    print("   - Fusion Layer (Combine outputs)")
    
    print("\n📊 Data Flow:")
    print("INPUT → NER → SYNTHETIC → GNN → PROJECTOR → FUSION → OUTPUT")
    print("         ↓        ↑")
    print("    FROZEN_BERT → FROZEN_EMB")
    
    print("\n✅ Demo completed! Use 'python -m models train' to start training.")


def run_bigquery_demo(table_id):
    """Run BigQuery dataset demonstration."""
    print("🔥 BigQuery Legal Graph Dataset Demo")
    print("=" * 50)
    
    try:
        print(f"📊 Connecting to BigQuery table: {table_id}")
        demonstrate_bigquery_dataset(table_id)
        
        print("\n📈 BigQuery Integration Benefits:")
        print("   📊 Large-scale legal document datasets")
        print("   🔄 Real-time data updates")
        print("   📈 Scalable training pipelines")
        print("   🔍 Advanced querying capabilities")
        print("   🇺🇦 Ukrainian legal corpus processing")
        
        print(f"\n💡 Next Steps:")
        print("   1. 📔 Use in notebook: models/notebooks/comprehensive_training_notebook.ipynb")
        print("   2. 🏋️ Set USE_BIGQUERY = True in notebook")
        print(f"   3. 📝 Update BIGQUERY_TABLE_ID = '{table_id}'")
        print("   4. 🚀 Run full training pipeline")
        
    except Exception as e:
        print(f"❌ BigQuery demo failed: {e}")
        print("💡 Common issues:")
        print("   - BigQuery credentials not configured")
        print("   - Table doesn't exist or no permissions")
        print("   - Missing google-cloud-bigquery package")
        print("   - Network connectivity issues")
        print("\n🔧 Setup BigQuery:")
        print("   pip install google-cloud-bigquery")
        print("   gcloud auth application-default login")


def train_model(config_path=None):
    """Train a model with optional configuration."""
    print("🏋️ Training Vision-Compliant GraphCheck Model")
    print("=" * 50)
    
    if config_path and os.path.exists(config_path):
        print(f"📄 Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print("⚙️ Using default configuration")
        # Default configuration
        class DefaultConfig:
            llm_model_path = "microsoft/DialoGPT-medium"
            ner_model_name = "bert-base-uncased"
            num_legal_labels = 8
            gnn_in_dim = 768
            gnn_hidden_dim = 256
            gnn_num_layers = 3
            gnn_dropout = 0.1
            gnn_num_heads = 4
            max_txt_len = 512
            max_new_tokens = 128
            learning_rate = 2e-5
            batch_size = 2
            num_epochs = 5
        
        config = DefaultConfig()
    
    print("🏗️ Creating model...")
    try:
        from types import SimpleNamespace
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        
        model = create_model(config)
        print("✅ Model created successfully!")
        
        # Create sample data for training
        documents = create_sample_dataset()
        print(f"📊 Using {len(documents)} sample documents for training")
        
        print("\n🚀 Training would start here...")
        print("💡 Use the Jupyter notebook for full training: models/notebooks/comprehensive_training_notebook.ipynb")
        print("💡 For BigQuery data: python -m models bigquery-demo --table your-table-id")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        print("💡 Make sure you have all dependencies installed:")
        print("   pip install torch torch-geometric transformers sklearn matplotlib seaborn tqdm")


def test_model(model_path, data_path=None):
    """Test a trained model."""
    print("🧪 Testing Vision-Compliant GraphCheck Model")
    print("=" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"📥 Loading model from: {model_path}")
    print("🧪 Testing functionality would be implemented here...")
    print("💡 Use the Jupyter notebook for full testing: models/notebooks/comprehensive_training_notebook.ipynb")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vision-Compliant GraphCheck Models CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m models info          # Show package information
    python -m models demo          # Run demonstration with sample data
    python -m models bigquery-demo --table project.dataset.table  # BigQuery demo
    python -m models train         # Train with default config
    python -m models train --config config.json    # Train with custom config
    python -m models test --model model.pt         # Test trained model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show package information')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration with sample data')
    
    # BigQuery demo command
    bigquery_parser = subparsers.add_parser('bigquery-demo', help='Run BigQuery dataset demonstration')
    bigquery_parser.add_argument('--table', type=str, required=True, help='BigQuery table ID (project.dataset.table)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a trained model')
    test_parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
    test_parser.add_argument('--data', type=str, help='Path to test data JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'info':
        show_info()
    elif args.command == 'demo':
        run_demo()
    elif args.command == 'bigquery-demo':
        run_bigquery_demo(args.table)
    elif args.command == 'train':
        train_model(args.config)
    elif args.command == 'test':
        test_model(args.model, args.data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 