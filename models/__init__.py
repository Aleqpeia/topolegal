"""
Vision-Compliant GraphCheck Models Package

This package contains the implementation of the vision-compliant graph-based legal reference
validation system, including all necessary components for training, testing, and inference.

Key Components:
- VisionCompliantGraphCheck: Main model implementing the diagram architecture
- EntityExtractor: Trainable NER model for legal entity extraction
- SyntheticDataProcessor: Converts entities to graph nodes/JSON structure
- GraphEncoder: Trainable GNN for legal knowledge graphs
- Projector: Maps GNN output to frozen embedding space
- AttentionFusion: Combines GNN and frozen transformer outputs
- ReferenceValidationDataset: PyTorch Geometric dataset for graph data
- Various trainer classes for different model configurations
"""

# Vision-compliant implementation (main architecture)
from .vision_compliant_graphcheck import (
    VisionCompliantGraphCheck,
    EntityExtractor,
    SyntheticDataProcessor,
    GraphEncoder,
    Projector,
    AttentionFusion
)

# Dataset and data handling
from .graph_dataset import (
    ReferenceValidationDataset,
    create_graph_dataset,
    load_json_documents
)

# Original GraphCheck implementation
from .graphcheck import (
    GraphCheck,
    create_legal_graphcheck_model
)

# Training utilities
from .trainer import GraphCheckTrainer
from .train_validation import ValidationTrainer
from .train_legal_validation import LegalValidationTrainer

# Available classifier components
from .classifier import (
    LegalDocumentClassifier,
    LegalEntityExtractor
)

# Example and demo functions
from .example_legal_validation import create_sample_legal_data

# Version information
__version__ = "1.0.0"
__author__ = "Vision-Compliant GraphCheck Team"
__description__ = "Graph-based legal reference validation with frozen transformers"

# Main exports for easy importing
__all__ = [
    # Main vision-compliant model
    "VisionCompliantGraphCheck",
    "EntityExtractor", 
    "SyntheticDataProcessor",
    "GraphEncoder",
    "Projector",
    "AttentionFusion",
    
    # Dataset handling
    "ReferenceValidationDataset",
    "create_graph_dataset",
    "load_json_documents",
    
    # Original implementations
    "GraphCheck",
    "create_legal_graphcheck_model",
    
    # Training
    "GraphCheckTrainer",
    "ValidationTrainer", 
    "LegalValidationTrainer",
    
    # Utilities
    "LegalDocumentClassifier",
    "LegalEntityExtractor",
    
    # Examples and demos
    "create_sample_legal_data",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

# Module-level convenience functions
def get_available_models():
    """Get list of available model classes."""
    return [
        "VisionCompliantGraphCheck",
        "GraphCheck", 
        "LegalDocumentClassifier"
    ]

def get_available_trainers():
    """Get list of available trainer classes."""
    return [
        "GraphCheckTrainer",
        "ValidationTrainer",
        "LegalValidationTrainer"
    ]

def create_vision_compliant_model(config_or_args):
    """
    Convenience function to create a vision-compliant model.
    
    Args:
        config_or_args: Configuration object or args with model parameters
        
    Returns:
        VisionCompliantGraphCheck: Initialized model
    """
    return VisionCompliantGraphCheck(config_or_args)

def create_sample_dataset():
    """
    Create a sample dataset for testing and demonstration.
    
    Returns:
        list: Sample Ukrainian legal documents with knowledge graphs
    """
    return create_sample_legal_data()

# Print package information when imported
def _print_package_info():
    """Print package information on import."""
    print(f"📦 Vision-Compliant GraphCheck v{__version__}")
    print("🏗️ Architecture: INPUT → SYNTHETIC → GNN → PROJECTOR → FUSION → OUTPUT")
    print("🔒 Frozen: Transformer | 🔄 Trainable: NER, Synthetic, GNN, Projector, Fusion")
    print("🇺🇦 Ukrainian Legal Codes: КК України, КПК України, ЦК України, КоАП України")

# Uncomment the line below to show info on import
# _print_package_info() 