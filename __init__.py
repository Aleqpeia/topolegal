"""
TopoLegal: Vision-Compliant Graph-Based Legal Reference Validation

A comprehensive system for validating legal references in Ukrainian documents using
graph neural networks and frozen transformer models.

This package implements a vision-compliant architecture that follows the exact
data flow: INPUT → SYNTHETIC → GNN → PROJECTOR → FUSION → OUTPUT

Key Features:
- 🔒 Frozen transformer components (BERT/T5/RoBERTa)
- 🔄 Trainable graph neural networks
- 🇺🇦 Ukrainian legal code validation
- 📊 PyTorch Geometric graph processing
- 📈 Comprehensive training and evaluation tools

Usage:
    from topolegal.models import VisionCompliantGraphCheck, create_vision_compliant_model
    from topolegal.models import create_sample_dataset
    
    # Create model
    model = create_vision_compliant_model(config)
    
    # Get sample data
    documents = create_sample_dataset()
"""

# Import main models package
from . import models

# Package metadata
__version__ = "1.0.0"
__author__ = "TopoLegal Team"
__description__ = "Vision-compliant graph-based legal reference validation system"
__license__ = "MIT"

# Main exports
__all__ = [
    "models",
    "__version__",
    "__author__", 
    "__description__",
    "__license__"
] 