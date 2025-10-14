"""
Usage:
    # Get sample data
    documents = create_sample_dataset()
"""

# Import main models package
from src.legal_graph import processors

# Package metadata
__version__ = "0.1.0"
__description__ = "LLM knowledge usage in graph neural network for highly accurate multi-task modular AI/ML system"
__authors__ = [
    "Illia Savenko <savenko.ilya@lll.kpi.ua>",
    "Mykyta Bobylyow <mbobylyov@biph.kiev.ua>"
]
__readme__ = "README.md"
__license__ = "MIT"

# Main exports
__all__ = [
    "processors",
    "__version__",
    "__author__", 
    "__description__",
    "__license__"
] 
