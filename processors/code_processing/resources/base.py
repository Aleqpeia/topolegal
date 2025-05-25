from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path


class ResourceConnector(ABC):
    """Abstract base class for resource connectors"""
    
    @abstractmethod
    def fetch(self, resource_path: str) -> str:
        """
        Fetch the resource content
        
        Args:
            resource_path: Path to the resource (local path or URL)
            
        Returns:
            Raw content of the resource
        """
        pass
    
    @abstractmethod
    def parse_to_json(self, content: str) -> Dict[str, Any]:
        """
        Parse HTML content to JSON structure
        
        Args:
            content: HTML content as string
            
        Returns:
            JSON representation of the HTML structure including:
            - tag names
            - classes
            - ids
            - contents
            - attributes
        """
        pass
    
    def fetch_and_parse(self, resource_path: str) -> Dict[str, Any]:
        """
        Convenience method to fetch and parse in one step
        
        Args:
            resource_path: Path to the resource
            
        Returns:
            JSON representation of the HTML structure
        """
        content = self.fetch(resource_path)
        return self.parse_to_json(content) 