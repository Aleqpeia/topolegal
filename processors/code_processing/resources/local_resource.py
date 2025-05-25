from pathlib import Path
from typing import Dict, Any, List
from bs4 import BeautifulSoup, NavigableString, Tag
import json

from .base import ResourceConnector


class LocalResourceConnector(ResourceConnector):
    """Local resource connector for fetching HTML files from filesystem"""
    
    def __init__(self, base_path: Path = None):
        """
        Initialize local resource connector
        
        Args:
            base_path: Base path for local resources. Defaults to codes/ directory
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent / "codes"
        self.base_path = Path(base_path)
    
    def fetch(self, resource_path: str) -> str:
        """
        Fetch HTML content from local file
        
        Args:
            resource_path: Relative path to the HTML file
            
        Returns:
            HTML content as string
        """
        full_path = self.base_path / resource_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Resource not found: {full_path}")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _parse_element(self, element) -> Dict[str, Any]:
        """
        Recursively parse an HTML element to dictionary
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Dictionary representation of the element
        """
        if isinstance(element, NavigableString):
            # Handle text nodes
            text = str(element).strip()
            if text:
                return {
                    "type": "text",
                    "content": text
                }
            return None
        
        if not isinstance(element, Tag):
            return None
        
        # Parse tag element
        result = {
            "tag": element.name,
            "classes": element.get("class", []),
            "id": element.get("id"),
            "attributes": dict(element.attrs),
            "content": element.get_text(strip=True),
            "children": []
        }
        
        # Parse children
        for child in element.children:
            parsed_child = self._parse_element(child)
            if parsed_child:
                result["children"].append(parsed_child)
        
        return result
    
    def parse_to_json(self, content: str) -> Dict[str, Any]:
        """
        Parse HTML content to JSON structure
        
        Args:
            content: HTML content as string
            
        Returns:
            JSON representation of the HTML structure
        """
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find the main article div
        article_div = soup.find('div', id='article')
        if not article_div:
            # If no article div, parse the whole body
            article_div = soup.body if soup.body else soup
        
        parsed_structure = self._parse_element(article_div)
        
        return {
            "structure": parsed_structure,
            "raw_html": str(article_div),
            "encoding": soup.original_encoding
        } 