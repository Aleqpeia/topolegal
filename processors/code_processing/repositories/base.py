from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..models import LegalCodeEntry


class LegalCodeRepository(ABC):
    """Abstract base class for legal code repository"""
    
    @abstractmethod
    def create_table(self) -> None:
        """Create the table/schema if it doesn't exist"""
        pass
    
    @abstractmethod
    def insert_entry(self, entry: LegalCodeEntry) -> None:
        """
        Insert a single legal code entry
        
        Args:
            entry: LegalCodeEntry object to insert
        """
        pass
    
    @abstractmethod
    def insert_entries_batch(self, entries: List[LegalCodeEntry]) -> None:
        """
        Insert multiple legal code entries in batch
        
        Args:
            entries: List of LegalCodeEntry objects to insert
        """
        pass
    
    @abstractmethod
    def get_entry_by_article(self, article_number: int) -> List[LegalCodeEntry]:
        """
        Get all entries for a specific article
        
        Args:
            article_number: Article number to query
            
        Returns:
            List of legal code entries for the article
        """
        pass
    
    @abstractmethod
    def get_entries_by_section(self, section: str) -> List[LegalCodeEntry]:
        """
        Get all entries for a specific section
        
        Args:
            section: Section identifier
            
        Returns:
            List of legal code entries for the section
        """
        pass
    
    @abstractmethod
    def search_entries(self, search_text: str) -> List[LegalCodeEntry]:
        """
        Search entries by text content
        
        Args:
            search_text: Text to search for
            
        Returns:
            List of matching legal code entries
        """
        pass
    
    @abstractmethod
    def delete_all_entries(self) -> None:
        """Delete all entries from the repository"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the stored entries
        
        Returns:
            Dictionary with statistics (count by type, total count, etc.)
        """
        pass 