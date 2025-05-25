import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup, Tag

from ..resources.base import ResourceConnector
from ..repositories.base import LegalCodeRepository
from ..models import LegalCodeEntry


logger = logging.getLogger(__name__)


class CodeExtractionService:
    """Service for extracting legal code entries from HTML documents"""
    
    # CSS class patterns for different elements
    SECTION_CLASSES = ['rvts15', 'rvts23']  # Bold, larger text for sections
    HEADING_CLASSES = ['rvts15', 'rvts23']  # Bold text for headings
    ARTICLE_CLASSES = ['rvts9']  # Bold text for articles
    ITALIC_CLASSES = ['rvts11', 'rvts46', 'rvts3', 'rvts5']  # Italic for additional info
    
    def __init__(self, resource_connector: ResourceConnector, repository: LegalCodeRepository):
        """
        Initialize code extraction service
        
        Args:
            resource_connector: Connector for fetching resources
            repository: Repository for storing extracted entries
        """
        self.resource_connector = resource_connector
        self.repository = repository
        self.spacy_pipeline = None  # Placeholder for spacy pipeline
        
    def set_spacy_pipeline(self, pipeline):
        """
        Set spacy pipeline for NLP processing
        
        Args:
            pipeline: Spacy pipeline instance
        """
        self.spacy_pipeline = pipeline
        logger.info("Spacy pipeline set")
    
    def extract_and_store(self, resource_path: str) -> Dict[str, Any]:
        """
        Extract legal code entries from HTML and store in repository
        
        Args:
            resource_path: Path to the HTML resource
            
        Returns:
            Dictionary with extraction statistics
        """
        # Fetch and parse HTML
        json_data = self.resource_connector.fetch_and_parse(resource_path)
        
        # Extract entries from parsed structure
        entries = self._extract_entries(json_data)
        
        # Store entries in repository
        if entries:
            self.repository.create_table()
            self.repository.insert_entries_batch(entries)
        
        return {
            "total_entries": len(entries),
            "resource_path": resource_path,
            "statistics": self._calculate_statistics(entries)
        }
    
    def _extract_entries(self, json_data: Dict[str, Any]) -> List[LegalCodeEntry]:
        """
        Extract legal code entries from parsed JSON structure
        
        Args:
            json_data: Parsed HTML structure as JSON
            
        Returns:
            List of legal code entries
        """
        entries = []
        
        # Parse the HTML structure
        structure = json_data.get("structure", {})
        if not structure:
            return entries
        
        # Convert back to BeautifulSoup for easier traversal
        html_content = json_data.get("raw_html", "")
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Current context
        current_section = None
        current_section_name = None
        current_heading = None
        current_heading_name = None
        current_article_number = None
        current_article_name = None
        
        # Find all paragraphs
        for element in soup.find_all(['p', 'div']):
            # Skip empty elements
            text = element.get_text(strip=True)
            if not text:
                continue
            
            # Skip amendments/changes in italic (starting with {)
            if text.startswith('{'):
                continue
            
            # Check if this is a section
            if self._is_section(element):
                section_info = self._extract_section_info(text)
                if section_info:
                    current_section = section_info[0]
                    current_section_name = section_info[1]
                    current_heading = None
                    current_heading_name = None
                    current_article_number = None
                    current_article_name = None
                    logger.debug(f"Found section: {current_section} - {current_section_name}")
                    continue
            
            # Check if this is a heading
            if self._is_heading(element):
                heading_info = self._extract_heading_info(text)
                if heading_info:
                    current_heading = heading_info[0]
                    current_heading_name = heading_info[1]
                    current_article_number = None
                    current_article_name = None
                    logger.debug(f"Found heading: {current_heading} - {current_heading_name}")
                    continue
            
            # Check if this is an article
            if self._is_article(element):
                article_info = self._extract_article_info(text)
                if article_info:
                    current_article_number = article_info[0]
                    current_article_name = article_info[1]
                    logger.debug(f"Found article: {current_article_number} - {current_article_name}")
                    continue
            
            # Check if this is a code entry (numbered item within an article)
            if current_article_number:
                code_entry_match = re.match(r'^(\d+(?:-\d+)?)\.\s+(.+)', text, re.DOTALL)
                if code_entry_match:
                    code_entry_num = code_entry_match.group(1)
                    entry_text = code_entry_match.group(2).strip()
                    
                    # Extract additional info from italic elements
                    additional_info = self._extract_additional_info(element)
                    
                    # Clean entry text from additional info
                    if additional_info:
                        for info in additional_info:
                            entry_text = entry_text.replace(info, '').strip()
                    
                    entry = LegalCodeEntry(
                        code_entry=code_entry_num,
                        code_entry_text=entry_text,
                        additional_info=' '.join(additional_info) if additional_info else None,
                        article_number=current_article_number,
                        article_name=current_article_name,
                        paragraph=None,  # Code entries don't have paragraphs
                        heading=current_heading,
                        heading_name=current_heading_name,
                        section=current_section,
                        section_name=current_section_name
                    )
                    entries.append(entry)
                    logger.debug(f"Added code entry {code_entry_num} for article {current_article_number}")
                # Also check for entries with parentheses format: 1) text
                else:
                    paren_match = re.match(r'^(\d+(?:-\d+)?)\)\s*(.+)', text, re.DOTALL)
                    if paren_match:
                        code_entry_num = paren_match.group(1)
                        entry_text = paren_match.group(2).strip()
                        
                        # Extract additional info from italic elements
                        additional_info = self._extract_additional_info(element)
                        
                        # Clean entry text from additional info
                        if additional_info:
                            for info in additional_info:
                                entry_text = entry_text.replace(info, '').strip()
                        
                        entry = LegalCodeEntry(
                            code_entry=code_entry_num,
                            code_entry_text=entry_text,
                            additional_info=' '.join(additional_info) if additional_info else None,
                            article_number=current_article_number,
                            article_name=current_article_name,
                            paragraph=None,
                            heading=current_heading,
                            heading_name=current_heading_name,
                            section=current_section,
                            section_name=current_section_name
                        )
                        entries.append(entry)
                        logger.debug(f"Added code entry {code_entry_num} for article {current_article_number}")
        
        return entries
    
    def _is_section(self, element: Tag) -> bool:
        """Check if element is a section header"""
        text = element.get_text(strip=True)
        
        # Check for section pattern (e.g., "Розділ I ЗАГАЛЬНІ ПОЛОЖЕННЯ")
        if re.match(r'Розділ\s+[IVX]+', text):
            return True
        
        return False
    
    def _is_heading(self, element: Tag) -> bool:
        """Check if element is a heading"""
        text = element.get_text(strip=True)
        
        # Check for heading pattern (e.g., "Глава 1. Кримінальне процесуальне...")
        if re.match(r'Глава\s+\d+\.?\s*', text):
            return True
        
        return False
    
    def _is_article(self, element: Tag) -> bool:
        """Check if element is an article header"""
        text = element.get_text(strip=True)
        
        # Check for article pattern
        if re.match(r'Стаття\s+\d+\.?\s*', text):
            return True
        
        # Also check for bold spans with article pattern
        bold_spans = element.find_all('span', class_=self.ARTICLE_CLASSES)
        for span in bold_spans:
            if re.match(r'Стаття\s+\d+', span.get_text(strip=True)):
                return True
        
        return False
    
    def _extract_section_info(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract section number and name"""
        # Pattern for "Розділ I ЗАГАЛЬНІ ПОЛОЖЕННЯ" or with line break
        match = re.match(r'Розділ\s+([IVX]+)\s*[\n\r]*(.+)', text, re.DOTALL)
        if match:
            section_num = match.group(1)
            section_name = match.group(2).strip()
            # Convert Roman to Arabic for storage
            section_num_arabic = self._roman_to_arabic(section_num)
            return (str(section_num_arabic), section_name)
        return None
    
    def _extract_heading_info(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract heading number and name"""
        match = re.match(r'Глава\s+(\d+)\.?\s*(.+)', text, re.DOTALL)
        if match:
            heading_num = match.group(1)
            heading_name = match.group(2).strip()
            return (heading_num, heading_name)
        return None
    
    def _extract_article_info(self, text: str) -> Optional[Tuple[int, str]]:
        """Extract article number and name"""
        match = re.match(r'Стаття\s+(\d+)\.?\s*(.+)', text, re.DOTALL)
        if match:
            article_num = int(match.group(1))
            article_name = match.group(2).strip()
            return (article_num, article_name)
        return None
    
    def _extract_additional_info(self, element: Tag) -> List[str]:
        """Extract additional info from italic elements"""
        additional_info = []
        italic_elements = element.find_all('span', class_=self.ITALIC_CLASSES)
        for italic in italic_elements:
            italic_text = italic.get_text(strip=True)
            if italic_text and italic_text.startswith('{'):
                additional_info.append(italic_text)
        return additional_info
    
    def _roman_to_arabic(self, roman: str) -> int:
        """Convert Roman numerals to Arabic numbers"""
        roman_numerals = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        result = 0
        prev_value = 0
        
        for char in reversed(roman):
            value = roman_numerals.get(char, 0)
            if value < prev_value:
                result -= value
            else:
                result += value
            prev_value = value
        
        return result
    
    def _calculate_statistics(self, entries: List[LegalCodeEntry]) -> Dict[str, int]:
        """Calculate statistics about extracted entries"""
        stats = {
            'total': len(entries),
            'by_section': {},
            'by_article': {},
            'with_additional_info': 0
        }
        
        for entry in entries:
            # Count by section
            if entry.section:
                stats['by_section'][entry.section] = stats['by_section'].get(entry.section, 0) + 1
            
            # Count by article
            if entry.article_number:
                stats['by_article'][entry.article_number] = stats['by_article'].get(entry.article_number, 0) + 1
            
            # Count entries with additional info
            if entry.additional_info:
                stats['with_additional_info'] += 1
        
        return stats 