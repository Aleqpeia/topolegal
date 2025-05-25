from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class GranularityLevel(Enum):
    """Enumeration for different levels of legal code structure"""
    SECTION = "section"          # Розділ
    HEADING = "heading"          # Глава
    ARTICLE = "article"          # Стаття
    PARAGRAPH = "paragraph"      # Частина
    CODE_ENTRY = "code_entry"    # Пункт


@dataclass
class LegalCodeEntry:
    """Model for a legal code entry (smallest granularity)"""
    code_entry: Optional[str] = None           # Пункт (e.g., "1", "2-1")
    code_entry_text: str = ""                  # Text content of the entry
    additional_info: Optional[str] = None      # Italic text (notes, amendments)
    article_number: Optional[int] = None       # Article number
    article_name: Optional[str] = None         # Article name
    paragraph: Optional[str] = None            # Частина (typically None for code entries)
    paragraph_name: Optional[str] = None       # Paragraph name
    heading: Optional[str] = None              # Глава (stored as Arabic numeral string)
    heading_name: Optional[str] = None         # Heading name
    section: Optional[str] = None              # Розділ (stored as Arabic numeral string)
    section_name: Optional[str] = None         # Section name
    

@dataclass
class ParsedElement:
    """Represents a parsed HTML element with its attributes"""
    tag: str
    classes: List[str]
    id: Optional[str]
    content: str
    attributes: dict
    children: List['ParsedElement']
    

@dataclass
class HtmlParseResult:
    """Result of HTML parsing"""
    elements: List[ParsedElement]
    raw_json: dict 