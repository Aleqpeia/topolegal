#!/usr/bin/env python3
"""
Test script to verify the extracted structure is correct
"""

import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from processors.code_processing.resources import LocalResourceConnector
from processors.code_processing.services.code_extraction_service import CodeExtractionService


class MockRepository:
    def create_table(self): pass
    def insert_entries_batch(self, entries): pass
    def get_statistics(self): return {}


def test_specific_structure():
    """Test if the parser extracts the correct structure for Article 1"""
    
    # Initialize components
    resource_connector = LocalResourceConnector()
    repository = MockRepository()
    extraction_service = CodeExtractionService(resource_connector, repository)
    
    # Parse and extract
    json_data = resource_connector.fetch_and_parse("criminal_process_code.htm")
    entries = extraction_service._extract_entries(json_data)
    
    # Find Article 1 entries
    article_1_entries = [e for e in entries if e.article_number == 1]
    
    print("=== TESTING ARTICLE 1 STRUCTURE ===")
    print(f"Found {len(article_1_entries)} entries for Article 1")
    print()
    
    expected_structure = {
        "section": "1",
        "section_name": "ЗАГАЛЬНІ ПОЛОЖЕННЯ", 
        "heading": "1",
        "heading_name": "Кримінальне процесуальне законодавство України та сфера його дії",
        "article_number": 1,
        "article_name": "Кримінальне процесуальне законодавство України"
    }
    
    for i, entry in enumerate(article_1_entries):
        print(f"Entry {i+1}:")
        print(f"  code_entry: {entry.code_entry}")
        print(f"  article_number: {entry.article_number}")
        print(f"  article_name: {entry.article_name}")
        print(f"  section: {entry.section}")
        print(f"  section_name: {entry.section_name}")
        print(f"  heading: {entry.heading}")
        print(f"  heading_name: {entry.heading_name}")
        print(f"  paragraph: {entry.paragraph}")
        print(f"  text: {entry.code_entry_text[:100]}...")
        print()
        
        # Verify structure
        errors = []
        if entry.section != expected_structure["section"]:
            errors.append(f"Section mismatch: got '{entry.section}', expected '{expected_structure['section']}'")
        if entry.section_name != expected_structure["section_name"]:
            errors.append(f"Section name mismatch: got '{entry.section_name}', expected '{expected_structure['section_name']}'")
        if entry.heading != expected_structure["heading"]:
            errors.append(f"Heading mismatch: got '{entry.heading}', expected '{expected_structure['heading']}'")
        if entry.article_number != expected_structure["article_number"]:
            errors.append(f"Article number mismatch: got '{entry.article_number}', expected '{expected_structure['article_number']}'")
        if entry.paragraph is not None:
            errors.append(f"Paragraph should be None for code entries, got '{entry.paragraph}'")
            
        if errors:
            print(f"  ❌ ERRORS: {'; '.join(errors)}")
        else:
            print(f"  ✅ Structure correct")
        print()
    
    # Test expected text content
    expected_texts = [
        "Порядок кримінального провадження на території України визначається лише кримінальним процесуальним законодавством України.",
        "Кримінальне процесуальне законодавство України складається з відповідних положень",
        "Зміни до кримінального процесуального законодавства України можуть вноситися виключно законами"
    ]
    
    print("=== TESTING TEXT CONTENT ===")
    for i, (entry, expected_start) in enumerate(zip(article_1_entries, expected_texts)):
        if entry.code_entry_text.startswith(expected_start):
            print(f"✅ Entry {i+1} text starts correctly")
        else:
            print(f"❌ Entry {i+1} text mismatch:")
            print(f"   Expected start: {expected_start}")
            print(f"   Actual start: {entry.code_entry_text[:len(expected_start)]}")
    
    return len(article_1_entries) == 3


if __name__ == "__main__":
    success = test_specific_structure()
    if success:
        print("\n🎉 All tests passed! The parser is working correctly.")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1) 