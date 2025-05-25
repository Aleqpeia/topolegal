#!/usr/bin/env python3
"""
Test script for legal code parser
This script tests the parser without requiring BigQuery connection
"""

import sys
import json
import logging
from pathlib import Path

# Add the processors directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from processors.code_processing.resources import LocalResourceConnector
from processors.code_processing.services.code_extraction_service import CodeExtractionService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockRepository:
    """Mock repository for testing without BigQuery"""
    
    def create_table(self):
        logger.info("Mock: Would create table")
    
    def insert_entries_batch(self, entries):
        logger.info(f"Mock: Would insert {len(entries)} entries")
    
    def get_statistics(self):
        return {"mock": True}


def test_parser():
    """Test the parser with a dry run"""
    
    logger.info("Starting parser test...")
    
    # Initialize resource connector
    resource_connector = LocalResourceConnector()
    
    # Use mock repository for testing
    repository = MockRepository()
    
    # Create extraction service (without actually using the repository)
    extraction_service = CodeExtractionService(
        resource_connector=resource_connector,
        repository=repository
    )
    
    # Test file
    test_file = "criminal_process_code.htm"
    
    try:
        # Parse the HTML
        logger.info(f"Parsing {test_file}...")
        json_data = resource_connector.fetch_and_parse(test_file)
        
        # Extract entries
        logger.info("Extracting legal code entries...")
        entries = extraction_service._extract_entries(json_data)
        
        logger.info(f"Successfully extracted {len(entries)} entries")
        
        # Show sample entries
        if entries:
            logger.info("\nSample entries:")
            for i, entry in enumerate(entries[:5]):
                logger.info(f"\nEntry {i+1}:")
                logger.info(f"  Article: {entry.article_number} - {entry.article_name}")
                logger.info(f"  Section: {entry.section} - {entry.section_name}")
                logger.info(f"  Heading: {entry.heading} - {entry.heading_name}")
                logger.info(f"  Paragraph: {entry.paragraph}")
                logger.info(f"  Code Entry: {entry.code_entry}")
                logger.info(f"  Text: {entry.code_entry_text[:100]}...")
                if entry.additional_info:
                    logger.info(f"  Additional Info: {entry.additional_info[:100]}...")
        
        # Calculate statistics
        stats = extraction_service._calculate_statistics(entries)
        logger.info(f"\nStatistics: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        
        # Save a sample to JSON
        sample_output = {
            'total_entries': len(entries),
            'sample_entries': [
                {
                    'code_entry': e.code_entry,
                    'code_entry_text': e.code_entry_text[:200] + '...' if len(e.code_entry_text) > 200 else e.code_entry_text,
                    'additional_info': e.additional_info,
                    'article_number': e.article_number,
                    'article_name': e.article_name,
                    'paragraph': e.paragraph,
                    'heading': e.heading,
                    'heading_name': e.heading_name,
                    'section': e.section,
                    'section_name': e.section_name,
                }
                for e in entries[:10]  # First 10 entries
            ]
        }
        
        with open('test_output_sample.json', 'w', encoding='utf-8') as f:
            json.dump(sample_output, f, ensure_ascii=False, indent=2)
        
        logger.info("\nSaved sample output to test_output_sample.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_parser()
    if success:
        logger.info("\nTest completed successfully!")
    else:
        logger.error("\nTest failed!")
        exit(1) 