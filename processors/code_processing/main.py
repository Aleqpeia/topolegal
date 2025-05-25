#!/usr/bin/env python3
"""
Main entry point for legal code parser

Usage:
    python main.py --resource criminal_process_code.htm
    python main.py --resource criminal_process_code.htm --use-spacy
"""

import sys
import argparse
import logging
import json
from pathlib import Path

# Add the project root to Python path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from processors.code_processing.config import Config
from processors.code_processing.resources import LocalResourceConnector
from processors.code_processing.repositories import BigQueryLegalCodeRepository
from processors.code_processing.services import CodeExtractionService


# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to parse legal code and store in BigQuery"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parse legal code HTML documents')
    parser.add_argument(
        '--resource',
        type=str,
        default='criminal_process_code.htm',
        help='HTML file to parse (relative to codes directory)'
    )
    parser.add_argument(
        '--use-spacy',
        action='store_true',
        help='Use spacy pipeline for NLP processing'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Extract entries but do not store in BigQuery'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Output extracted entries to JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Resource connector
        resource_connector = LocalResourceConnector(Config.CODES_PATH)
        
        # Repository
        repository = BigQueryLegalCodeRepository(
            project_id=Config.BIGQUERY_PROJECT_ID,
            dataset_id=Config.BIGQUERY_DATASET_ID,
            table_id=Config.BIGQUERY_TABLE_ID
        )
        
        # Extraction service
        extraction_service = CodeExtractionService(
            resource_connector=resource_connector,
            repository=repository
        )
        
        # Set up spacy pipeline if requested
        if args.use_spacy:
            try:
                import spacy
                nlp = spacy.load(Config.SPACY_MODEL)
                extraction_service.set_spacy_pipeline(nlp)
                logger.info(f"Loaded spacy model: {Config.SPACY_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to load spacy model: {e}")
        
        # Extract entries
        logger.info(f"Processing resource: {args.resource}")
        
        if args.dry_run:
            # Dry run - just extract without storing
            json_data = resource_connector.fetch_and_parse(args.resource)
            entries = extraction_service._extract_entries(json_data)
            
            logger.info(f"Extracted {len(entries)} entries (dry run)")
            
            # Output to JSON if requested
            if args.output_json:
                output_data = {
                    'resource': args.resource,
                    'total_entries': len(entries),
                    'entries': [
                        {
                            'code_entry': e.code_entry,
                            'code_entry_text': e.code_entry_text,
                            'additional_info': e.additional_info,
                            'article_number': e.article_number,
                            'article_name': e.article_name,
                            'paragraph': e.paragraph,
                            'paragraph_name': e.paragraph_name,
                            'heading': e.heading,
                            'heading_name': e.heading_name,
                            'section': e.section,
                            'section_name': e.section_name,
                        }
                        for e in entries
                    ]
                }
                
                with open(args.output_json, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved extracted entries to: {args.output_json}")
            
            # Print statistics
            stats = extraction_service._calculate_statistics(entries)
            logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
            
        else:
            # Normal run - extract and store
            result = extraction_service.extract_and_store(args.resource)
            
            logger.info(f"Successfully processed {result['total_entries']} entries")
            logger.info(f"Statistics: {json.dumps(result['statistics'], indent=2)}")
            
            # Get repository statistics
            repo_stats = repository.get_statistics()
            logger.info(f"Repository statistics: {json.dumps(repo_stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


def example_usage():
    """Example of how to use the parser programmatically"""
    
    # Initialize components
    resource_connector = LocalResourceConnector()
    repository = BigQueryLegalCodeRepository(
        project_id="your-project",
        dataset_id="legal_codes",
        table_id="criminal_code"
    )
    extraction_service = CodeExtractionService(
        resource_connector=resource_connector,
        repository=repository
    )
    
    # Extract and store
    result = extraction_service.extract_and_store("criminal_process_code.htm")
    print(f"Processed {result['total_entries']} entries")
    
    # Query examples
    # Get all entries for article 42
    article_entries = repository.get_entry_by_article(42)
    print(f"Article 42 has {len(article_entries)} entries")
    
    # Search for specific text
    search_results = repository.search_entries("слідчий")
    print(f"Found {len(search_results)} entries containing 'слідчий'")
    
    # Get entries by section
    section_entries = repository.get_entries_by_section("I")
    print(f"Section I has {len(section_entries)} entries")


if __name__ == "__main__":
    exit(main()) 