#!/usr/bin/env python3
"""
Main entry point for CSV custom parser

Usage:
    python -m processors.csv_custom_parser
    python -m processors.csv_custom_parser --csv-file my_data.csv
    python -m processors.csv_custom_parser --csv-file data.csv --table-name custom_table
    python -m processors.csv_custom_parser --dry-run --verbose
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the project root to Python path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from processors.csv_custom_parser.parser import GenericCsvParser


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to parse CSV and ingest to BigQuery"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parse any CSV file and ingest to BigQuery with auto-generated schema')
    parser.add_argument(
        '--csv-file',
        type=str,
        help='CSV file to parse (relative to resources directory). If not specified, lists available CSV files.'
    )
    parser.add_argument(
        '--table-name',
        type=str,
        help='Override BigQuery table name (defaults to CSV filename without extension)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of records to insert per batch'
    )
    parser.add_argument(
        '--delimiter',
        type=str,
        default='\t',
        help='CSV delimiter character (default: tab). Use "auto" for auto-detection.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse CSV but do not ingest to BigQuery'
    )
    parser.add_argument(
        '--project-id',
        type=str,
        help='GCP project ID (falls back to GOOGLE_CLOUD_PROJECT env var)'
    )
    parser.add_argument(
        '--dataset-id',
        type=str,
        default='court_data_2022',
        help='BigQuery dataset ID'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Build path to resources directory
        resources_dir = Path(__file__).parent.parent / 'resources'
        
        # If no CSV file specified, list available files
        if not args.csv_file:
            logger.info("No CSV file specified. Available CSV files in resources directory:")
            csv_files = list(resources_dir.glob("*.csv"))
            if csv_files:
                for csv_file in sorted(csv_files):
                    logger.info("  - %s", csv_file.name)
                logger.info("\nUsage: python -m processors.csv_custom_parser --csv-file <filename>")
            else:
                logger.warning("No CSV files found in resources directory: %s", resources_dir)
            sys.exit(0)
        
        # Build full path to CSV file
        csv_path = resources_dir / args.csv_file
        
        if not csv_path.exists():
            logger.error("CSV file not found: %s", csv_path)
            # Show available files
            csv_files = list(resources_dir.glob("*.csv"))
            if csv_files:
                logger.info("Available CSV files:")
                for csv_file in sorted(csv_files):
                    logger.info("  - %s", csv_file.name)
            sys.exit(1)
        
        logger.info("Processing CSV file: %s", csv_path)
        logger.info("File size: %.2f MB", csv_path.stat().st_size / (1024 * 1024))
        
        if args.dry_run:
            logger.info("DRY RUN MODE: Will not ingest to BigQuery")
            
            # Initialize parser (might fail if no credentials, but that's ok for dry run)
            try:
                csv_parser = GenericCsvParser(
                    project_id=args.project_id,
                    dataset_id=args.dataset_id,
                    csv_delimiter=args.delimiter
                )
            except Exception as e:
                logger.warning("Could not initialize BigQuery client (dry run): %s", e)
                # Create a mock parser for dry run
                csv_parser = None
            
            # Parse the CSV
            if csv_parser:
                records = csv_parser.parse_csv_file(csv_path)
                table_name = args.table_name or csv_parser._get_table_name_from_filename(csv_path)
            else:
                # Parse manually for dry run
                import csv
                from processors.csv_custom_parser.models import CsvRecord
                
                records = []
                delimiter = '\t' if args.delimiter == '\t' else args.delimiter
                with open(csv_path, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file, delimiter=delimiter)
                    for row in reader:
                        try:
                            record = CsvRecord.from_csv_row(row)
                            if record.data:
                                records.append(record)
                        except Exception as e:
                            logger.error("Error parsing row: %s", e)
                            continue
                
                # Generate table name manually
                import re
                table_name = args.table_name or re.sub(r'[^\w\d]', '_', csv_path.stem).lower()
            
            logger.info("DRY RUN: Would process %d records", len(records))
            logger.info("Target table would be: %s", table_name)
            
            # Show sample data
            if records:
                logger.info("Detected columns: %s", list(records[0].data.keys()))
                logger.info("Sample records:")
                for i, record in enumerate(records[:3]):
                    logger.info("  Record %d: %s", i + 1, 
                               {k: (str(v)[:50] + "..." if len(str(v)) > 50 else v) 
                                for k, v in record.data.items()})
                if len(records) > 3:
                    logger.info("  ... and %d more records", len(records) - 3)
                
                # Show inferred schema
                from processors.csv_custom_parser.models import SchemaInference
                schema_info = SchemaInference.infer_schema(records)
                logger.info("Inferred BigQuery schema:")
                for col in schema_info:
                    logger.info("  - %s: %s (%s)", col["name"], col["type"], col["mode"])
            
        else:
            # Initialize parser
            logger.info("Initializing CSV parser...")
            csv_parser = GenericCsvParser(
                project_id=args.project_id,
                dataset_id=args.dataset_id,
                csv_delimiter=args.delimiter
            )
            
            # Process CSV file
            result = csv_parser.process_csv_file(csv_path, args.table_name, args.batch_size)
            
            # Report results
            logger.info("Processing complete!")
            logger.info("Table created/updated: %s", result['table_ref'])
            logger.info("Total processed: %d", result['total_processed'])
            logger.info("Total errors: %d", result['total_errors'])
            
            if result['total_errors'] > 0:
                logger.warning("Some records failed to process. Check logs for details.")
                sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Processing failed: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 