import asyncio
import json
import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging
from pathlib import Path
import argparse
from datetime import datetime
import os
import sys

# Import corruption functionality
from . import LegalReferenceCorruptor, DataCorruptor

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import processing functions
try:
    from ..graph_extraction import LegalGraphExtractor, process_legal_document
    process_document = process_legal_document  # Alias for compatibility
except ImportError:
    logger.warning("Graph extraction module not available. Some features may be limited.")
    LegalGraphExtractor = None
    process_document = None



async def run_bigquery(table_id: str, batch: int, project: Optional[str] = None,
                       key_path: Optional[str] = None, update_existing: bool = False):
    """Process documents from BigQuery table and update with triplet extraction results"""
    from ..bigquery_utils import BigQueryClient, parse_json_column

    # Initialize client
    client = BigQueryClient(project=project, key_path=key_path)

    # Initialize extractor
    if not LegalGraphExtractor or not process_document:
        logger.error("Graph extraction not available")
        return

    extractor = LegalGraphExtractor()

    total = 0
    processed = 0

    # Determine WHERE clause based on update mode
    if update_existing:
        where_clause = "(triplets_count IS NULL OR triplets_count = 0) AND text IS NOT NULL AND tags IS NOT NULL AND LENGTH(text) > 10"
    else:
        where_clause = "triplets_count IS NULL AND text IS NOT NULL AND tags IS NOT NULL AND LENGTH(text) > 10"

    while True:
        # Fetch batch
        sql = f"""
            SELECT doc_id, text, tags, triplets_count
            FROM `{table_id}`
            WHERE {where_clause}
            LIMIT {batch}
        """

        rows_df = client.query_to_dataframe(sql)

        if rows_df.empty:
            logger.info("No more documents to process")
            break

        for _, row in rows_df.iterrows():
            try:
                doc_id = str(row.doc_id)
                text = str(row.text)

                # Parse entities
                entities = parse_json_column(row.get('tags')) or []

                # Process document
                result = await process_document(text, entities, doc_id, extractor)

                # Prepare update data
                triplets = result.get('knowledge_graph', {}).get('triplets', [])
                triplets_json = json.dumps(triplets, ensure_ascii=False)
                triplets_json_escaped = triplets_json.replace("'", "\\'").replace('"', '\\"')
                triplets_count = result.get('triplets_count', 0)

                # Update table
                update_sql = f"""
                    UPDATE `{table_id}`
                    SET triplets = JSON '{triplets_json_escaped}',
                        triplets_count = @triplets_count
                    WHERE CAST(doc_id AS STRING) = @doc_id
                """

                job_config = client.bigquery.QueryJobConfig(
                    query_parameters=[
                        client.bigquery.ScalarQueryParameter("triplets_count", "INTEGER", triplets_count),
                        client.bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id),
                    ]
                )

                update_job = client.client.query(update_sql, job_config=job_config)
                update_job.result()

                total += 1
                logger.info(f"Processed doc {doc_id}: {triplets_count} triplets")

            except Exception as e:
                logger.error(f"BQ doc {row.doc_id}: {e}")

        processed += len(rows_df)
        logger.info(f"Processed {processed} rows in this batch, {total} successful")


def check_bigquery_schema(table_id: str):
    """Check if the BigQuery table has the required columns"""
    from ..bigquery_utils import BigQueryClient, check_legal_document_schema

    try:
        client = BigQueryClient()
        is_valid, missing = check_legal_document_schema(
            client=client,
            table_id=table_id,
            require_entities=True,
            require_triplets=False
        )

        # Get full schema for logging
        schema = client.get_table_schema(table_id)

        logger.info(f"Table schema check for {table_id}:")
        logger.info("Actual table schema:")
        for col, col_type in schema.items():
            logger.info(f"  {col}: {col_type}")

        if not is_valid:
            logger.error(f"Missing required columns: {missing}")
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to check table schema: {e}")
        return False


def get_bigquery_stats(table_id: str):
    """Get statistics about triplet processing status"""
    from ..bigquery_utils import BigQueryClient, get_triplet_statistics, format_table_stats

    try:
        client = BigQueryClient()
        stats = get_triplet_statistics(client, table_id)

        if stats:
            logger.info(format_table_stats(stats))
        else:
            logger.warning("No statistics available")

    except Exception as e:
        logger.error(f"Failed to get processing stats: {e}")


async def corrupt_bigquery_data(table_id: str, output_table_id: str, corruption_probability: float = 0.5,
                                project: Optional[str] = None, key_path: Optional[str] = None,
                                batch_size: int = 100):
    """Create a corrupted dataset from BigQuery table for binary classification training"""
    from ..bigquery_utils import (
        BigQueryClient, BigQueryBatchProcessor,
        create_legal_document_table, parse_json_column
    )

    # Initialize client
    client = BigQueryClient(project=project, key_path=key_path)

    # Initialize corruptor
    corruptor = LegalReferenceCorruptor(corruption_probability=corruption_probability)

    logger.info(f"Starting corruption process: {table_id} -> {output_table_id}")
    logger.info(f"Corruption probability: {corruption_probability}")

    # Create output table
    create_legal_document_table(
        client=client,
        table_id=output_table_id,
        include_entities=True,
        include_triplets=True,
        include_labels=True,
        label_column='is_valid'
    )

    def process_batch(batch_data):
        """Process batch and apply corruption"""
        dataset = []
        for row in batch_data:
            entities = parse_json_column(row.get('tags'))
            triplets = parse_json_column(row.get('triplets'))

            if not triplets:
                continue

            dataset.append({
                'doc_id': str(row.get('doc_id')),
                'text': str(row.get('text')),
                'entities': entities or [],
                'triplets': triplets,
                'triplets_count': int(row.get('triplets_count', 0))
            })

        # Apply corruption
        corrupted_dataset = corruptor.corrupt_dataset(dataset, label_column='is_valid')

        # Format for insertion
        rows_to_insert = []
        for item in corrupted_dataset:
            rows_to_insert.append({
                'doc_id': item['doc_id'],
                'text': item['text'],
                'entities': item['entities'],
                'triplets': item['triplets'],
                'triplets_count': item['triplets_count'],
                'is_valid': item['is_valid'],
                'processing_timestamp': datetime.now().isoformat()
            })

        return rows_to_insert

    # Process table using batch processor
    processor = BigQueryBatchProcessor(client, batch_size=batch_size)
    stats = processor.process_table(
        table_id=table_id,
        processor_func=process_batch,
        output_table_id=output_table_id,
        where_clause="triplets_count IS NOT NULL AND triplets_count > 0 AND text IS NOT NULL AND tags IS NOT NULL AND LENGTH(text) > 10"
    )

    # Show final statistics
    corruption_stats = corruptor.get_corruption_stats()
    logger.info(f"Batch processing complete: {stats}")
    logger.info(f"Corruption stats: {corruption_stats}")




async def corrupt_csv_data(input_csv: str, output_csv: str, corruption_probability: float = 0.5):
    """Create a corrupted dataset from CSV file for binary classification training"""
    try:
        # Load data
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} documents from {input_csv}")
        
        # Convert to dataset format
        dataset = []
        for _, row in df.iterrows():
            try:
                entities = json.loads(row.tags) if pd.notna(row.tags) else []
                triplets = json.loads(row.triplets) if pd.notna(row.triplets) else []
                
                dataset.append({
                    'doc_id': str(row.get('doc_id', f"doc_{len(dataset)}")),
                    'text': str(row.text),
                    'entities': entities,
                    'triplets': triplets,
                    'triplets_count': len(triplets)
                })
            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue
        
        # Initialize corruptor and apply corruption
        corruptor = LegalReferenceCorruptor(corruption_probability=corruption_probability)
        corrupted_dataset = corruptor.corrupt_dataset(dataset, label_column='is_valid')
        
        # Convert back to DataFrame and save
        output_data = []
        for item in corrupted_dataset:
            output_data.append({
                'doc_id': item['doc_id'],
                'text': item['text'],
                'tags': json.dumps(item['entities'], ensure_ascii=False),
                'triplets': json.dumps(item['triplets'], ensure_ascii=False),
                'triplets_count': item['triplets_count'],
                'is_valid': item['is_valid']
            })
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_csv, index=False)
        
        # Show statistics
        stats = corruptor.get_corruption_stats()
        logger.info(f"Corruption complete! Saved to {output_csv}")
        logger.info(f"Total samples: {len(output_df)}")
        logger.info(f"Valid samples: {len(output_df[output_df['is_valid']])}")
        logger.info(f"Corrupted samples: {len(output_df[~output_df['is_valid']])}")
        logger.info(f"Corruption stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Legal Data Balance and Corruption Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # BigQuery processing command
    bigquery_parser = subparsers.add_parser('bigquery', help='Process BigQuery data')
    bigquery_parser.add_argument('table_id', help='BigQuery table ID (project.dataset.table)')
    bigquery_parser.add_argument('--batch', type=int, default=100, help='Batch size for processing')
    bigquery_parser.add_argument('--project', help='Google Cloud Project ID')
    bigquery_parser.add_argument('--key-path', help='Path to service account key file')
    bigquery_parser.add_argument('--update-existing', action='store_true', help='Update existing records')
    
    # BigQuery corruption command
    corrupt_bq_parser = subparsers.add_parser('corrupt-bigquery', help='Create corrupted dataset from BigQuery')
    corrupt_bq_parser.add_argument('input_table', help='Input BigQuery table ID')
    corrupt_bq_parser.add_argument('output_table', help='Output BigQuery table ID')
    corrupt_bq_parser.add_argument('--probability', type=float, default=0.5, help='Corruption probability (0.0-1.0)')
    corrupt_bq_parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    corrupt_bq_parser.add_argument('--project', help='Google Cloud Project ID')
    corrupt_bq_parser.add_argument('--key-path', help='Path to service account key file')
    
    # CSV corruption command
    corrupt_csv_parser = subparsers.add_parser('corrupt-csv', help='Create corrupted dataset from CSV')
    corrupt_csv_parser.add_argument('input_csv', help='Input CSV file path')
    corrupt_csv_parser.add_argument('output_csv', help='Output CSV file path')
    corrupt_csv_parser.add_argument('--probability', type=float, default=0.5, help='Corruption probability (0.0-1.0)')
    
    # Schema check command
    schema_parser = subparsers.add_parser('check-schema', help='Check BigQuery table schema')
    schema_parser.add_argument('table_id', help='BigQuery table ID')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Get BigQuery table statistics')
    stats_parser.add_argument('table_id', help='BigQuery table ID')
    
    args = parser.parse_args()
    
    if args.command == 'bigquery':
        asyncio.run(run_bigquery(
            table_id=args.table_id,
            batch=args.batch,
            project=args.project,
            key_path=args.key_path,
            update_existing=args.update_existing
        ))
    
    elif args.command == 'corrupt-bigquery':
        asyncio.run(corrupt_bigquery_data(
            table_id=args.input_table,
            output_table_id=args.output_table,
            corruption_probability=args.probability,
            project=args.project,
            key_path=args.key_path,
            batch_size=args.batch_size
        ))
    
    elif args.command == 'corrupt-csv':
        asyncio.run(corrupt_csv_data(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            corruption_probability=args.probability
        ))
    
    elif args.command == 'check-schema':
        check_bigquery_schema(args.table_id)
    
    elif args.command == 'stats':
        get_bigquery_stats(args.table_id)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()