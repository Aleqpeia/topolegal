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
    try:
        from google.cloud import bigquery
    except ImportError:
        logger.error("BigQuery not available. Install: pip install google-cloud-bigquery")
        return

    # Honour CLI / .env overrides
    if key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    if project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project

    client = bigquery.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location="us-central1")

    total = 0
    while True:
        # Pull the next batch of unprocessed rows
        if update_existing:
            # Update existing records that don't have triplets or have 0 triplets
            sql = f"""
                SELECT  doc_id,
                        text,
                        tags,
                        triplets_count
                FROM    `{table_id}`
                WHERE   (triplets_count IS NULL OR triplets_count = 0)
                    AND text IS NOT NULL
                    AND tags IS NOT NULL
                    AND LENGTH(text) > 10
                LIMIT   {batch}
            """
        else:
            # Process all records that don't have triplets
            sql = f"""
                SELECT  doc_id,
                        text,
                        tags,
                        triplets_count
                FROM    `{table_id}`
                WHERE   triplets_count IS NULL
                    AND text IS NOT NULL
                    AND tags IS NOT NULL
                    AND LENGTH(text) > 10
                LIMIT   {batch}
            """

        job = client.query(sql)
        logger.info("Batch job %s submitted", job.job_id)
        rows = job.result().to_dataframe()

        if rows.empty:
            logger.info("No more documents to process")
            break

        for _, row in rows.iterrows():
            try:
                doc_id = str(row.doc_id)
                text = str(row.text)

                # Parse entities from tags column
                entities = []
                if hasattr(row, 'tags') and pd.notna(row.tags):
                    try:
                        entities = json.loads(str(row.tags))
                    except:
                        logger.warning(f"Could not parse entities for doc {doc_id}")

                # Initialize extractor if not already done
                if 'extractor' not in locals() or not globals().get('extractor'):
                    extractor = LegalGraphExtractor() if LegalGraphExtractor else None
                
                # Process document - use await directly since we're in an async function
                if process_document and extractor:
                    result = await process_document(text, entities, doc_id, extractor)
                else:
                    logger.error("Graph extraction not available")
                    continue

                # Prepare the update data
                triplets_json = json.dumps(result.get('knowledge_graph', {}).get('triplets', []), ensure_ascii=False)
                # Escape the JSON for BigQuery
                triplets_json_escaped = triplets_json.replace("'", "\\'").replace('"', '\\"')
                triplets_count = result.get('triplets_count', 0)

                # Update the existing table with triplet results
                update_sql = f"""
                    UPDATE `{table_id}`
                    SET triplets = JSON '{triplets_json_escaped}',
                        triplets_count = @triplets_count
                    WHERE CAST(doc_id AS STRING) = @doc_id
                """

                # Execute the update
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("triplets_count", "INTEGER", triplets_count),
                        bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id),
                    ]
                )

                update_job = client.query(update_sql, job_config=job_config)
                update_job.result()  # Wait for the update to complete

                total += 1
                logger.info(f"Processed doc {doc_id}: {triplets_count} triplets")

            except Exception as e:
                logger.error(f"BQ doc {row.doc_id}: {e}")

        logger.info("Processed %d rows in this batch", total)


def check_bigquery_schema(table_id: str):
    """Check if the BigQuery table has the required columns"""
    try:
        from google.cloud import bigquery

        client = bigquery.Client()
        table = client.get_table(table_id)
        columns = [field.name for field in table.schema]

        required_columns = ['doc_id', 'text', 'tags']
        optional_columns = ['triplets', 'triplets_count', 'processing_timestamp']

        missing_required = [col for col in required_columns if col not in columns]
        existing_optional = [col for col in optional_columns if col in columns]

        logger.info(f"Table schema check for {table_id}:")
        logger.info(f"Required columns: {required_columns}")
        logger.info(f"Optional columns: {optional_columns}")
        logger.info(f"Missing required: {missing_required}")
        logger.info(f"Existing optional: {existing_optional}")

        # Log the actual schema details
        logger.info("Actual table schema:")
        for field in table.schema:
            logger.info(f"  {field.name}: {field.field_type}")

        if missing_required:
            logger.error(f"Missing required columns: {missing_required}")
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to check table schema: {e}")
        return False


def get_bigquery_stats(table_id: str):
    """Get statistics about triplet processing status"""
    try:
        from google.cloud import bigquery

        client = bigquery.Client()

        sql = f"""
        SELECT 
            COUNT(*) as total_docs,
            COUNTIF(triplets_count IS NULL) as docs_without_triplets,
            COUNTIF(triplets_count = 0) as docs_with_zero_triplets,
            COUNTIF(triplets_count > 0) as docs_with_triplets,
            AVG(triplets_count) as avg_triplets,
            MAX(triplets_count) as max_triplets
        FROM `{table_id}`
        WHERE text IS NOT NULL
        """

        job = client.query(sql)
        results = job.result().to_dataframe()

        if not results.empty:
            row = results.iloc[0]
            logger.info(f"""
            Processing Statistics:
            ====================
            Total documents: {row.total_docs}
            Documents without triplets: {row.docs_without_triplets}
            Documents with zero triplets: {row.docs_with_zero_triplets}
            Documents with triplets: {row.docs_with_triplets}
            Average triplets per doc: {row.avg_triplets:.2f}
            Maximum triplets in a doc: {row.max_triplets}
            """)

    except Exception as e:
        logger.error(f"Failed to get processing stats: {e}")


async def corrupt_bigquery_data(table_id: str, output_table_id: str, corruption_probability: float = 0.5,
                                project: Optional[str] = None, key_path: Optional[str] = None, 
                                batch_size: int = 100):
    """Create a corrupted dataset from BigQuery table for binary classification training"""
    try:
        from google.cloud import bigquery
    except ImportError:
        logger.error("BigQuery not available. Install: pip install google-cloud-bigquery")
        return

    # Honour CLI / .env overrides
    if key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    if project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project

    client = bigquery.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location="us-central1")
    
    # Initialize corruptor
    corruptor = LegalReferenceCorruptor(corruption_probability=corruption_probability)
    
    logger.info(f"Starting corruption process: {table_id} -> {output_table_id}")
    logger.info(f"Corruption probability: {corruption_probability}")
    
    # Create output table if it doesn't exist
    create_output_table(client, output_table_id)
    
    processed = 0
    while True:
        # Fetch batch of data
        sql = f"""
            SELECT  doc_id,
                    text,
                    tags,
                    triplets,
                    triplets_count
            FROM    `{table_id}`
            WHERE   triplets_count IS NOT NULL
                AND triplets_count > 0
                AND text IS NOT NULL
                AND tags IS NOT NULL
                AND LENGTH(text) > 10
            LIMIT   {batch_size}
            OFFSET  {processed}
        """
        
        job = client.query(sql)
        rows = job.result().to_dataframe()
        
        if rows.empty:
            logger.info("No more documents to process")
            break
        
        # Convert rows to dataset format
        dataset = []
        for _, row in rows.iterrows():
            try:
                entities = json.loads(str(row.tags)) if pd.notna(row.tags) else []
                triplets = json.loads(str(row.triplets)) if pd.notna(row.triplets) else []
                
                dataset.append({
                    'doc_id': str(row.doc_id),
                    'text': str(row.text),
                    'entities': entities,
                    'triplets': triplets,
                    'triplets_count': int(row.triplets_count)
                })
            except Exception as e:
                logger.warning(f"Error parsing row {row.doc_id}: {e}")
                continue
        
        # Apply corruption
        corrupted_dataset = corruptor.corrupt_dataset(dataset, label_column='is_valid')
        
        # Insert into output table
        insert_corrupted_data(client, output_table_id, corrupted_dataset)
        
        processed += len(rows)
        logger.info(f"Processed {processed} documents")
        
        # Show corruption stats
        stats = corruptor.get_corruption_stats()
        logger.info(f"Corruption stats: {stats}")


def create_output_table(client, table_id: str):
    """Create output table for corrupted data"""
    try:
        from google.cloud import bigquery
        
        schema = [
            bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("entities", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("triplets", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("triplets_count", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("is_valid", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("processing_timestamp", "TIMESTAMP", mode="NULLABLE")
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table = client.create_table(table, exists_ok=True)
        logger.info(f"Created table {table_id}")
        
    except Exception as e:
        logger.error(f"Failed to create output table: {e}")


def insert_corrupted_data(client, table_id: str, data: List[Dict]):
    """Insert corrupted data into BigQuery table"""
    try:
        from google.cloud import bigquery
        
        # Prepare rows for insertion
        rows_to_insert = []
        for item in data:
            row = {
                'doc_id': item['doc_id'],
                'text': item['text'],
                'entities': json.dumps(item['entities'], ensure_ascii=False),
                'triplets': json.dumps(item['triplets'], ensure_ascii=False),
                'triplets_count': item['triplets_count'],
                'is_valid': item['is_valid'],
                'processing_timestamp': datetime.now().isoformat()
            }
            rows_to_insert.append(row)
        
        # Insert data
        errors = client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            logger.error(f"Errors inserting data: {errors}")
        else:
            logger.info(f"Inserted {len(rows_to_insert)} rows into {table_id}")
            
    except Exception as e:
        logger.error(f"Failed to insert data: {e}")


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