"""
Generic CSV parser with BigQuery ingestion
"""

import csv
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from google.cloud import bigquery
from google.auth import default
from google.oauth2 import service_account

from .models import CsvRecord, SchemaInference


logger = logging.getLogger(__name__)


class GenericCsvParser:
    """Generic parser for CSV files with BigQuery integration"""
    
    def __init__(self, 
                 project_id: Optional[str] = None,
                 dataset_id: str = "court_data_2022",
                 credentials_path: Optional[str] = None,
                 csv_delimiter: str = '\t'):
        """
        Initialize the parser with BigQuery configuration
        
        Args:
            project_id: GCP project ID (falls back to GOOGLE_CLOUD_PROJECT env var)
            dataset_id: BigQuery dataset ID
            credentials_path: Path to service account JSON (falls back to GOOGLE_APPLICATION_CREDENTIALS)
            csv_delimiter: CSV delimiter character (default: tab)
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.dataset_id = dataset_id
        self.csv_delimiter = csv_delimiter
        
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable or project_id parameter is required")
        
        # Initialize BigQuery client
        self.client = self._initialize_bigquery_client(credentials_path)
        
    def _initialize_bigquery_client(self, credentials_path: Optional[str] = None) -> bigquery.Client:
        """Initialize BigQuery client with proper credentials"""
        try:
            creds_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            if creds_path and os.path.exists(creds_path):
                # Load credentials from service account file
                credentials = service_account.Credentials.from_service_account_file(
                    creds_path,
                    scopes=['https://www.googleapis.com/auth/bigquery',
                           'https://www.googleapis.com/auth/cloud-platform']
                )
                logger.info("Using service account credentials from: %s", creds_path)
            else:
                # Fallback to default credentials
                credentials, _ = default(scopes=['https://www.googleapis.com/auth/bigquery',
                                               'https://www.googleapis.com/auth/cloud-platform'])
                logger.info("Using default credentials (ADC or metadata service)")
            
            client = bigquery.Client(project=self.project_id, credentials=credentials)
            logger.info("BigQuery client initialized for project: %s", self.project_id)
            return client
            
        except Exception as e:
            logger.error("Failed to initialize BigQuery client: %s", e)
            raise
    
    def _get_table_name_from_filename(self, csv_path: Path) -> str:
        """Extract table name from CSV filename"""
        # Remove extension and clean the name
        table_name = csv_path.stem
        
        # Clean table name for BigQuery compatibility
        # Replace special characters with underscores
        import re
        table_name = re.sub(r'[^\w\d]', '_', table_name).lower()
        
        # Ensure it starts with a letter or underscore
        if table_name and not table_name[0].isalpha() and table_name[0] != '_':
            table_name = f"table_{table_name}"
        
        # Ensure it's not empty
        if not table_name:
            table_name = "csv_data"
            
        return table_name
    
    def parse_csv_file(self, csv_path: Path) -> List[CsvRecord]:
        """
        Parse CSV file and return list of CsvRecord objects
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of CsvRecord objects
        """
        records = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                # Try to detect if file has headers by reading first few lines
                first_line = file.readline()
                file.seek(0)  # Reset to beginning
                
                # Use csv.Sniffer to detect delimiter if not specified
                if self.csv_delimiter == 'auto':
                    sample = file.read(1024)
                    file.seek(0)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    logger.info("Auto-detected delimiter: %r", delimiter)
                else:
                    delimiter = self.csv_delimiter
                
                reader = csv.DictReader(file, delimiter=delimiter)
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 since header is row 1
                    try:
                        if not any(row.values()):  # Skip empty rows
                            continue
                            
                        record = CsvRecord.from_csv_row(row)
                        if record.data:  # Only add if there's actual data
                            records.append(record)
                        
                    except (ValueError, TypeError) as e:
                        logger.error("Error parsing row %d: %s", row_num, e)
                        continue
                        
        except FileNotFoundError:
            logger.error("CSV file not found: %s", csv_path)
            raise
        except Exception as e:
            logger.error("Error reading CSV file: %s", e)
            raise
        
        logger.info("Successfully parsed %d records from CSV", len(records))
        return records
    
    def create_table_if_not_exists(self, table_name: str, records: List[CsvRecord]):
        """Create BigQuery table if it doesn't exist"""
        try:
            table_ref = f"{self.project_id}.{self.dataset_id}.{table_name}"
            
            # Infer schema from records
            schema_info = SchemaInference.infer_schema(records)
            
            if not schema_info:
                raise ValueError("Cannot create table: no schema could be inferred")
            
            # Convert to BigQuery schema
            schema = [
                bigquery.SchemaField(
                    col["name"], 
                    col["type"], 
                    mode=col["mode"]
                ) for col in schema_info
            ]
            
            table = bigquery.Table(table_ref, schema=schema)
            table.description = f"Auto-generated table from CSV file with {len(schema)} columns"
            
            # Create table if it doesn't exist
            table = self.client.create_table(table, exists_ok=True)
            logger.info("Table %s ready with schema: %s", 
                       table_ref, 
                       [f"{field.name}:{field.field_type}" for field in schema])
            
        except Exception as e:
            logger.error("Error creating table: %s", e)
            raise
    
    def ingest_to_bigquery(self, table_name: str, records: List[CsvRecord], 
                          batch_size: int = 1000) -> dict:
        """
        Ingest records to BigQuery
        
        Args:
            table_name: BigQuery table name
            records: List of CsvRecord objects to ingest
            batch_size: Number of records to insert per batch
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not records:
            logger.warning("No records to ingest")
            return {"total_processed": 0, "total_errors": 0}
        
        # Ensure table exists
        self.create_table_if_not_exists(table_name, records)
        
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_name}"
        total_processed = 0
        total_errors = 0
        
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            try:
                # Convert to dictionaries for BigQuery insertion
                rows_to_insert = [record.to_dict() for record in batch]
                
                # Generate row IDs for idempotent insert (use hash of data)
                row_ids = [str(hash(str(sorted(record.data.items())))) for record in batch]
                
                # Insert batch
                errors = self.client.insert_rows_json(
                    table_ref,
                    rows_to_insert,
                    row_ids=row_ids
                )
                
                if errors:
                    logger.error("BigQuery insert errors for batch %d-%d: %s", 
                               i, i + len(batch), errors)
                    total_errors += len(errors)
                else:
                    logger.info("Successfully inserted batch %d-%d (%d records)", 
                               i, i + len(batch), len(batch))
                    total_processed += len(batch)
                    
            except Exception as e:
                logger.error("Error inserting batch %d-%d: %s", i, i + len(batch), e)
                total_errors += len(batch)
        
        logger.info("Ingestion complete: %d processed, %d errors", 
                   total_processed, total_errors)
        
        return {
            "total_processed": total_processed,
            "total_errors": total_errors,
            "table_name": table_name,
            "table_ref": table_ref
        }
    
    def process_csv_file(self, csv_path: Path, table_name: Optional[str] = None, 
                        batch_size: int = 1000) -> dict:
        """
        Complete processing pipeline: parse CSV and ingest to BigQuery
        
        Args:
            csv_path: Path to the CSV file
            table_name: Override table name (defaults to filename without extension)
            batch_size: Number of records to insert per batch
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting CSV processing pipeline for: %s", csv_path)
        
        # Determine table name
        if table_name is None:
            table_name = self._get_table_name_from_filename(csv_path)
        
        logger.info("Target table: %s.%s.%s", self.project_id, self.dataset_id, table_name)
        
        # Parse CSV
        records = self.parse_csv_file(csv_path)
        
        if not records:
            logger.warning("No valid records found in CSV file")
            return {
                "total_processed": 0,
                "total_errors": 0,
                "table_name": table_name,
                "table_ref": f"{self.project_id}.{self.dataset_id}.{table_name}"
            }
        
        # Show sample of parsed data
        logger.info("Sample record structure: %s", list(records[0].data.keys()))
        
        # Ingest to BigQuery
        result = self.ingest_to_bigquery(table_name, records, batch_size)
        
        logger.info("Processing pipeline complete")
        return result 