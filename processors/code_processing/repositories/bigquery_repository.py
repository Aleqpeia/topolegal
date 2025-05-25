from typing import List, Dict, Any, Optional
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging
from datetime import datetime

from .base import LegalCodeRepository
from ..models import LegalCodeEntry


logger = logging.getLogger(__name__)


class BigQueryLegalCodeRepository(LegalCodeRepository):
    """BigQuery implementation of legal code repository"""
    
    def __init__(self, project_id: str, dataset_id: str, table_id: str = "legal_code_entries"):
        """
        Initialize BigQuery repository
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID (default: legal_code_entries)
        """
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
    def create_table(self) -> None:
        """Create the BigQuery table with the required schema"""
        schema = [
            bigquery.SchemaField("code_entry", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("code_entry_text", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("additional_info", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("article_number", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("article_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("paragraph", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("paragraph_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("heading", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("heading_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("section", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("section_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        table = bigquery.Table(self.full_table_id, schema=schema)
        
        try:
            self.client.get_table(self.full_table_id)
            logger.info(f"Table {self.full_table_id} already exists")
        except NotFound:
            table = self.client.create_table(table)
            logger.info(f"Created table {self.full_table_id}")
    
    def _entry_to_row(self, entry: LegalCodeEntry) -> Dict[str, Any]:
        """Convert LegalCodeEntry to BigQuery row"""
        return {
            "code_entry": entry.code_entry,
            "code_entry_text": entry.code_entry_text,
            "additional_info": entry.additional_info,
            "article_number": entry.article_number,
            "article_name": entry.article_name,
            "paragraph": entry.paragraph,
            "paragraph_name": entry.paragraph_name,
            "heading": entry.heading,
            "heading_name": entry.heading_name,
            "section": entry.section,
            "section_name": entry.section_name,
            "created_at": datetime.utcnow().isoformat(),
        }
    
    def _row_to_entry(self, row: Dict[str, Any]) -> LegalCodeEntry:
        """Convert BigQuery row to LegalCodeEntry"""
        return LegalCodeEntry(
            code_entry=row.get("code_entry"),
            code_entry_text=row.get("code_entry_text", ""),
            additional_info=row.get("additional_info"),
            article_number=row.get("article_number"),
            article_name=row.get("article_name"),
            paragraph=row.get("paragraph"),
            paragraph_name=row.get("paragraph_name"),
            heading=row.get("heading"),
            heading_name=row.get("heading_name"),
            section=row.get("section"),
            section_name=row.get("section_name"),
        )
    
    def insert_entry(self, entry: LegalCodeEntry) -> None:
        """Insert a single legal code entry"""
        table = self.client.get_table(self.full_table_id)
        rows_to_insert = [self._entry_to_row(entry)]
        
        errors = self.client.insert_rows_json(table, rows_to_insert)
        if errors:
            raise Exception(f"Failed to insert row: {errors}")
        
        logger.info(f"Inserted 1 entry to {self.full_table_id}")
    
    def insert_entries_batch(self, entries: List[LegalCodeEntry]) -> None:
        """Insert multiple legal code entries in batch"""
        if not entries:
            return
        
        table = self.client.get_table(self.full_table_id)
        rows_to_insert = [self._entry_to_row(entry) for entry in entries]
        
        # BigQuery recommends chunks of 500 rows or less
        chunk_size = 500
        for i in range(0, len(rows_to_insert), chunk_size):
            chunk = rows_to_insert[i:i + chunk_size]
            errors = self.client.insert_rows_json(table, chunk)
            if errors:
                raise Exception(f"Failed to insert batch: {errors}")
        
        logger.info(f"Inserted {len(entries)} entries to {self.full_table_id}")
    
    def get_entry_by_article(self, article_number: int) -> List[LegalCodeEntry]:
        """Get all entries for a specific article"""
        query = f"""
        SELECT *
        FROM `{self.full_table_id}`
        WHERE article_number = @article_number
        ORDER BY paragraph, code_entry
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("article_number", "INT64", article_number)
            ]
        )
        
        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()
        
        return [self._row_to_entry(dict(row)) for row in results]
    
    def get_entries_by_section(self, section: str) -> List[LegalCodeEntry]:
        """Get all entries for a specific section"""
        query = f"""
        SELECT *
        FROM `{self.full_table_id}`
        WHERE section = @section
        ORDER BY article_number, paragraph, code_entry
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("section", "STRING", section)
            ]
        )
        
        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()
        
        return [self._row_to_entry(dict(row)) for row in results]
    
    def search_entries(self, search_text: str) -> List[LegalCodeEntry]:
        """Search entries by text content"""
        query = f"""
        SELECT *
        FROM `{self.full_table_id}`
        WHERE LOWER(code_entry_text) LIKE LOWER(@search_text)
           OR LOWER(additional_info) LIKE LOWER(@search_text)
        ORDER BY section, article_number, paragraph, code_entry
        LIMIT 100
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("search_text", "STRING", f"%{search_text}%")
            ]
        )
        
        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()
        
        return [self._row_to_entry(dict(row)) for row in results]
    
    def delete_all_entries(self) -> None:
        """Delete all entries from the table"""
        query = f"DELETE FROM `{self.full_table_id}` WHERE TRUE"
        query_job = self.client.query(query)
        query_job.result()
        logger.info(f"Deleted all entries from {self.full_table_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the stored entries"""
        query = f"""
        SELECT 
            COUNT(*) as total_entries,
            COUNT(DISTINCT section) as total_sections,
            COUNT(DISTINCT heading) as total_headings,
            COUNT(DISTINCT article_number) as total_articles,
            COUNT(DISTINCT CONCAT(article_number, '-', paragraph)) as total_paragraphs
        FROM `{self.full_table_id}`
        """
        
        query_job = self.client.query(query)
        results = list(query_job.result())
        
        if results:
            row = dict(results[0])
            return {
                "total_entries": row["total_entries"],
                "total_sections": row["total_sections"],
                "total_headings": row["total_headings"],
                "total_articles": row["total_articles"],
                "total_paragraphs": row["total_paragraphs"],
            }
        
        return {
            "total_entries": 0,
            "total_sections": 0,
            "total_headings": 0,
            "total_articles": 0,
            "total_paragraphs": 0,
        } 