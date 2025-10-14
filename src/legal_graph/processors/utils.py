"""
BigQuery Utilities for Processors Module

Centralized BigQuery utilities to reduce boilerplate across child processor modules.
Provides common functions for:
- Client initialization
- Schema management
- Data querying and insertion
- Batch processing
- Statistics gathering
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BigQueryClient:
    """
    Wrapper around google.cloud.bigquery.Client with common functionality
    for legal document processing.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        key_path: Optional[str] = None,
        location: str = "us-central1",
    ):
        """
        Initialize BigQuery client with optional credentials.

        Args:
            project: Google Cloud Project ID (uses env var if None)
            key_path: Path to service account key file
            location: BigQuery location/region
        """
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery not installed. "
                "Install with: pip install google-cloud-bigquery"
            )

        # Setup credentials
        if key_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        if project:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project

        self.client = bigquery.Client(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=location
        )
        self.bigquery = bigquery
        logger.info(f"Initialized BigQuery client for project: {self.client.project}")

    def query(self, sql: str, **kwargs) -> Any:
        """Execute a SQL query and return results."""
        job = self.client.query(sql, **kwargs)
        return job.result()

    def query_to_dataframe(self, sql: str, **kwargs):
        """Execute a SQL query and return results as pandas DataFrame."""
        job = self.client.query(sql, **kwargs)
        return job.result().to_dataframe()

    def insert_rows_json(self, table_id: str, rows: List[Dict]) -> List:
        """
        Insert rows into BigQuery table.

        Args:
            table_id: BigQuery table ID (project.dataset.table)
            rows: List of dictionaries representing rows

        Returns:
            List of errors (empty if successful)
        """
        errors = self.client.insert_rows_json(table_id, rows)
        if errors:
            logger.error(f"Errors inserting {len(errors)} rows: {errors[:5]}")
        else:
            logger.debug(f"Successfully inserted {len(rows)} rows into {table_id}")
        return errors

    def create_table(self, table_id: str, schema: List, exists_ok: bool = True):
        """
        Create a BigQuery table with given schema.

        Args:
            table_id: BigQuery table ID
            schema: List of SchemaField objects
            exists_ok: If True, don't raise error if table exists
        """
        table = self.bigquery.Table(table_id, schema=schema)
        table = self.client.create_table(table, exists_ok=exists_ok)
        logger.info(f"Created/verified table {table_id}")
        return table

    def table_exists(self, table_id: str) -> bool:
        """Check if a BigQuery table exists."""
        try:
            self.client.get_table(table_id)
            return True
        except Exception:
            return False

    def get_table_schema(self, table_id: str) -> Dict:
        """
        Get schema information for a BigQuery table.

        Returns:
            Dictionary mapping column names to types
        """
        try:
            table = self.client.get_table(table_id)
            schema = {field.name: field.field_type for field in table.schema}
            return schema
        except Exception as e:
            logger.error(f"Error getting schema for {table_id}: {e}")
            return {}

    def check_required_columns(
        self, table_id: str, required_columns: List[str]
    ) -> tuple[bool, List[str]]:
        """
        Check if table has required columns.

        Args:
            table_id: BigQuery table ID
            required_columns: List of required column names

        Returns:
            Tuple of (all_present: bool, missing_columns: List[str])
        """
        schema = self.get_table_schema(table_id)
        missing = [col for col in required_columns if col not in schema]
        return len(missing) == 0, missing

    def get_table_stats(self, table_id: str) -> Dict:
        """
        Get basic statistics about a table.

        Returns:
            Dictionary with row_count, size_bytes, etc.
        """
        try:
            table = self.client.get_table(table_id)
            return {
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created,
                "modified": table.modified,
                "schema_fields": len(table.schema),
            }
        except Exception as e:
            logger.error(f"Error getting stats for {table_id}: {e}")
            return {}


class BigQueryBatchProcessor:
    """
    Batch processor for BigQuery data with progress tracking
    and error handling.
    """

    def __init__(
        self, client: BigQueryClient, batch_size: int = 100, log_interval: int = 10
    ):
        """
        Initialize batch processor.

        Args:
            client: BigQueryClient instance
            batch_size: Number of rows to process per batch
            log_interval: Log progress every N batches
        """
        self.client = client
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0

    def process_table(
        self,
        table_id: str,
        processor_func: Callable[[List[Dict]], List[Dict]],
        output_table_id: Optional[str] = None,
        where_clause: Optional[str] = None,
        select_columns: Optional[List[str]] = None,
    ) -> Dict:
        """
        Process all rows in a table in batches.

        Args:
            table_id: Source table ID
            processor_func: Function to process each batch (takes list of dicts, returns list of dicts)
            output_table_id: Optional destination table for results
            where_clause: Optional SQL WHERE clause for filtering
            select_columns: Optional list of columns to select (defaults to *)

        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Starting batch processing: {table_id}")
        logger.info(f"Batch size: {self.batch_size}")

        columns = ", ".join(select_columns) if select_columns else "*"
        where = f"WHERE {where_clause}" if where_clause else ""

        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0

        while True:
            # Fetch batch
            sql = f"""
                SELECT {columns}
                FROM `{table_id}`
                {where}
                LIMIT {self.batch_size}
                OFFSET {self.processed_count}
            """

            rows_df = self.client.query_to_dataframe(sql)

            if rows_df.empty:
                logger.info("✅ No more rows to process")
                break

            # Convert to list of dicts
            batch_data = rows_df.to_dict("records")

            try:
                # Process batch
                processed_data = processor_func(batch_data)

                # Insert to output table if specified
                if output_table_id and processed_data:
                    errors = self.client.insert_rows_json(
                        output_table_id, processed_data
                    )
                    if errors:
                        self.error_count += len(errors)
                    else:
                        self.success_count += len(processed_data)

                self.processed_count += len(batch_data)

                # Log progress
                if self.processed_count % (self.batch_size * self.log_interval) == 0:
                    logger.info(
                        f"📊 Processed {self.processed_count} rows "
                        f"(success: {self.success_count}, errors: {self.error_count})"
                    )

            except Exception as e:
                logger.error(
                    f"Error processing batch at offset {self.processed_count}: {e}"
                )
                self.error_count += len(batch_data)
                self.processed_count += len(batch_data)
                continue

        return {
            "total_processed": self.processed_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
        }


# ============================================================================
# BigQuery Dataset Reformatting (moved to balance module)
# ============================================================================

# For backward compatibility, import from balance module
try:
    from .balance import BigQueryDatasetReformatter
except ImportError:
    logger.warning(
        "Balance module not available. BigQueryDatasetReformatter not imported."
    )
    BigQueryDatasetReformatter = None

# Legacy alias for backward compatibility
BigQueryJSONFunctions = BigQueryDatasetReformatter


def create_legal_document_table(
    client: BigQueryClient,
    table_id: str,
    include_entities: bool = True,
    include_triplets: bool = True,
    include_labels: bool = False,
    label_column: str = "is_valid",
):
    """
    Create a standard legal document table schema.

    Args:
        client: BigQueryClient instance
        table_id: Table ID to create
        include_entities: Include entities/tags column
        include_triplets: Include triplets column
        include_labels: Include label column for classification
        label_column: Name of the label column
    """
    schema = [
        client.bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
        client.bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
    ]

    if include_entities:
        schema.append(client.bigquery.SchemaField("entities", "JSON", mode="NULLABLE"))
        schema.append(
            client.bigquery.SchemaField("tags", "STRING", mode="NULLABLE")
        )  # Legacy

    if include_triplets:
        schema.append(client.bigquery.SchemaField("triplets", "JSON", mode="NULLABLE"))
        schema.append(
            client.bigquery.SchemaField("triplets_count", "INTEGER", mode="NULLABLE")
        )

    if include_labels:
        schema.append(
            client.bigquery.SchemaField(label_column, "BOOLEAN", mode="REQUIRED")
        )

    schema.append(
        client.bigquery.SchemaField(
            "processing_timestamp", "TIMESTAMP", mode="NULLABLE"
        )
    )

    return client.create_table(table_id, schema, exists_ok=True)


def check_legal_document_schema(
    client: BigQueryClient,
    table_id: str,
    require_entities: bool = False,
    require_triplets: bool = False,
) -> tuple[bool, List[str]]:
    """
    Check if table has standard legal document schema.

    Args:
        client: BigQueryClient instance
        table_id: Table ID to check
        require_entities: Require entities/tags column
        require_triplets: Require triplets column

    Returns:
        Tuple of (is_valid: bool, missing_columns: List[str])
    """
    required_columns = ["doc_id", "text"]

    if require_entities:
        required_columns.append("tags")  # Accept either tags or entities

    if require_triplets:
        required_columns.extend(["triplets", "triplets_count"])

    is_valid, missing = client.check_required_columns(table_id, required_columns)

    if not is_valid:
        logger.warning(f"Table {table_id} missing required columns: {missing}")
        logger.warning(f"Expected schema: {required_columns}")

    return is_valid, missing


def get_triplet_statistics(client: BigQueryClient, table_id: str) -> Dict:
    """
    Get statistics about triplets in a table.

    Args:
        client: BigQueryClient instance
        table_id: Table ID to analyze

    Returns:
        Dictionary with triplet statistics
    """
    sql = f"""
    SELECT
        COUNT(*) as total_documents,
        COUNTIF(triplets IS NOT NULL) as docs_with_triplets,
        COUNTIF(triplets_count > 0) as docs_with_positive_count,
        AVG(triplets_count) as avg_triplets_per_doc,
        MAX(triplets_count) as max_triplets_per_doc,
        SUM(triplets_count) as total_triplets
    FROM `{table_id}`
    """

    try:
        result = client.query_to_dataframe(sql)
        if not result.empty:
            return result.iloc[0].to_dict()
        return {}
    except Exception as e:
        logger.error(f"Error getting triplet statistics: {e}")
        return {}


def parse_json_column(value: Any) -> Any:
    """
    Safely parse a JSON column value from BigQuery.

    Handles multiple formats:
    - Already parsed Python objects (list, dict)
    - NumPy arrays
    - JSON strings

    Args:
        value: Column value from BigQuery

    Returns:
        Parsed Python object (list or dict) or None
    """
    import numpy as np

    # Already a Python object
    if isinstance(value, (list, dict)):
        return value

    # NumPy array (from BigQuery JSON columns)
    if isinstance(value, np.ndarray):
        return value.tolist()

    # JSON string
    if isinstance(value, str) and value.strip() and value != "nan":
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None

    return None


def format_table_stats(stats: Dict) -> str:
    """
    Format table statistics for logging.

    Args:
        stats: Statistics dictionary from get_triplet_statistics

    Returns:
        Formatted string for logging
    """
    lines = [
        "📊 Table Statistics:",
        f"  Total documents: {stats.get('total_documents', 0):,}",
        f"  Documents with triplets: {stats.get('docs_with_triplets', 0):,}",
        f"  Documents with positive count: {stats.get('docs_with_positive_count', 0):,}",
        f"  Average triplets per doc: {stats.get('avg_triplets_per_doc', 0):.2f}",
        f"  Max triplets per doc: {stats.get('max_triplets_per_doc', 0):,}",
        f"  Total triplets: {stats.get('total_triplets', 0):,}",
    ]
    return "\n".join(lines)


__all__ = [
    "BigQueryClient",
    "BigQueryBatchProcessor",
    "BigQueryDatasetReformatter",  # Preferred name
    "BigQueryJSONFunctions",  # Legacy alias for backward compatibility
    "create_legal_document_table",
    "check_legal_document_schema",
    "get_triplet_statistics",
    "parse_json_column",
    "format_table_stats",
]
