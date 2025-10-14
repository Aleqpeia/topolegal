"""
Array-Based Reformatting

Instead of exploding triplets into multiple rows (UNNEST), this module keeps
triplets as ARRAY<STRUCT> for more efficient storage and querying.

Benefits:
- Preserves document structure (one row per document)
- More efficient storage (no row duplication)
- Faster queries when working with full documents
- Better for aggregations and statistics
- Native BigQuery ARRAY support

Trade-offs:
- More complex queries for triplet-level operations
- Requires UNNEST in queries that need triplet-level access
- GraphDataset loader needs updates to handle arrays
"""

from google.cloud import bigquery
from google.oauth2 import service_account
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ArrayBasedReformatter:
    """Reformats dataset using ARRAY<STRUCT> instead of row explosion."""

    def __init__(self, project_id: str, credentials_path: str = None):
        """Initialize BigQuery client."""
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            self.client = bigquery.Client(credentials=credentials, project=project_id)
        else:
            self.client = bigquery.Client(project=project_id)

    def reformat_to_arrays(self, source_table: str, dest_table: str) -> None:
        """
        Reformat dataset keeping triplets as ARRAY<STRUCT>.

        Schema:
            doc_id: STRING
            text: STRING
            tags: STRING (JSON)
            triplets_count: INT64
            triplets: ARRAY<STRUCT<
                legal_reference: STRING,
                source: STRING,
                target: STRING,
                relation: STRING,
                confidence: FLOAT64,
                article_number: STRING,
                part_number: STRING,
                legal_code: STRING,
                legal_reference_grouped: STRING,
                relation_grouped: STRING
            >>
            reformatted_timestamp: TIMESTAMP

        Args:
            source_table: Source table with JSON triplets
            dest_table: Destination table with structured arrays
        """
        logger.info(f"Reformatting to array format: {source_table} → {dest_table}")

        query = f"""
        CREATE OR REPLACE TABLE `{dest_table}` AS
        WITH triplet_data AS (
            SELECT
                doc_id,
                text,
                tags,
                CAST(triplets_count AS INT64) AS triplets_count,
                JSON_EXTRACT_ARRAY(triplets) AS triplet_array
            FROM `{source_table}`
            WHERE triplets IS NOT NULL
                AND triplets != '[]'
                AND triplets_count > 0
        )
        SELECT
            doc_id,
            text,
            tags,
            triplets_count,

            -- Create array of structs with all triplet data + metadata
            ARRAY(
                SELECT AS STRUCT
                    JSON_EXTRACT_SCALAR(triplet, '$.legal_reference') AS legal_reference,
                    JSON_EXTRACT_SCALAR(triplet, '$.source') AS source,
                    JSON_EXTRACT_SCALAR(triplet, '$.target') AS target,
                    JSON_EXTRACT_SCALAR(triplet, '$.relation') AS relation,
                    CAST(JSON_EXTRACT_SCALAR(triplet, '$.confidence') AS FLOAT64) AS confidence,

                    -- Metadata fields
                    REGEXP_EXTRACT(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'ст\\.\\s*(\\d+)') AS article_number,
                    REGEXP_EXTRACT(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'ч\\.\\s*(\\d+)') AS part_number,
                    REGEXP_EXTRACT(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'(КК|КПК|ЦК|КоАП|СК|КЗпП)\\s*України') AS legal_code,

                    -- Grouped fields (enhanced strategy)
                    CONCAT(
                        'ст.',
                        REGEXP_EXTRACT(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'ст\\.\\s*(\\d+(?:-\\d+)?)'),
                        ' ',
                        CASE
                            WHEN REGEXP_CONTAINS(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'КПК\\s*України') THEN 'КПК України'
                            WHEN REGEXP_CONTAINS(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'КК\\s*України') THEN 'КК України'
                            WHEN REGEXP_CONTAINS(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'ЦК\\s*України') THEN 'ЦК України'
                            WHEN REGEXP_CONTAINS(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'КоАП\\s*України') THEN 'КоАП України'
                            WHEN REGEXP_CONTAINS(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'СК\\s*України') THEN 'СК України'
                            WHEN REGEXP_CONTAINS(JSON_EXTRACT_SCALAR(triplet, '$.legal_reference'), r'КЗпП\\s*України') THEN 'КЗпП України'
                            ELSE 'OTHER'
                        END
                    ) AS legal_reference_grouped,

                    SPLIT(JSON_EXTRACT_SCALAR(triplet, '$.relation'), '_')[SAFE_OFFSET(0)] AS relation_grouped

                FROM UNNEST(triplet_array) AS triplet
            ) AS triplets,

            CURRENT_TIMESTAMP() AS reformatted_timestamp

        FROM triplet_data
        ORDER BY doc_id
        """

        job = self.client.query(query)
        result = job.result()

        logger.info(f"✅ Array-based reformatting complete")
        logger.info(f"   Destination: {dest_table}")
        logger.info(f"   Documents: {result.total_rows}")

    def create_flattened_view(self, array_table: str, view_name: str) -> None:
        """
        Create a flattened view from array-based table.

        This creates a traditional row-per-triplet view for queries that
        need triplet-level access (e.g., training data loading).

        Args:
            array_table: Source table with array format
            view_name: View name for flattened access
        """
        logger.info(f"Creating flattened view: {view_name}")

        query = f"""
        CREATE OR REPLACE VIEW `{view_name}` AS
        SELECT
            doc_id,
            text,
            tags,
            triplets_count,
            CAST(ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY doc_id) AS INT64) AS triplet_index,

            -- Triplet fields
            triplet.legal_reference,
            triplet.source,
            triplet.target,
            triplet.relation,
            triplet.confidence,

            -- Metadata fields
            triplet.article_number,
            triplet.part_number,
            triplet.legal_code,

            -- Grouped fields
            triplet.legal_reference_grouped,
            triplet.relation_grouped

        FROM `{array_table}`,
        UNNEST(triplets) AS triplet
        ORDER BY doc_id, triplet_index
        """

        job = self.client.query(query)
        job.result()

        logger.info(f"✅ Flattened view created: {view_name}")

    def analyze_array_statistics(self, array_table: str) -> dict:
        """
        Analyze statistics for array-based table.

        Args:
            array_table: Table with array format

        Returns:
            Dictionary with statistics
        """
        logger.info(f"Analyzing array-based table: {array_table}")

        query = f"""
        WITH triplet_stats AS (
            SELECT
                COUNT(DISTINCT doc_id) AS unique_docs,
                SUM(triplets_count) AS total_triplets,
                AVG(triplets_count) AS avg_triplets_per_doc,
                MIN(triplets_count) AS min_triplets_per_doc,
                MAX(triplets_count) AS max_triplets_per_doc
            FROM `{array_table}`
        ),
        flattened_stats AS (
            SELECT
                COUNT(DISTINCT triplet.legal_reference_grouped) AS unique_legal_refs_grouped,
                COUNT(DISTINCT triplet.relation_grouped) AS unique_relations_grouped,
                COUNT(DISTINCT triplet.legal_reference) AS unique_legal_refs_original,
                COUNT(DISTINCT triplet.relation) AS unique_relations_original
            FROM `{array_table}`,
            UNNEST(triplets) AS triplet
        )
        SELECT * FROM triplet_stats, flattened_stats
        """

        job = self.client.query(query)
        result = list(job.result())[0]

        stats = {
            "unique_docs": result.unique_docs,
            "total_triplets": result.total_triplets,
            "avg_triplets_per_doc": float(result.avg_triplets_per_doc),
            "min_triplets_per_doc": result.min_triplets_per_doc,
            "max_triplets_per_doc": result.max_triplets_per_doc,
            "unique_legal_refs_original": result.unique_legal_refs_original,
            "unique_legal_refs_grouped": result.unique_legal_refs_grouped,
            "unique_relations_original": result.unique_relations_original,
            "unique_relations_grouped": result.unique_relations_grouped,
        }

        # Calculate reductions
        stats["legal_ref_reduction_pct"] = (
            (stats["unique_legal_refs_original"] - stats["unique_legal_refs_grouped"])
            / stats["unique_legal_refs_original"]
            * 100
        )
        stats["relation_reduction_pct"] = (
            (stats["unique_relations_original"] - stats["unique_relations_grouped"])
            / stats["unique_relations_original"]
            * 100
        )

        logger.info(f"✅ Statistics:")
        logger.info(f"   Documents: {stats['unique_docs']}")
        logger.info(f"   Total triplets: {stats['total_triplets']}")
        logger.info(f"   Avg per doc: {stats['avg_triplets_per_doc']:.1f}")
        logger.info(
            f"   Legal refs: {stats['unique_legal_refs_original']} → {stats['unique_legal_refs_grouped']} ({stats['legal_ref_reduction_pct']:.1f}% reduction)"
        )
        logger.info(
            f"   Relations: {stats['unique_relations_original']} → {stats['unique_relations_grouped']} ({stats['relation_reduction_pct']:.1f}% reduction)"
        )

        return stats

    def create_array_splits(
        self,
        array_table: str,
        doc_ids_train: list,
        doc_ids_test: list,
        doc_ids_val: list,
        train_table: str,
        test_table: str,
        val_table: str,
    ) -> None:
        """
        Create train/test/val splits preserving array format.

        Args:
            array_table: Source table with arrays
            doc_ids_train: List of training document IDs
            doc_ids_test: List of test document IDs
            doc_ids_val: List of validation document IDs
            train_table: Training table destination
            test_table: Test table destination
            val_table: Validation table destination
        """
        splits = [
            (doc_ids_train, train_table, "train"),
            (doc_ids_test, test_table, "test"),
            (doc_ids_val, val_table, "val"),
        ]

        for doc_ids, dest_table, split_name in splits:
            # Create temporary table with doc IDs
            temp_table = f"{dest_table}_temp_ids"

            # Insert doc IDs
            doc_ids_str = ",".join([f"'{doc_id}'" for doc_id in doc_ids])

            query = f"""
            CREATE OR REPLACE TABLE `{dest_table}` AS
            SELECT *
            FROM `{array_table}`
            WHERE doc_id IN ({doc_ids_str})
            """

            job = self.client.query(query)
            result = job.result()

            logger.info(
                f"✅ {split_name.upper()} split created: {dest_table} ({result.total_rows} docs)"
            )


# ============================================================================
# Comparison: Row-Based vs Array-Based
# ============================================================================

COMPARISON = """
================================================================================
ROW-BASED vs ARRAY-BASED REFORMATTING
================================================================================

ROW-BASED (Current - UNNEST approach):
--------------------------------------
Schema:
  doc_id, text, tags, triplet_index,
  legal_reference, source, target, relation, confidence,
  article_number, part_number, legal_code

Storage:
  • One row per triplet
  • Document fields (doc_id, text, tags) duplicated for each triplet
  • Example: 1000 docs × 10 triplets/doc = 10,000 rows

Pros:
  ✓ Simple queries (one row = one triplet)
  ✓ Easy to filter/aggregate at triplet level
  ✓ Compatible with traditional SQL tools
  ✓ Current GraphDataset loader works as-is

Cons:
  ❌ Storage inefficient (doc fields duplicated)
  ❌ More rows to scan for document-level queries
  ❌ JOIN overhead for document-level operations


ARRAY-BASED (New - ARRAY<STRUCT> approach):
--------------------------------------------
Schema:
  doc_id, text, tags, triplets_count,
  triplets: ARRAY<STRUCT<
    legal_reference, source, target, relation, confidence,
    article_number, part_number, legal_code,
    legal_reference_grouped, relation_grouped
  >>

Storage:
  • One row per document
  • Triplets stored as structured array
  • Example: 1000 docs = 1,000 rows (regardless of triplets/doc)

Pros:
  ✓ 5-10x storage reduction (no duplication)
  ✓ Faster document-level queries
  ✓ Preserves document structure
  ✓ Better for aggregations
  ✓ Native BigQuery ARRAY support
  ✓ Grouped fields pre-computed in array

Cons:
  ❌ Triplet-level queries need UNNEST
  ❌ Slightly more complex queries
  ❌ GraphDataset loader needs update


RECOMMENDATION
--------------

Use ARRAY-BASED for:
  ✓ Primary storage (more efficient)
  ✓ Document-level operations
  ✓ Statistics and analysis
  ✓ Long-term storage

Create FLATTENED VIEW for:
  ✓ Model training (GraphDataset)
  ✓ Triplet-level queries
  ✓ Backward compatibility

Best of both worlds:
  • Store as arrays (efficient)
  • Query via flattened view (simple)
  • Update GraphDataset to use flattened view

================================================================================
MIGRATION PATH
================================================================================

1. Reformat to array-based table:
   reformatter.reformat_to_arrays(source, dest)

2. Create flattened view for training:
   reformatter.create_flattened_view(dest, "dest_flattened")

3. Update GraphDataset to use flattened view:
   dataset = GraphDataset(table_id="project.dataset.dest_flattened")

4. Keep both:
   • Array table for storage/analysis
   • Flattened view for training

================================================================================
"""
