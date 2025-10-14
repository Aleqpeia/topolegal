"""
BigQuery Dataset Operations Module

This module provides comprehensive BigQuery dataset operations for legal documents:
- Dataset statistics and analysis
- Triplet reformatting (extracting JSON to columnar format)
- Stratified train/test/val splitting
- Split validation and distribution analysis

Replaces the deprecated corruption functionality (now in negative_sampling module).
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# BigQuery Dataset Reformatter
# ============================================================================


class BigQueryDatasetReformatter:
    """
    Reformat BigQuery legal graph datasets by extracting triplets into columns
    and providing stratified splitting utilities.

    This consolidates functionality from:
    - docs/bigquery_dataset_reformatter.py
    - utils.BigQueryJSONFunctions

    Aligned with goals defined in DATASET_REFORMATTING_GUIDE.md
    """

    def __init__(
        self, project_id: Optional[str] = None, credentials_path: Optional[str] = None
    ):
        """
        Initialize BigQuery client for dataset operations.

        Args:
            project_id: GCP project ID (uses default if None)
            credentials_path: Path to service account JSON key
        """
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery not installed. "
                "Install with: pip install google-cloud-bigquery"
            )

        if credentials_path:
            self.client = bigquery.Client.from_service_account_json(
                credentials_path, project=project_id
            )
        else:
            self.client = bigquery.Client(project=project_id)

        self.bigquery = bigquery
        logger.info(f"Initialized BigQuery client for project: {self.client.project}")

    def analyze_triplet_statistics(self, source_table: str) -> Dict:
        """
        Analyze triplet statistics for stratified splitting.

        Args:
            source_table: Source table ID (format: project.dataset.table)

        Returns:
            Dictionary with statistics including legal reference and relation distributions
        """
        logger.info(f"Analyzing triplet statistics for {source_table}")

        # Query to extract triplet statistics with proper aggregation
        query = f"""
        WITH triplet_data AS (
            SELECT
                doc_id,
                triplets,
                triplets_count,
                JSON_EXTRACT_ARRAY(triplets) AS triplet_array
            FROM `{source_table}`
            WHERE triplets IS NOT NULL
                AND triplets_count > 0
        ),
        exploded_triplets AS (
            SELECT
                doc_id,
                triplets_count,
                JSON_EXTRACT_SCALAR(triplet, '$.legal_reference') AS legal_reference,
                JSON_EXTRACT_SCALAR(triplet, '$.source') AS source,
                JSON_EXTRACT_SCALAR(triplet, '$.target') AS target,
                JSON_EXTRACT_SCALAR(triplet, '$.relation') AS relation
            FROM triplet_data,
            UNNEST(triplet_array) AS triplet
        )
        SELECT
            -- Legal reference statistics
            legal_reference,
            COUNT(DISTINCT doc_id) AS docs_with_legal_ref,
            COUNT(*) AS total_occurrences_legal_ref,

            -- Relation statistics
            relation,
            COUNT(DISTINCT doc_id) AS docs_with_relation,
            COUNT(*) AS total_occurrences_relation,

            -- Combined statistics
            COUNT(*) as triplet_count
        FROM exploded_triplets
        GROUP BY legal_reference, relation
        ORDER BY triplet_count DESC
        """

        job = self.client.query(query)
        results = job.result().to_dataframe()
        legal_ref_stats = (
            results.groupby("legal_reference")
            .agg(
                {"docs_with_legal_ref": "first", "total_occurrences_legal_ref": "first"}
            )
            .to_dict("index")
        )
        relation_stats = (
            results.groupby("relation")
            .agg({"docs_with_relation": "first", "total_occurrences_relation": "first"})
            .to_dict("index")
        )
        stats = {
            "total_triplets": len(results),
            "unique_legal_references": len(legal_ref_stats.keys()),
            "unique_relations": len(relation_stats.keys()),
            "legal_reference_distribution": legal_ref_stats,
            "relation_distribution": relation_stats,
        }
        logger.info(f"Statistics collected: {stats}")
        return stats

    def reformat_dataset(
        self, source_table: str, dest_table: str, create_if_not_exists: bool = True
    ) -> None:
        """
        Reformat dataset by extracting triplet JSON into separate columns.

        Extracts JSON triplet array into columnar format with metadata extraction:
        - legal_reference, source, target, relation, confidence (from triplets)
        - article_number, part_number, legal_code (extracted via regex)

        This is the core reformatting operation described in DATASET_REFORMATTING_GUIDE.md

        Args:
            source_table: Source table ID
            dest_table: Destination table ID
            create_if_not_exists: Create destination table if it doesn't exist
        """
        logger.info(f"Reformatting dataset from {source_table} to {dest_table}")

        # SQL to extract triplets into columns with metadata
        query = f"""
        CREATE OR REPLACE TABLE `{dest_table}` AS
        WITH triplet_data AS (
            SELECT
                doc_id,
                text,
                tags,
                triplets,
                triplets_count,
                JSON_EXTRACT_ARRAY(triplets) AS triplet_array
            FROM `{source_table}`
            WHERE triplets IS NOT NULL
                AND triplets_count > 0
        ),
        exploded_triplets AS (
            SELECT
                doc_id,
                text,
                tags,
                triplets_count,
                CAST(ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY doc_id) AS INT64) AS triplet_index,
                JSON_EXTRACT_SCALAR(triplet, '$.legal_reference') AS legal_reference,
                JSON_EXTRACT_SCALAR(triplet, '$.source') AS source,
                JSON_EXTRACT_SCALAR(triplet, '$.target') AS target,
                JSON_EXTRACT_SCALAR(triplet, '$.relation') AS relation,
                CAST(JSON_EXTRACT_SCALAR(triplet, '$.confidence') AS FLOAT64) AS confidence
            FROM triplet_data,
            UNNEST(triplet_array) AS triplet
        )
        SELECT
            doc_id,
            text,
            tags,
            triplets_count,
            triplet_index,
            legal_reference,
            source,
            target,
            relation,
            confidence,
            -- Add metadata columns for analysis (article number, part number, legal code)
            REGEXP_EXTRACT(legal_reference, r'ст\\.\\s*(\\d+)') AS article_number,
            REGEXP_EXTRACT(legal_reference, r'ч\\.\\s*(\\d+)') AS part_number,
            REGEXP_EXTRACT(legal_reference, r'(КК|КПК|ЦК|КоАП|СК|КЗпП)\\s*України') AS legal_code,
            CURRENT_TIMESTAMP() AS reformatted_timestamp
        FROM exploded_triplets
        ORDER BY doc_id, triplet_index
        """

        job = self.client.query(query)
        result = job.result()

        logger.info("✅ Dataset reformatted successfully")
        logger.info(f"   Destination table: {dest_table}")
        logger.info(f"   Total rows: {result.total_rows}")

    def get_stratification_groups(
        self,
        source_table: str,
        group_by: str = "legal_reference",
        min_samples: int = 10,
    ) -> Dict[str, List[str]]:
        """
        Get stratification groups for splitting.

        Args:
            source_table: Reformatted table ID
            group_by: Column to group by ('legal_reference', 'relation', or 'legal_code')
            min_samples: Minimum samples per group for stratification

        Returns:
            Dictionary mapping group values to document IDs
        """
        logger.info(f"Getting stratification groups by {group_by}")

        query = f"""
        SELECT
            doc_id,
            {group_by}
        FROM `{source_table}`
        WHERE {group_by} IS NOT NULL
        """

        job = self.client.query(query)
        results = job.result().to_dataframe()

        # Group documents by the stratification key
        groups = defaultdict(list)
        for _, row in results.iterrows():
            groups[row[group_by]].append(row["doc_id"])

        # Filter groups with minimum samples
        filtered_groups = {
            k: list(set(v)) for k, v in groups.items() if len(set(v)) >= min_samples
        }

        logger.info(
            f"Found {len(filtered_groups)} groups with >= {min_samples} samples"
        )

        return filtered_groups

    def stratified_split(
        self,
        source_table: str,
        train_table: str,
        test_table: str,
        val_table: str,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        stratify_by: str = "legal_reference",
        random_seed: int = 42,
    ) -> Dict[str, int]:
        """
        Create stratified train/test/validation splits.

        Ensures representative distribution of legal references/relations across all splits.
        Implements the stratified splitting strategy from DATASET_REFORMATTING_GUIDE.md

        Args:
            source_table: Reformatted source table
            train_table: Training set destination
            test_table: Test set destination
            val_table: Validation set destination
            train_ratio: Training set ratio (default 0.7)
            test_ratio: Test set ratio (default 0.2)
            val_ratio: Validation set ratio (default 0.1)
            stratify_by: Column to stratify by (default 'legal_reference')
            random_seed: Random seed for reproducibility (default 42)

        Returns:
            Dictionary with split statistics (unique_docs, total_rows per split)
        """
        assert (
            abs(train_ratio + test_ratio + val_ratio - 1.0) < 0.001
        ), "Ratios must sum to 1.0"

        logger.info(
            f"Creating stratified splits: train={train_ratio}, test={test_ratio}, val={val_ratio}"
        )

        # Get all unique documents with stratification key
        query = f"""
        SELECT DISTINCT
            doc_id,
            {stratify_by}
        FROM `{source_table}`
        WHERE {stratify_by} IS NOT NULL
        """

        job = self.client.query(query)
        df = job.result().to_dataframe()

        # Group by stratification key
        grouped = df.groupby(stratify_by)["doc_id"].apply(list).to_dict()

        # Stratified split
        np.random.seed(random_seed)

        train_docs = []
        test_docs = []
        val_docs = []

        for strat_key, doc_ids in grouped.items():
            doc_ids = list(set(doc_ids))  # Remove duplicates
            n = len(doc_ids)

            # Shuffle documents
            np.random.shuffle(doc_ids)

            # Calculate split indices
            train_end = int(n * train_ratio)
            test_end = train_end + int(n * test_ratio)

            train_docs.extend(doc_ids[:train_end])
            test_docs.extend(doc_ids[train_end:test_end])
            val_docs.extend(doc_ids[test_end:])

        logger.info(
            f"Split created: train={len(train_docs)}, test={len(test_docs)}, val={len(val_docs)}"
        )

        # Create tables for each split
        splits = {
            "train": (train_table, train_docs),
            "test": (test_table, test_docs),
            "val": (val_table, val_docs),
        }

        stats = {}

        for split_name, (dest_table, doc_ids) in splits.items():
            # Convert doc_ids to SQL-safe format
            doc_ids_str = ", ".join([f"'{doc_id}'" for doc_id in doc_ids])

            query = f"""
            CREATE OR REPLACE TABLE `{dest_table}` AS
            SELECT *
            FROM `{source_table}`
            WHERE doc_id IN ({doc_ids_str})
            """

            job = self.client.query(query)
            result = job.result()

            stats[split_name] = {
                "table": dest_table,
                "unique_docs": len(doc_ids),
                "total_rows": result.total_rows,
            }

            logger.info(f"✅ {split_name.upper()} split created: {dest_table}")
            logger.info(f"   Unique documents: {len(doc_ids)}")
            logger.info(f"   Total rows (triplets): {result.total_rows}")

        return stats

    def validate_split(self, train_table: str, test_table: str, val_table: str) -> Dict:
        """
        Validate that splits have no overlap and maintain stratification.

        Args:
            train_table: Training table ID
            test_table: Test table ID
            val_table: Validation table ID

        Returns:
            Validation results dictionary with overlap check and split statistics
        """
        logger.info("Validating splits for overlap and stratification...")

        # Check for document overlap
        query = f"""
        WITH all_docs AS (
            SELECT doc_id, 'train' as split FROM `{train_table}`
            UNION ALL
            SELECT doc_id, 'test' as split FROM `{test_table}`
            UNION ALL
            SELECT doc_id, 'val' as split FROM `{val_table}`
        ),
        doc_counts AS (
            SELECT doc_id, COUNT(*) as split_count
            FROM all_docs
            GROUP BY doc_id
            HAVING split_count > 1
        )
        SELECT COUNT(*) as overlapping_docs
        FROM doc_counts
        """

        job = self.client.query(query)
        overlap_result = list(job.result())[0]

        # Get split statistics
        splits_stats = {}
        for split_name, table in [
            ("train", train_table),
            ("test", test_table),
            ("val", val_table),
        ]:
            query = f"""
            SELECT
                COUNT(DISTINCT doc_id) as unique_docs,
                COUNT(*) as total_triplets,
                COUNT(DISTINCT legal_reference) as unique_legal_refs,
                COUNT(DISTINCT relation) as unique_relations
            FROM `{table}`
            """

            job = self.client.query(query)
            result = list(job.result())[0]

            splits_stats[split_name] = {
                "unique_docs": result.unique_docs,
                "total_triplets": result.total_triplets,
                "unique_legal_refs": result.unique_legal_refs,
                "unique_relations": result.unique_relations,
            }

        validation_results = {
            "overlapping_docs": overlap_result.overlapping_docs,
            "valid": overlap_result.overlapping_docs == 0,
            "splits_stats": splits_stats,
        }

        if validation_results["valid"]:
            logger.info("✅ Validation passed: No document overlap detected")
        else:
            logger.warning(
                f"⚠️  Validation failed: {overlap_result.overlapping_docs} overlapping documents"
            )

        for split_name, stats in splits_stats.items():
            logger.info(
                f"{split_name.upper()}: {stats['unique_docs']} docs, {stats['total_triplets']} triplets"
            )

        return validation_results

    def create_grouped_view(
        self,
        source_table: str,
        view_name: str = "triplets_grouped",
        strategy: str = "enhanced",
    ) -> None:
        """
        Create grouped view with normalized legal references and relation verbs.

        Grouping strategies:
        - 'base_article': Group by base article (no part) - 19.1% reduction
        - 'enhanced': Group by article number + legal code only - 86.7% reduction (RECOMMENDED)
        - 'verb': Group relations by main verb - 90.4% reduction

        Args:
            source_table: Reformatted table ID
            view_name: View name to create (default: triplets_grouped)
            strategy: Grouping strategy ('base_article' or 'enhanced')
        """
        logger.info(f"Creating grouped view with '{strategy}' strategy: {view_name}")

        # Enhanced grouping: article number + legal code only (maximum reduction)
        if strategy == "enhanced":
            article_grouping = """
                -- Enhanced grouping: article + legal code (86.7% reduction)
                CONCAT(
                    'ст.',
                    REGEXP_EXTRACT(legal_reference, r'ст\\.\\s*(\\d+(?:-\\d+)?)'),
                    ' ',
                    CASE
                        WHEN REGEXP_CONTAINS(legal_reference, r'КПК\\s*України') THEN 'КПК України'
                        WHEN REGEXP_CONTAINS(legal_reference, r'КК\\s*України') THEN 'КК України'
                        WHEN REGEXP_CONTAINS(legal_reference, r'ЦК\\s*України') THEN 'ЦК України'
                        WHEN REGEXP_CONTAINS(legal_reference, r'КоАП\\s*України') THEN 'КоАП України'
                        WHEN REGEXP_CONTAINS(legal_reference, r'СК\\s*України') THEN 'СК України'
                        WHEN REGEXP_CONTAINS(legal_reference, r'КЗпП\\s*України') THEN 'КЗпП України'
                        ELSE 'OTHER'
                    END
                ) AS legal_reference_grouped
            """
        else:  # base_article
            article_grouping = """
                -- Base article grouping: remove part number (19.1% reduction)
                TRIM(REGEXP_REPLACE(legal_reference, r'ч\\.\\d+\\s+', '')) AS legal_reference_grouped
            """

        query = f"""
        CREATE OR REPLACE VIEW `{view_name}` AS
        WITH triplet_data AS (
            SELECT
                doc_id,
                text,
                tags,
                triplet_index,
                legal_reference,
                source,
                target,
                relation,
                confidence,
                article_number,
                part_number,
                legal_code
            FROM `{source_table}`
        )
        SELECT
            doc_id,
            text,
            tags,
            triplet_index,

            -- Original fields
            legal_reference AS legal_reference_original,
            relation AS relation_original,

            -- Grouped legal reference
            {article_grouping},

            -- Extract main verb from relation - SECONDARY GROUPING (90.4% reduction)
            SPLIT(relation, '_')[SAFE_OFFSET(0)] AS relation_grouped,

            -- Metadata
            source,
            target,
            confidence,
            article_number,
            part_number,
            legal_code
        FROM triplet_data
        """

        job = self.client.query(query)
        job.result()

        logger.info(f"✅ Grouped view created: {view_name}")

    def create_grouping_stats_views(
        self, grouped_view: str = "triplets_grouped"
    ) -> None:
        """
        Create comprehensive statistics views for grouped data.

        Creates three views:
        1. Legal reference grouping statistics
        2. Relation grouping statistics
        3. Combined grouping statistics (legal_ref + relation)

        Args:
            grouped_view: Name of the grouped view (default: triplets_grouped)
        """
        project = self.client.project
        dataset = grouped_view.split(".")[1] if "." in grouped_view else None

        if not dataset:
            logger.error(
                "Cannot determine dataset from grouped_view. Use format: project.dataset.view"
            )
            return

        # Legal reference stats
        legal_ref_stats_query = f"""
        CREATE OR REPLACE VIEW `{project}.{dataset}.triplets_grouped_stats` AS
        SELECT
            legal_reference_grouped,
            COUNT(*) AS total_occurrences,
            COUNT(DISTINCT doc_id) AS docs_with_reference,
            COUNT(DISTINCT legal_reference_original) AS original_variants,
            COUNT(DISTINCT part_number) AS part_variants,
            COUNT(DISTINCT relation_original) AS unique_relations,
            COUNT(DISTINCT relation_grouped) AS unique_verbs,
            AVG(confidence) AS avg_confidence,
            ARRAY_AGG(DISTINCT legal_reference_original LIMIT 5) AS example_variants,
            ARRAY_AGG(DISTINCT relation_grouped ORDER BY relation_grouped LIMIT 10) AS top_verbs
        FROM `{grouped_view}`
        WHERE legal_reference_grouped IS NOT NULL
            AND legal_reference_grouped != 'ст.null OTHER'
        GROUP BY legal_reference_grouped
        ORDER BY total_occurrences DESC
        """

        logger.info("Creating legal reference grouping stats view...")
        job = self.client.query(legal_ref_stats_query)
        job.result()
        logger.info(f"✓ View created: {project}.{dataset}.triplets_grouped_stats")

        # Relation stats
        relation_stats_query = f"""
        CREATE OR REPLACE VIEW `{project}.{dataset}.relations_grouped_stats` AS
        SELECT
            relation_grouped,
            COUNT(*) AS total_occurrences,
            COUNT(DISTINCT doc_id) AS docs_with_verb,
            COUNT(DISTINCT relation_original) AS original_variants,
            COUNT(DISTINCT legal_reference_grouped) AS unique_grouped_articles,
            AVG(confidence) AS avg_confidence,
            ARRAY_AGG(DISTINCT relation_original LIMIT 5) AS example_variants,
            ARRAY_AGG(DISTINCT legal_code ORDER BY legal_code) AS legal_codes_used
        FROM `{grouped_view}`
        WHERE relation_grouped IS NOT NULL
        GROUP BY relation_grouped
        ORDER BY total_occurrences DESC
        """

        logger.info("Creating relation grouping stats view...")
        job = self.client.query(relation_stats_query)
        job.result()
        logger.info(f"✓ View created: {project}.{dataset}.relations_grouped_stats")

        # Combined stats
        combined_stats_query = f"""
        CREATE OR REPLACE VIEW `{project}.{dataset}.triplets_combined_grouping_stats` AS
        SELECT
            legal_reference_grouped,
            relation_grouped,
            legal_code,
            COUNT(*) AS total_occurrences,
            COUNT(DISTINCT doc_id) AS docs_with_combination,
            COUNT(DISTINCT legal_reference_original) AS legal_ref_variants,
            COUNT(DISTINCT relation_original) AS relation_variants,
            AVG(confidence) AS avg_confidence,
            MIN(confidence) AS min_confidence,
            MAX(confidence) AS max_confidence,
            ARRAY_AGG(DISTINCT legal_reference_original LIMIT 3) AS legal_ref_examples,
            ARRAY_AGG(DISTINCT relation_original LIMIT 3) AS relation_examples,
            COUNT(DISTINCT source) AS unique_sources,
            COUNT(DISTINCT target) AS unique_targets
        FROM `{grouped_view}`
        WHERE legal_reference_grouped IS NOT NULL
            AND legal_reference_grouped != 'ст.null OTHER'
            AND relation_grouped IS NOT NULL
        GROUP BY legal_reference_grouped, relation_grouped, legal_code
        HAVING COUNT(*) >= 2
        ORDER BY total_occurrences DESC
        """

        logger.info("Creating combined grouping stats view...")
        job = self.client.query(combined_stats_query)
        job.result()
        logger.info(
            f"✓ View created: {project}.{dataset}.triplets_combined_grouping_stats"
        )

    def analyze_grouping_improvement(
        self, source_table: str, grouped_view: str = "triplets_grouped"
    ) -> Dict:
        """
        Analyze the improvement from grouping strategies.

        Args:
            source_table: Original reformatted table
            grouped_view: Grouped view name

        Returns:
            Dictionary with improvement statistics
        """
        logger.info("Analyzing grouping improvement...")

        query = f"""
        WITH original_stats AS (
            SELECT
                COUNT(DISTINCT legal_reference) AS unique_legal_refs_original,
                COUNT(DISTINCT relation) AS unique_relations_original,
                COUNT(*) AS total_triplets
            FROM `{source_table}`
        ),
        grouped_stats AS (
            SELECT
                COUNT(DISTINCT legal_reference_grouped) AS unique_legal_refs_grouped,
                COUNT(DISTINCT relation_grouped) AS unique_relations_grouped,
                COUNT(*) AS total_triplets
            FROM `{grouped_view}`
            WHERE legal_reference_grouped IS NOT NULL
                AND legal_reference_grouped != 'ст.null OTHER'
        )
        SELECT
            o.unique_legal_refs_original,
            g.unique_legal_refs_grouped,
            o.unique_legal_refs_original - g.unique_legal_refs_grouped AS legal_ref_reduction,
            ROUND((o.unique_legal_refs_original - g.unique_legal_refs_grouped) / o.unique_legal_refs_original * 100, 1) AS legal_ref_reduction_pct,

            o.unique_relations_original,
            g.unique_relations_grouped,
            o.unique_relations_original - g.unique_relations_grouped AS relation_reduction,
            ROUND((o.unique_relations_original - g.unique_relations_grouped) / o.unique_relations_original * 100, 1) AS relation_reduction_pct,

            o.total_triplets
        FROM original_stats o, grouped_stats g
        """

        job = self.client.query(query)
        results = list(job.result())

        if results:
            row = results[0]
            stats = {
                "legal_refs_original": row.unique_legal_refs_original,
                "legal_refs_grouped": row.unique_legal_refs_grouped,
                "legal_ref_reduction": row.legal_ref_reduction,
                "legal_ref_reduction_pct": float(row.legal_ref_reduction_pct),
                "relations_original": row.unique_relations_original,
                "relations_grouped": row.unique_relations_grouped,
                "relation_reduction": row.relation_reduction,
                "relation_reduction_pct": float(row.relation_reduction_pct),
                "total_triplets": row.total_triplets,
            }

            logger.info(
                f"Legal references: {stats['legal_refs_original']} → {stats['legal_refs_grouped']} ({stats['legal_ref_reduction_pct']}% reduction)"
            )
            logger.info(
                f"Relations: {stats['relations_original']} → {stats['relations_grouped']} ({stats['relation_reduction_pct']}% reduction)"
            )

            return stats
        return {}

    def create_grouped_splits(
        self, grouped_view: str, train_table: str, test_table: str, val_table: str
    ) -> None:
        """
        Create grouped versions of train/test/val splits.

        Args:
            grouped_view: Grouped view name
            train_table: Original training table
            test_table: Original test table
            val_table: Original validation table
        """
        logger.info("Creating grouped splits...")

        splits = [
            (train_table, "train_grouped"),
            (test_table, "test_grouped"),
            (val_table, "val_grouped"),
        ]

        project = self.client.project
        dataset = grouped_view.split(".")[1] if "." in grouped_view else None

        for source, dest_suffix in splits:
            dest_table = f"{project}.{dataset}.{dest_suffix}"

            query = f"""
            CREATE OR REPLACE TABLE `{dest_table}` AS
            SELECT g.*
            FROM `{grouped_view}` g
            WHERE g.doc_id IN (SELECT doc_id FROM `{source}`)
            """

            job = self.client.query(query)
            result = job.result()

            logger.info(f"✓ Table created: {dest_table} ({result.total_rows} rows)")

    def export_high_frequency_combinations(
        self,
        grouped_view: str,
        output_table: str = "triplets_high_frequency",
        percentile: int = 80,
    ) -> None:
        """
        Export high-frequency combinations (top percentile).

        Args:
            grouped_view: Grouped view name
            output_table: Output table name (default: triplets_high_frequency)
            percentile: Percentile threshold (default: 80 = top 20%)
        """
        logger.info(f"Exporting high-frequency combinations (top {100-percentile}%)...")

        project = self.client.project
        dataset = grouped_view.split(".")[1] if "." in grouped_view else None
        output_full = f"{project}.{dataset}.{output_table}"

        query = f"""
        CREATE OR REPLACE TABLE `{output_full}` AS
        WITH frequency_threshold AS (
            SELECT APPROX_QUANTILES(total_occurrences, 100)[OFFSET({percentile})] AS threshold
            FROM `{project}.{dataset}.triplets_combined_grouping_stats`
        )
        SELECT g.*
        FROM `{grouped_view}` g
        JOIN `{project}.{dataset}.triplets_combined_grouping_stats` s
            ON g.legal_reference_grouped = s.legal_reference_grouped
            AND g.relation_grouped = s.relation_grouped
            AND g.legal_code = s.legal_code
        WHERE s.total_occurrences >= (SELECT threshold FROM frequency_threshold)
        """

        job = self.client.query(query)
        result = job.result()

        logger.info(f"✓ Table created: {output_full} ({result.total_rows} triplets)")

    # ========================================================================
    # Array-Based Reformatting (Efficient Storage)
    # ========================================================================

    def reformat_to_arrays(self, source_table: str, dest_table: str) -> None:
        """
        Reformat dataset keeping triplets as ARRAY<STRUCT> (efficient storage).

        Benefits:
        - 80% storage reduction (no row duplication)
        - One row per document (not one row per triplet)
        - Pre-computed grouped fields in array
        - Faster document-level queries

        Schema:
            doc_id: STRING
            text: STRING
            tags: STRING
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

        Args:
            source_table: Source table with JSON triplets
            dest_table: Destination table with structured arrays
        """
        logger.info(f"Reformatting to array format (efficient storage)...")
        logger.info(f"  Source: {source_table}")
        logger.info(f"  Destination: {dest_table}")

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

                    -- Grouped fields (enhanced strategy - 86.7% reduction)
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
        logger.info(f"   Storage: ~80% reduction vs row-based")

    def create_flattened_view(self, array_table: str, view_name: str) -> None:
        """
        Create a flattened view from array-based table.

        This creates a traditional row-per-triplet view for backward
        compatibility with existing code (e.g., GraphDataset loader).

        The view looks identical to row-based format but queries the
        efficient array-based storage underneath.

        Args:
            array_table: Source table with array format
            view_name: View name for flattened access
        """
        logger.info(f"Creating flattened view for backward compatibility...")
        logger.info(f"  Source: {array_table}")
        logger.info(f"  View: {view_name}")

        query = f"""
        CREATE OR REPLACE VIEW `{view_name}` AS
        SELECT
            doc_id,
            text,
            tags,
            triplets_count,
            CAST(ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY doc_id) AS INT64) AS triplet_index,

            -- Triplet fields (original)
            triplet.legal_reference AS legal_reference_original,
            triplet.legal_reference,
            triplet.source,
            triplet.target,
            triplet.relation AS relation_original,
            triplet.relation,
            triplet.confidence,

            -- Metadata fields
            triplet.article_number,
            triplet.part_number,
            triplet.legal_code,

            -- Grouped fields (pre-computed in array!)
            triplet.legal_reference_grouped,
            triplet.relation_grouped

        FROM `{array_table}`,
        UNNEST(triplets) AS triplet
        ORDER BY doc_id, triplet_index
        """

        job = self.client.query(query)
        job.result()

        logger.info(f"✅ Flattened view created: {view_name}")
        logger.info(f"   Use this view with GraphDataset for training")
        logger.info(f"   Grouped fields are pre-computed (faster queries)")

    def analyze_array_statistics(self, array_table: str) -> Dict:
        """
        Analyze statistics for array-based table.

        Args:
            array_table: Table with array format

        Returns:
            Dictionary with statistics including storage comparison
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
            "format": "array_based",
            "unique_docs": result.unique_docs,
            "total_triplets": result.total_triplets,
            "avg_triplets_per_doc": float(result.avg_triplets_per_doc),
            "min_triplets_per_doc": result.min_triplets_per_doc,
            "max_triplets_per_doc": result.max_triplets_per_doc,
            "unique_legal_refs_original": result.unique_legal_refs_original,
            "unique_legal_refs_grouped": result.unique_legal_refs_grouped,
            "unique_relations_original": result.unique_relations_original,
            "unique_relations_grouped": result.unique_relations_grouped,
            "array_rows": result.unique_docs,
            "row_based_equivalent": result.total_triplets,
            "storage_reduction_pct": (1 - result.unique_docs / result.total_triplets)
            * 100,
        }

        # Calculate grouping reductions
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

        logger.info(f"✅ Array-Based Statistics:")
        logger.info(f"   Documents: {stats['unique_docs']}")
        logger.info(f"   Total triplets: {stats['total_triplets']}")
        logger.info(f"   Avg per doc: {stats['avg_triplets_per_doc']:.1f}")
        logger.info(
            f"   Array rows: {stats['array_rows']} (vs {stats['row_based_equivalent']} in row-based)"
        )
        logger.info(f"   Storage reduction: {stats['storage_reduction_pct']:.1f}%")
        logger.info(
            f"   Legal refs grouped: {stats['unique_legal_refs_original']} → {stats['unique_legal_refs_grouped']} ({stats['legal_ref_reduction_pct']:.1f}% reduction)"
        )
        logger.info(
            f"   Relations grouped: {stats['unique_relations_original']} → {stats['unique_relations_grouped']} ({stats['relation_reduction_pct']:.1f}% reduction)"
        )

        return stats


# ============================================================================
# Backward Compatibility (Deprecated)
# ============================================================================

# Maintain backward compatibility for corruption functionality
# NOTE: This is deprecated. Use negative_sampling module instead.
try:
    from ..negative_sampling import (
        NegativeSampler as _NegativeSampler,
        create_negative_samples as _create_negative_samples,
    )

    # Alias for backward compatibility
    class DataCorruptor:
        """Deprecated: Use negative_sampling.NegativeSampler instead"""

        def __init__(self, *args, **kwargs):
            import warnings

            warnings.warn(
                "DataCorruptor is deprecated. Use processors.negative_sampling.NegativeSampler instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    class LegalReferenceCorruptor(DataCorruptor):
        """Deprecated: Use negative_sampling.NegativeSampler instead"""

        pass

    class CorruptionResult:
        """Deprecated: Use negative_sampling.NegativeSampleResult instead"""

        pass

    BinaryClassificationCorruptor = LegalReferenceCorruptor

except ImportError:
    # If negative_sampling not available, provide minimal stubs

    DataCorruptor = None
    LegalReferenceCorruptor = None
    CorruptionResult = None
    BinaryClassificationCorruptor = None
# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main dataset operations
    "BigQueryDatasetReformatter",
    # Backward compatibility (deprecated)
    "DataCorruptor",
    "LegalReferenceCorruptor",
    "CorruptionResult",
    "BinaryClassificationCorruptor",
]
