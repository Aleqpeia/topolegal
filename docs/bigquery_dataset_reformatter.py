#!/usr/bin/env python3
"""
BigQuery Dataset Reformatter for Legal Graph Training

This script reformats the triplets column in BigQuery by extracting JSON array
elements into separate columns for each triplet component (legal_reference,
source, target, relation). It also provides utilities for stratified splitting
based on legal reference and relation distributions.

Usage:
    # Reformat dataset
    python bigquery_dataset_reformatter.py reformat \
        --source-table project.dataset.source_table \
        --dest-table project.dataset.reformatted_table

    # Analyze statistics
    python bigquery_dataset_reformatter.py analyze \
        --table project.dataset.source_table

    # Create train/test/val splits
    python bigquery_dataset_reformatter.py split \
        --source-table project.dataset.reformatted_table \
        --train-table project.dataset.train \
        --test-table project.dataset.test \
        --val-table project.dataset.val \
        --train-ratio 0.7 \
        --test-ratio 0.2 \
        --val-ratio 0.1
"""

import argparse
import json
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BigQueryDatasetReformatter:
    """
    Reformat BigQuery legal graph datasets by extracting triplets into columns
    and providing stratified splitting utilities.
    """

    def __init__(self, project_id: Optional[str] = None, credentials_path: Optional[str] = None):
        """
        Initialize BigQuery client.

        Args:
            project_id: GCP project ID (uses default if None)
            credentials_path: Path to service account JSON key
        """
        if credentials_path:
            self.client = bigquery.Client.from_service_account_json(
                credentials_path,
                project=project_id
            )
        else:
            self.client = bigquery.Client(project=project_id)

        logger.info(f"Initialized BigQuery client for project: {self.client.project}")

    def analyze_triplet_statistics(self, source_table: str) -> Dict:
        """
        Analyze triplet statistics for stratified splitting.

        Args:
            source_table: Source table ID (format: project.dataset.table)

        Returns:
            Dictionary with statistics
        """
        logger.info(f"Analyzing triplet statistics for {source_table}")

        # Query to extract triplet statistics
        query = f"""
        WITH triplet_data AS (
            SELECT
                doc_id,
                triplets,
                triplets_count,
                JSON_EXTRACT_ARRAY(triplets) AS triplet_array
            FROM `{source_table}`
            WHERE triplets IS NOT NULL
                AND triplets != '[]'
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

        # Aggregate statistics
        legal_ref_stats = results.groupby('legal_reference').agg({
            'docs_with_legal_ref': 'first',
            'total_occurrences_legal_ref': 'first'
        }).to_dict('index')

        relation_stats = results.groupby('relation').agg({
            'docs_with_relation': 'first',
            'total_occurrences_relation': 'first'
        }).to_dict('index')

        stats = {
            'total_triplets': len(results),
            'unique_legal_references': len(legal_ref_stats),
            'unique_relations': len(relation_stats),
            'legal_reference_distribution': legal_ref_stats,
            'relation_distribution': relation_stats,
            'top_legal_references': results.nlargest(10, 'total_occurrences_legal_ref')[
                ['legal_reference', 'total_occurrences_legal_ref']
            ].to_dict('records'),
            'top_relations': results.nlargest(10, 'total_occurrences_relation')[
                ['relation', 'total_occurrences_relation']
            ].to_dict('records')
        }

        logger.info(f"Statistics collected:")
        logger.info(f"  - Total triplets: {stats['total_triplets']}")
        logger.info(f"  - Unique legal references: {stats['unique_legal_references']}")
        logger.info(f"  - Unique relations: {stats['unique_relations']}")

        return stats

    def reformat_dataset(
        self,
        source_table: str,
        dest_table: str,
        create_if_not_exists: bool = True
    ) -> None:
        """
        Reformat dataset by extracting triplet JSON into separate columns.

        Args:
            source_table: Source table ID
            dest_table: Destination table ID
            create_if_not_exists: Create destination table if it doesn't exist
        """
        logger.info(f"Reformatting dataset from {source_table} to {dest_table}")

        # SQL to extract triplets into columns
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
                AND triplets != '[]'
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
            -- Add metadata columns for analysis
            REGEXP_EXTRACT(legal_reference, r'ст\\.\\s*(\\d+)') AS article_number,
            REGEXP_EXTRACT(legal_reference, r'ч\\.\\s*(\\d+)') AS part_number,
            REGEXP_EXTRACT(legal_reference, r'(КК|КПК|ЦК|КоАП|СК|КЗпП)\\s*України') AS legal_code,
            CURRENT_TIMESTAMP() AS reformatted_timestamp
        FROM exploded_triplets
        ORDER BY doc_id, triplet_index
        """

        job = self.client.query(query)
        result = job.result()

        logger.info(f"✅ Dataset reformatted successfully")
        logger.info(f"   Destination table: {dest_table}")
        logger.info(f"   Total rows: {result.total_rows}")

    def get_stratification_groups(
        self,
        source_table: str,
        group_by: str = "legal_reference",
        min_samples: int = 10
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
            groups[row[group_by]].append(row['doc_id'])

        # Filter groups with minimum samples
        filtered_groups = {
            k: list(set(v)) for k, v in groups.items()
            if len(set(v)) >= min_samples
        }

        logger.info(f"Found {len(filtered_groups)} groups with >= {min_samples} samples")

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
        random_seed: int = 42
    ) -> Dict[str, int]:
        """
        Create stratified train/test/validation splits.

        Args:
            source_table: Reformatted source table
            train_table: Training set destination
            test_table: Test set destination
            val_table: Validation set destination
            train_ratio: Training set ratio
            test_ratio: Test set ratio
            val_ratio: Validation set ratio
            stratify_by: Column to stratify by
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with split statistics
        """
        assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1.0"

        logger.info(f"Creating stratified splits: train={train_ratio}, test={test_ratio}, val={val_ratio}")

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
        grouped = df.groupby(stratify_by)['doc_id'].apply(list).to_dict()

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

        logger.info(f"Split created: train={len(train_docs)}, test={len(test_docs)}, val={len(val_docs)}")

        # Create tables for each split
        splits = {
            'train': (train_table, train_docs),
            'test': (test_table, test_docs),
            'val': (val_table, val_docs)
        }

        stats = {}

        for split_name, (dest_table, doc_ids) in splits.items():
            # Convert doc_ids to SQL-safe format
            doc_ids_str = ', '.join([f"'{doc_id}'" for doc_id in doc_ids])

            query = f"""
            CREATE OR REPLACE TABLE `{dest_table}` AS
            SELECT *
            FROM `{source_table}`
            WHERE doc_id IN ({doc_ids_str})
            """

            job = self.client.query(query)
            result = job.result()

            stats[split_name] = {
                'table': dest_table,
                'unique_docs': len(doc_ids),
                'total_rows': result.total_rows
            }

            logger.info(f"✅ {split_name.upper()} split created: {dest_table}")
            logger.info(f"   Unique documents: {len(doc_ids)}")
            logger.info(f"   Total rows (triplets): {result.total_rows}")

        return stats

    def validate_split(
        self,
        train_table: str,
        test_table: str,
        val_table: str
    ) -> Dict:
        """
        Validate that splits have no overlap and maintain stratification.

        Args:
            train_table: Training table ID
            test_table: Test table ID
            val_table: Validation table ID

        Returns:
            Validation results dictionary
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
        for split_name, table in [('train', train_table), ('test', test_table), ('val', val_table)]:
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
                'unique_docs': result.unique_docs,
                'total_triplets': result.total_triplets,
                'unique_legal_refs': result.unique_legal_refs,
                'unique_relations': result.unique_relations
            }

        validation_results = {
            'overlapping_docs': overlap_result.overlapping_docs,
            'valid': overlap_result.overlapping_docs == 0,
            'splits_stats': splits_stats
        }

        if validation_results['valid']:
            logger.info("✅ Validation passed: No document overlap detected")
        else:
            logger.warning(f"⚠️  Validation failed: {overlap_result.overlapping_docs} overlapping documents")

        for split_name, stats in splits_stats.items():
            logger.info(f"{split_name.upper()}: {stats['unique_docs']} docs, {stats['total_triplets']} triplets")

        return validation_results

    def create_grouping_view(
        self,
        source_table: str,
        view_name: str,
        group_by: List[str] = ["legal_reference", "relation"]
    ) -> None:
        """
        Create a BigQuery view with aggregated grouping statistics.

        Args:
            source_table: Reformatted table ID
            view_name: View name to create
            group_by: Columns to group by
        """
        logger.info(f"Creating grouping view: {view_name}")

        group_by_str = ', '.join(group_by)

        query = f"""
        CREATE OR REPLACE VIEW `{view_name}` AS
        SELECT
            {group_by_str},
            COUNT(DISTINCT doc_id) AS unique_docs,
            COUNT(*) AS total_triplets,
            AVG(confidence) AS avg_confidence,
            MIN(confidence) AS min_confidence,
            MAX(confidence) AS max_confidence,
            COUNT(DISTINCT legal_code) AS legal_codes_count,
            STRING_AGG(DISTINCT legal_code, ', ') AS legal_codes,
            CURRENT_TIMESTAMP() AS view_created_at
        FROM `{source_table}`
        GROUP BY {group_by_str}
        ORDER BY total_triplets DESC
        """

        job = self.client.query(query)
        job.result()

        logger.info(f"✅ Grouping view created: {view_name}")


def main():
    parser = argparse.ArgumentParser(
        description="BigQuery Dataset Reformatter for Legal Graph Training"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Reformat command
    reformat_parser = subparsers.add_parser('reformat', help='Reformat dataset')
    reformat_parser.add_argument('--source-table', required=True, help='Source table ID')
    reformat_parser.add_argument('--dest-table', required=True, help='Destination table ID')
    reformat_parser.add_argument('--project-id', help='GCP project ID')
    reformat_parser.add_argument('--credentials', help='Path to service account JSON')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze triplet statistics')
    analyze_parser.add_argument('--table', required=True, help='Table ID to analyze')
    analyze_parser.add_argument('--project-id', help='GCP project ID')
    analyze_parser.add_argument('--credentials', help='Path to service account JSON')
    analyze_parser.add_argument('--output', help='Output JSON file for statistics')

    # Split command
    split_parser = subparsers.add_parser('split', help='Create train/test/val splits')
    split_parser.add_argument('--source-table', required=True, help='Reformatted source table')
    split_parser.add_argument('--train-table', required=True, help='Training table destination')
    split_parser.add_argument('--test-table', required=True, help='Test table destination')
    split_parser.add_argument('--val-table', required=True, help='Validation table destination')
    split_parser.add_argument('--train-ratio', type=float, default=0.7, help='Training ratio')
    split_parser.add_argument('--test-ratio', type=float, default=0.2, help='Test ratio')
    split_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    split_parser.add_argument('--stratify-by', default='legal_reference',
                             help='Column to stratify by (legal_reference, relation, legal_code)')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    split_parser.add_argument('--project-id', help='GCP project ID')
    split_parser.add_argument('--credentials', help='Path to service account JSON')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate splits')
    validate_parser.add_argument('--train-table', required=True, help='Training table')
    validate_parser.add_argument('--test-table', required=True, help='Test table')
    validate_parser.add_argument('--val-table', required=True, help='Validation table')
    validate_parser.add_argument('--project-id', help='GCP project ID')
    validate_parser.add_argument('--credentials', help='Path to service account JSON')

    # Group view command
    group_parser = subparsers.add_parser('create-view', help='Create grouping view')
    group_parser.add_argument('--source-table', required=True, help='Reformatted table')
    group_parser.add_argument('--view-name', required=True, help='View name to create')
    group_parser.add_argument('--group-by', nargs='+', default=['legal_reference', 'relation'],
                             help='Columns to group by')
    group_parser.add_argument('--project-id', help='GCP project ID')
    group_parser.add_argument('--credentials', help='Path to service account JSON')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize reformatter
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id,
        credentials_path=args.credentials
    )

    # Execute command
    if args.command == 'reformat':
        reformatter.reformat_dataset(
            source_table=args.source_table,
            dest_table=args.dest_table
        )

    elif args.command == 'analyze':
        stats = reformatter.analyze_triplet_statistics(args.table)

        print("\n" + "="*60)
        print("TRIPLET STATISTICS")
        print("="*60)
        print(f"Total triplets: {stats['total_triplets']}")
        print(f"Unique legal references: {stats['unique_legal_references']}")
        print(f"Unique relations: {stats['unique_relations']}")

        print("\nTop 10 Legal References:")
        for item in stats['top_legal_references']:
            print(f"  {item['legal_reference']}: {item['total_occurrences_legal_ref']} occurrences")

        print("\nTop 10 Relations:")
        for item in stats['top_relations']:
            print(f"  {item['relation']}: {item['total_occurrences_relation']} occurrences")

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Statistics saved to {args.output}")

    elif args.command == 'split':
        stats = reformatter.stratified_split(
            source_table=args.source_table,
            train_table=args.train_table,
            test_table=args.test_table,
            val_table=args.val_table,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            stratify_by=args.stratify_by,
            random_seed=args.seed
        )

        print("\n" + "="*60)
        print("SPLIT STATISTICS")
        print("="*60)
        for split_name, split_stats in stats.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Table: {split_stats['table']}")
            print(f"  Unique documents: {split_stats['unique_docs']}")
            print(f"  Total rows: {split_stats['total_rows']}")

    elif args.command == 'validate':
        results = reformatter.validate_split(
            train_table=args.train_table,
            test_table=args.test_table,
            val_table=args.val_table
        )

        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        print(f"Valid: {'✅ YES' if results['valid'] else '❌ NO'}")
        print(f"Overlapping documents: {results['overlapping_docs']}")

        print("\nSplit Statistics:")
        for split_name, stats in results['splits_stats'].items():
            print(f"\n{split_name.upper()}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    elif args.command == 'create-view':
        reformatter.create_grouping_view(
            source_table=args.source_table,
            view_name=args.view_name,
            group_by=args.group_by
        )


if __name__ == "__main__":
    main()
