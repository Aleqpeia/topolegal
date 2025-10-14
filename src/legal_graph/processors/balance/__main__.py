#!/usr/bin/env python3
"""
BigQuery Dataset Operations CLI

This module provides command-line tools for BigQuery dataset operations:
- Dataset statistics and analysis
- Triplet reformatting (extracting JSON to columnar format)
- Stratified train/test/val splitting
- Split validation and distribution analysis

Aligned with DATASET_REFORMATTING_GUIDE.md workflow.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import BigQuery dataset reformatter
from . import BigQueryDatasetReformatter


def build_table_id(args, table_name: str) -> str:
    """
    Build full table ID from args.

    If table_name is already a full ID (contains '.'), return as-is.
    Otherwise, use --project-id and --dataset to construct full ID.

    Args:
        args: Parsed command-line arguments
        table_name: Table name or full table ID

    Returns:
        Full table ID in format project.dataset.table
    """
    # If already a full table ID, return as-is
    if "." in table_name and table_name.count(".") >= 2:
        return table_name

    # Otherwise, construct from global args
    if args.project_id and args.dataset:
        return f"{args.project_id}.{args.dataset}.{table_name}"

    # If no global args, assume it's already correct
    return table_name


def analyze_command(args):
    """Analyze triplet statistics for a BigQuery table"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    stats = reformatter.analyze_triplet_statistics(args.table)

    # Print statistics
    print("\n" + "=" * 60)
    print("TRIPLET STATISTICS")
    print("=" * 60)
    print(f"Total triplets: {stats['total_triplets']}")
    print(f"Unique legal references: {stats['unique_legal_references']}")
    print(f"Unique relations: {stats['unique_relations']}")

    print("\nTop 10 Legal References:")
    for item in stats["legal_reference_distribution"].items():
        print(f"  {item[0]}: {item[1]} occurrences")

    print("\nRelations:")
    for item in stats["relation_distribution"].items():
        print(f"  {item[0]}: {item[1]} occurrences")

    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to {args.output}")


def reformat_command(args):
    """Reformat dataset by extracting triplet JSON into separate columns"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    reformatter.reformat_dataset(
        source_table=args.source_table, dest_table=args.dest_table
    )


def split_command(args):
    """Create stratified train/test/val splits"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    stats = reformatter.stratified_split(
        source_table=args.source_table,
        train_table=args.train_table,
        test_table=args.test_table,
        val_table=args.val_table,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        stratify_by=args.stratify_by,
        random_seed=args.seed,
    )

    # Print split statistics
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Table: {split_stats['table']}")
        print(f"  Unique documents: {split_stats['unique_docs']}")
        print(f"  Total rows: {split_stats['total_rows']}")


def validate_command(args):
    """Validate that splits have no overlap and maintain stratification"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    results = reformatter.validate_split(
        train_table=args.train_table,
        test_table=args.test_table,
        val_table=args.val_table,
    )

    # Print validation results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Valid: {'✅ YES' if results['valid'] else '❌ NO'}")
    print(f"Overlapping documents: {results['overlapping_docs']}")

    print("\nSplit Statistics:")
    for split_name, stats in results["splits_stats"].items():
        print(f"\n{split_name.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def create_grouped_view_command(args):
    """Create grouped view with normalized legal references and relations"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    reformatter.create_grouped_view(
        source_table=args.source_table, view_name=args.view_name, strategy=args.strategy
    )


def create_grouping_stats_command(args):
    """Create grouping statistics views"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    reformatter.create_grouping_stats_views(grouped_view=args.grouped_view)


def analyze_grouping_command(args):
    """Analyze grouping improvement"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    stats = reformatter.analyze_grouping_improvement(
        source_table=args.source_table, grouped_view=args.grouped_view
    )

    # Print analysis results
    print("\n" + "=" * 60)
    print("GROUPING IMPROVEMENT ANALYSIS")
    print("=" * 60)
    print(f"\nLegal References:")
    print(f"  Original: {stats['legal_refs_original']}")
    print(f"  Grouped: {stats['legal_refs_grouped']}")
    print(
        f"  Reduction: {stats['legal_ref_reduction']} ({stats['legal_ref_reduction_pct']}%)"
    )
    print(f"\nRelations:")
    print(f"  Original: {stats['relations_original']}")
    print(f"  Grouped: {stats['relations_grouped']}")
    print(
        f"  Reduction: {stats['relation_reduction']} ({stats['relation_reduction_pct']}%)"
    )
    print(f"\nTotal triplets: {stats['total_triplets']}")


def create_grouped_splits_command(args):
    """Create grouped train/test/val splits"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    reformatter.create_grouped_splits(
        grouped_view=args.grouped_view,
        train_table=args.train_table,
        test_table=args.test_table,
        val_table=args.val_table,
    )


def export_high_frequency_command(args):
    """Export high-frequency combinations"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    reformatter.export_high_frequency_combinations(
        grouped_view=args.grouped_view,
        output_table=args.output_table,
        percentile=args.percentile,
    )


# ============================================================================
# Array-Based Commands
# ============================================================================


def array_reformat_command(args):
    """Reformat to array-based format (80% storage reduction)"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    # Build table IDs
    source = build_table_id(args, args.source_table)
    dest = build_table_id(args, args.dest_table)

    reformatter.reformat_to_arrays(source_table=source, dest_table=dest)


def array_flatten_command(args):
    """Create flattened view from array table"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    # Build table IDs
    array_table = build_table_id(args, args.array_table)
    view_name = build_table_id(args, args.view_name)

    reformatter.create_flattened_view(array_table=array_table, view_name=view_name)


def array_analyze_command(args):
    """Analyze array-based table statistics"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    # Build table ID
    array_table = build_table_id(args, args.array_table)

    stats = reformatter.analyze_array_statistics(array_table)

    # Print detailed stats
    print("\n" + "=" * 60)
    print("ARRAY-BASED TABLE STATISTICS")
    print("=" * 60)
    print(f"\nStorage Efficiency:")
    print(f"  Array rows: {stats['array_rows']}")
    print(f"  Row-based equivalent: {stats['row_based_equivalent']}")
    print(f"  Storage reduction: {stats['storage_reduction_pct']:.1f}%")
    print(f"\nTriplet Statistics:")
    print(f"  Total triplets: {stats['total_triplets']}")
    print(f"  Avg per doc: {stats['avg_triplets_per_doc']:.1f}")
    print(f"  Min per doc: {stats['min_triplets_per_doc']}")
    print(f"  Max per doc: {stats['max_triplets_per_doc']}")
    print(f"\nGrouping (Pre-computed in Array):")
    print(
        f"  Legal refs: {stats['unique_legal_refs_original']} → {stats['unique_legal_refs_grouped']} ({stats['legal_ref_reduction_pct']:.1f}% reduction)"
    )
    print(
        f"  Relations: {stats['unique_relations_original']} → {stats['unique_relations_grouped']} ({stats['relation_reduction_pct']:.1f}% reduction)"
    )


def array_workflow_command(args):
    """Run complete array-based workflow"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    # Build table IDs
    source = build_table_id(args, args.source_table)
    array_table = build_table_id(args, args.array_table)
    flattened_view = build_table_id(args, args.flattened_view)

    logger.info("Starting array-based reformatting workflow...")

    # Step 1: Reformat to arrays
    logger.info("Step 1: Reformatting to array format...")
    reformatter.reformat_to_arrays(source_table=source, dest_table=array_table)

    # Step 2: Create flattened view
    logger.info("Step 2: Creating flattened view...")
    reformatter.create_flattened_view(array_table=array_table, view_name=flattened_view)

    # Step 3: Analyze statistics
    logger.info("Step 3: Analyzing statistics...")
    stats = reformatter.analyze_array_statistics(array_table)

    # Summary
    print("\n" + "=" * 60)
    print("ARRAY-BASED WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"✅ Array table created: {array_table}")
    print(f"   Storage reduction: {stats['storage_reduction_pct']:.1f}%")
    print(
        f"   {stats['array_rows']} rows (vs {stats['row_based_equivalent']} in row-based)"
    )
    print(f"✅ Flattened view created: {flattened_view}")
    print(f"   Use this view with GraphDataset for training")
    print(f"✅ Grouping pre-computed:")
    print(f"   Legal refs: {stats['legal_ref_reduction_pct']:.1f}% reduction")
    print(f"   Relations: {stats['relation_reduction_pct']:.1f}% reduction")


def workflow_command(args):
    """Run complete dataset reformatting workflow"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    logger.info("Starting complete dataset reformatting workflow...")

    # Step 1: Analyze source data
    logger.info("Step 1: Analyzing source data...")
    stats = reformatter.analyze_triplet_statistics(args.source_table)
    logger.info(f"  - Total triplets: {stats['total_triplets']}")
    logger.info(f"  - Unique legal references: {stats['unique_legal_references']}")

    # Step 2: Reformat dataset
    logger.info("Step 2: Reformatting dataset...")
    reformatter.reformat_dataset(
        source_table=args.source_table, dest_table=args.reformatted_table
    )

    # Step 3: Create stratified splits
    logger.info("Step 3: Creating stratified splits...")
    split_stats = reformatter.stratified_split(
        source_table=args.reformatted_table,
        train_table=args.train_table,
        test_table=args.test_table,
        val_table=args.val_table,
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1,
        stratify_by=args.stratify_by,
        random_seed=42,
    )

    # Step 4: Validate splits
    logger.info("Step 4: Validating splits...")
    validation = reformatter.validate_split(
        train_table=args.train_table,
        test_table=args.test_table,
        val_table=args.val_table,
    )

    # Step 5: Create basic grouping view
    logger.info("Step 5: Creating basic grouping view...")
    reformatter.create_grouped_view(
        source_table=args.reformatted_table,
        view_name=args.grouping_view,
        strategy="base_article",
    )

    # Summary
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"✅ Source analyzed: {stats['total_triplets']} triplets")
    print(f"✅ Dataset reformatted: {args.reformatted_table}")
    print(f"✅ Splits created: train={split_stats['train']['unique_docs']} docs")
    print(f"✅ Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    print(f"✅ Grouping view created: {args.grouping_view}")


def grouping_workflow_command(args):
    """Run complete grouping workflow"""
    reformatter = BigQueryDatasetReformatter(
        project_id=args.project_id, credentials_path=args.credentials
    )

    logger.info("Starting complete grouping workflow...")

    # Step 1: Create grouped view
    logger.info(f"Step 1: Creating grouped view with '{args.strategy}' strategy...")
    reformatter.create_grouped_view(
        source_table=args.source_table, view_name=args.view_name, strategy=args.strategy
    )

    # Step 2: Create grouping statistics views
    logger.info("Step 2: Creating grouping statistics views...")
    full_view_name = f"{reformatter.client.project}.{args.source_table.split('.')[1]}.{args.view_name}"
    reformatter.create_grouping_stats_views(grouped_view=full_view_name)

    # Step 3: Analyze grouping improvement
    logger.info("Step 3: Analyzing grouping improvement...")
    stats = reformatter.analyze_grouping_improvement(
        source_table=args.source_table, grouped_view=full_view_name
    )

    # Step 4: Create grouped splits
    logger.info("Step 4: Creating grouped train/test/val splits...")
    reformatter.create_grouped_splits(
        grouped_view=full_view_name,
        train_table=args.train_table,
        test_table=args.test_table,
        val_table=args.val_table,
    )

    # Step 5: Export high-frequency combinations
    logger.info("Step 5: Exporting high-frequency combinations...")
    reformatter.export_high_frequency_combinations(
        grouped_view=full_view_name,
        output_table="triplets_high_frequency",
        percentile=80,
    )

    # Summary
    print("\n" + "=" * 60)
    print("GROUPING WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"✅ Grouped view created: {full_view_name}")
    print(f"   Strategy: {args.strategy}")
    print(
        f"   Legal refs: {stats['legal_refs_original']} → {stats['legal_refs_grouped']} ({stats['legal_ref_reduction_pct']}% reduction)"
    )
    print(
        f"   Relations: {stats['relations_original']} → {stats['relations_grouped']} ({stats['relation_reduction_pct']}% reduction)"
    )
    print(f"✅ Statistics views created")
    print(f"✅ Grouped splits created (train_grouped, test_grouped, val_grouped)")
    print(f"✅ High-frequency combinations exported")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="BigQuery Dataset Operations for Legal Graph Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset
  python -m legal_graph.processors.balance analyze project.dataset.table

  # Create grouped view with enhanced strategy (86.7% reduction)
  python -m legal_graph.processors.balance create-grouped-view \\
      --project-id project-id --dataset dataset table_name --strategy enhanced

  # Run complete grouping workflow
  python -m legal_graph.processors.balance grouping-workflow \\
      --project-id project-id --credentials key.json \\
      project.dataset.reformatted project.dataset.train \\
      project.dataset.test project.dataset.val
        """,
    )

    # Global arguments
    parser.add_argument(
        "--project-id", help="GCP project ID (alternative to full table IDs)"
    )
    parser.add_argument(
        "--dataset",
        help="BigQuery dataset name (used with --project-id for shorter table references)",
    )
    parser.add_argument("--credentials", help="Path to service account JSON")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze triplet statistics")
    analyze_parser.add_argument(
        "table", help="Table ID to analyze (project.dataset.table)"
    )
    analyze_parser.add_argument("--output", help="Output JSON file for statistics")
    analyze_parser.set_defaults(func=analyze_command)

    # Reformat command
    reformat_parser = subparsers.add_parser("reformat", help="Reformat dataset")
    reformat_parser.add_argument("source_table", help="Source table ID")
    reformat_parser.add_argument("dest_table", help="Destination table ID")
    reformat_parser.set_defaults(func=reformat_command)

    # Split command
    split_parser = subparsers.add_parser("split", help="Create train/test/val splits")
    split_parser.add_argument("source_table", help="Reformatted source table")
    split_parser.add_argument("train_table", help="Training table destination")
    split_parser.add_argument("test_table", help="Test table destination")
    split_parser.add_argument("val_table", help="Validation table destination")
    split_parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training ratio (default: 0.7)"
    )
    split_parser.add_argument(
        "--test-ratio", type=float, default=0.2, help="Test ratio (default: 0.2)"
    )
    split_parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)"
    )
    split_parser.add_argument(
        "--stratify-by",
        default="legal_reference",
        help="Column to stratify by (default: legal_reference)",
    )
    split_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    split_parser.set_defaults(func=split_command)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate splits")
    validate_parser.add_argument("train_table", help="Training table")
    validate_parser.add_argument("test_table", help="Test table")
    validate_parser.add_argument("val_table", help="Validation table")
    validate_parser.set_defaults(func=validate_command)

    # Create grouped view command
    grouped_view_parser = subparsers.add_parser(
        "create-grouped-view", help="Create grouped view with normalized references"
    )
    grouped_view_parser.add_argument("source_table", help="Reformatted table")
    grouped_view_parser.add_argument(
        "--view-name",
        default="triplets_grouped",
        help="View name (default: triplets_grouped)",
    )
    grouped_view_parser.add_argument(
        "--strategy",
        choices=["base_article", "enhanced"],
        default="enhanced",
        help="Grouping strategy (default: enhanced = 86.7%% reduction)",
    )
    grouped_view_parser.set_defaults(func=create_grouped_view_command)

    # Create grouping stats views command
    grouping_stats_parser = subparsers.add_parser(
        "create-grouping-stats", help="Create grouping statistics views"
    )
    grouping_stats_parser.add_argument(
        "--grouped-view",
        default="triplets_grouped",
        help="Grouped view name (default: triplets_grouped)",
    )
    grouping_stats_parser.set_defaults(func=create_grouping_stats_command)

    # Analyze grouping improvement command
    analyze_grouping_parser = subparsers.add_parser(
        "analyze-grouping", help="Analyze grouping improvement"
    )
    analyze_grouping_parser.add_argument(
        "source_table", help="Original reformatted table"
    )
    analyze_grouping_parser.add_argument(
        "--grouped-view",
        default="triplets_grouped",
        help="Grouped view name (default: triplets_grouped)",
    )
    analyze_grouping_parser.set_defaults(func=analyze_grouping_command)

    # Create grouped splits command
    grouped_splits_parser = subparsers.add_parser(
        "create-grouped-splits", help="Create grouped train/test/val splits"
    )
    grouped_splits_parser.add_argument(
        "--grouped-view", default="triplets_grouped", help="Grouped view name"
    )
    grouped_splits_parser.add_argument("train_table", help="Original training table")
    grouped_splits_parser.add_argument("test_table", help="Original test table")
    grouped_splits_parser.add_argument("val_table", help="Original validation table")
    grouped_splits_parser.set_defaults(func=create_grouped_splits_command)

    # Export high-frequency combinations command
    export_parser = subparsers.add_parser(
        "export-high-frequency", help="Export high-frequency combinations"
    )
    export_parser.add_argument(
        "--grouped-view", default="triplets_grouped", help="Grouped view name"
    )
    export_parser.add_argument(
        "--output-table",
        default="triplets_high_frequency",
        help="Output table name (default: triplets_high_frequency)",
    )
    export_parser.add_argument(
        "--percentile",
        type=int,
        default=80,
        help="Percentile threshold (default: 80 = top 20%%)",
    )
    export_parser.set_defaults(func=export_high_frequency_command)

    # Workflow command (complete pipeline)
    workflow_parser = subparsers.add_parser(
        "workflow", help="Run complete reformatting workflow"
    )
    workflow_parser.add_argument("source_table", help="Source table ID")
    workflow_parser.add_argument("reformatted_table", help="Reformatted table ID")
    workflow_parser.add_argument("train_table", help="Training table destination")
    workflow_parser.add_argument("test_table", help="Test table destination")
    workflow_parser.add_argument("val_table", help="Validation table destination")
    workflow_parser.add_argument("grouping_view", help="Grouping view name")
    workflow_parser.add_argument(
        "--stratify-by",
        default="legal_reference",
        help="Column to stratify by (default: legal_reference)",
    )
    workflow_parser.set_defaults(func=workflow_command)

    # Grouping workflow command (complete grouping pipeline)
    grouping_workflow_parser = subparsers.add_parser(
        "grouping-workflow", help="Run complete grouping workflow"
    )
    grouping_workflow_parser.add_argument(
        "source_table", help="Reformatted source table"
    )
    grouping_workflow_parser.add_argument("train_table", help="Original training table")
    grouping_workflow_parser.add_argument("test_table", help="Original test table")
    grouping_workflow_parser.add_argument("val_table", help="Original validation table")
    grouping_workflow_parser.add_argument(
        "--strategy",
        choices=["base_article", "enhanced"],
        default="enhanced",
        help="Grouping strategy (default: enhanced)",
    )
    grouping_workflow_parser.add_argument(
        "--view-name",
        default="triplets_grouped",
        help="View name (default: triplets_grouped)",
    )
    grouping_workflow_parser.set_defaults(func=grouping_workflow_command)

    # Array-based reformatting commands
    array_reformat_parser = subparsers.add_parser(
        "array-reformat", help="Reformat to array format (80%% storage reduction)"
    )
    array_reformat_parser.add_argument("source_table", help="Source table")
    array_reformat_parser.add_argument("dest_table", help="Destination array table")
    array_reformat_parser.set_defaults(func=array_reformat_command)

    array_flatten_parser = subparsers.add_parser(
        "array-flatten", help="Create flattened view from array table"
    )
    array_flatten_parser.add_argument("array_table", help="Array table")
    array_flatten_parser.add_argument(
        "view_name", help="View name for flattened access"
    )
    array_flatten_parser.set_defaults(func=array_flatten_command)

    array_analyze_parser = subparsers.add_parser(
        "array-analyze", help="Analyze array-based table statistics"
    )
    array_analyze_parser.add_argument("array_table", help="Array table to analyze")
    array_analyze_parser.set_defaults(func=array_analyze_command)

    array_workflow_parser = subparsers.add_parser(
        "array-workflow", help="Complete array-based workflow (recommended)"
    )
    array_workflow_parser.add_argument("source_table", help="Source table")
    array_workflow_parser.add_argument(
        "--array-table",
        default="reformatted_array",
        help="Array table name (default: reformatted_array)",
    )
    array_workflow_parser.add_argument(
        "--flattened-view",
        default="reformatted_flattened",
        help="Flattened view name (default: reformatted_flattened)",
    )
    array_workflow_parser.set_defaults(func=array_workflow_command)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
