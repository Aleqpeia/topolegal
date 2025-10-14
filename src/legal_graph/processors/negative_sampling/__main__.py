#!/usr/bin/env python3
"""
Negative Sampling CLI Tool

Command-line interface for creating negative samples from legal documents
by modifying legal reference components.

Usage:
    python -m legal_graph.processors.negative_sampling sample-csv input.csv output.csv --strategy mixed
    python -m legal_graph.processors.negative_sampling sample-bigquery input_table output_table --strategy aggressive
"""

import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
import argparse
from datetime import datetime
import os
import sys

# Import sampling functionality
from . import (
    NegativeSampler,
    create_negative_samples,
    create_strategy,
    DEFAULT_STRATEGY,
    CONSERVATIVE_STRATEGY,
    AGGRESSIVE_STRATEGY,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def sample_csv_data(
    input_csv: str,
    output_csv: str,
    strategy: str = "mixed",
    sampling_probability: float = 0.5,
    label_column: str = "is_valid",
):
    """
    Create negative samples from CSV file.

    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path
        strategy: Sampling strategy name
        sampling_probability: Probability of sampling each document
        label_column: Column name for validity labels
    """
    try:
        # Load data
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} documents from {input_csv}")

        # Convert to dataset format
        dataset = []
        for _, row in df.iterrows():
            try:
                # Parse entities and triplets if they exist
                entities = []
                triplets = []

                # Handle entities/tags
                if "tags" in row:
                    tags_value = row.tags
                    if (
                        tags_value is not None
                        and str(tags_value).strip()
                        and str(tags_value) != "nan"
                    ):
                        try:
                            entities = json.loads(str(tags_value))
                            if not isinstance(entities, list):
                                entities = []
                        except (json.JSONDecodeError, ValueError):
                            entities = []

                # Handle triplets
                if "triplets" in row:
                    triplets_value = row.triplets
                    if (
                        triplets_value is not None
                        and str(triplets_value).strip()
                        and str(triplets_value) != "nan"
                    ):
                        try:
                            triplets = json.loads(str(triplets_value))
                            if not isinstance(triplets, list):
                                triplets = []
                        except (json.JSONDecodeError, ValueError):
                            triplets = []

                # Skip if no triplets (we only want documents with triplets)
                if not triplets:
                    continue

                dataset.append(
                    {
                        "doc_id": str(row.get("doc_id", f"doc_{len(dataset)}")),
                        "text": str(row.text),
                        "entities": entities,
                        "triplets": triplets,
                        "triplets_count": len(triplets),
                    }
                )
            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue

        logger.info(f"Filtered to {len(dataset)} documents with non-empty triplets")

        if not dataset:
            logger.warning("⚠️  No documents with triplets found in dataset!")
            return

        # Initialize sampler and create negative samples
        sampler = NegativeSampler(
            sampling_probability=sampling_probability, strategy=strategy
        )

        logger.info(
            f"Creating negative samples with strategy: {sampler.strategy.get_strategy_name()}"
        )
        negative_dataset = sampler.create_negative_samples(
            dataset, label_column=label_column
        )

        # Convert back to DataFrame and save
        output_data = []
        for item in negative_dataset:
            output_data.append(
                {
                    "doc_id": item["doc_id"],
                    "text": item["text"],
                    "tags": json.dumps(item.get("entities", []), ensure_ascii=False),
                    "triplets": json.dumps(
                        item.get("triplets", []), ensure_ascii=False
                    ),
                    "triplets_count": item.get("triplets_count", 0),
                    label_column: item[label_column],
                }
            )

        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_csv, index=False)

        # Show statistics
        stats = sampler.get_sampling_stats()
        logger.info(f"✅ Negative sampling complete! Saved to {output_csv}")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Total samples: {len(output_df)}")
        logger.info(f"   Valid samples: {len(output_df[output_df[label_column]])}")
        logger.info(f"   Negative samples: {len(output_df[~output_df[label_column]])}")
        logger.info(f"   Sampling stats: {stats}")

        if stats.get("modifications_by_type"):
            logger.info(f"📝 Modifications by type:")
            for mod_type, count in stats["modifications_by_type"].items():
                logger.info(f"   {mod_type}: {count}")

    except Exception as e:
        logger.error(f"❌ Error processing CSV: {e}")
        raise


async def sample_json_data(
    input_json: str,
    output_json: str,
    strategy: str = "mixed",
    sampling_probability: float = 0.5,
    label_column: str = "is_valid",
):
    """
    Create negative samples from JSON file.

    Args:
        input_json: Input JSON file path
        output_json: Output JSON file path
        strategy: Sampling strategy name
        sampling_probability: Probability of sampling each document
        label_column: Column name for validity labels
    """
    try:
        # Load data
        with open(input_json, "r", encoding="utf-8") as f:
            raw_dataset = json.load(f)

        logger.info(f"Loaded {len(raw_dataset)} documents from {input_json}")

        # Filter for documents with non-empty triplets
        dataset = []
        for item in raw_dataset:
            triplets = item.get("triplets", [])
            if triplets and isinstance(triplets, list) and len(triplets) > 0:
                dataset.append(item)

        logger.info(f"Filtered to {len(dataset)} documents with non-empty triplets")

        if not dataset:
            logger.warning("⚠️  No documents with triplets found in dataset!")
            return

        # Initialize sampler and create negative samples
        sampler = NegativeSampler(
            sampling_probability=sampling_probability, strategy=strategy
        )

        logger.info(
            f"Creating negative samples with strategy: {sampler.strategy.get_strategy_name()}"
        )
        negative_dataset = sampler.create_negative_samples(
            dataset, label_column=label_column
        )

        # Save to JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(negative_dataset, f, ensure_ascii=False, indent=2)

        # Show statistics
        stats = sampler.get_sampling_stats()
        logger.info(f"✅ Negative sampling complete! Saved to {output_json}")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Total samples: {len(negative_dataset)}")
        logger.info(
            f"   Valid samples: {sum(1 for s in negative_dataset if s[label_column])}"
        )
        logger.info(
            f"   Negative samples: {sum(1 for s in negative_dataset if not s[label_column])}"
        )
        logger.info(f"   Sampling stats: {stats}")

        if stats.get("modifications_by_type"):
            logger.info(f"📝 Modifications by type:")
            for mod_type, count in stats["modifications_by_type"].items():
                logger.info(f"   {mod_type}: {count}")

    except Exception as e:
        logger.error(f"❌ Error processing JSON: {e}")
        raise


async def sample_bigquery_data(
    input_table: str,
    output_table: str,
    strategy: str = "mixed",
    sampling_probability: float = 0.5,
    project: Optional[str] = None,
    key_path: Optional[str] = None,
    batch_size: int = 100,
    label_column: str = "is_valid",
):
    """
    Create negative samples from BigQuery table.

    Args:
        input_table: Input BigQuery table ID
        output_table: Output BigQuery table ID
        strategy: Sampling strategy name
        sampling_probability: Probability of sampling each document
        project: Google Cloud Project ID
        key_path: Path to service account key file
        batch_size: Batch size for processing
        label_column: Column name for validity labels
    """
    try:
        from ..utils import (
            BigQueryClient,
            BigQueryBatchProcessor,
            create_legal_document_table,
            get_triplet_statistics,
            parse_json_column,
            format_table_stats,
        )
    except ImportError:
        logger.error("❌ BigQuery utilities not available")
        return

    # Initialize BigQuery client
    try:
        client = BigQueryClient(project=project, key_path=key_path)
    except Exception as e:
        logger.error(f"❌ Failed to initialize BigQuery client: {e}")
        return

    # Initialize sampler
    sampler = NegativeSampler(
        sampling_probability=sampling_probability, strategy=strategy
    )

    logger.info(f"🚀 Starting negative sampling: {input_table} -> {output_table}")
    logger.info(f"   Strategy: {sampler.strategy.get_strategy_name()}")
    logger.info(f"   Sampling probability: {sampling_probability}")

    # Get table statistics
    logger.info("🔍 Checking source table statistics...")
    stats = get_triplet_statistics(client, input_table)
    logger.info(format_table_stats(stats))

    # Create output table using centralized utility
    create_legal_document_table(
        client=client,
        table_id=output_table,
        include_entities=True,
        include_triplets=True,
        include_labels=True,
        label_column=label_column,
    )

    # Define processor function for batch processing
    def process_batch(batch_data):
        """Process a batch of documents and apply negative sampling"""
        dataset = []

        for row in batch_data:
            # Parse JSON columns using centralized utility
            entities = parse_json_column(row.get("tags"))
            triplets = parse_json_column(row.get("triplets"))

            # Skip if no triplets
            if not triplets or not isinstance(triplets, list):
                continue

            dataset.append(
                {
                    "doc_id": str(row["doc_id"]),
                    "text": str(row["text"]),
                    "entities": entities or [],
                    "triplets": triplets,
                    "triplets_count": len(triplets),
                }
            )

        if not dataset:
            return []

        # Apply negative sampling
        negative_dataset = sampler.create_negative_samples(
            dataset, label_column=label_column
        )

        # Format for insertion
        rows_to_insert = []
        for item in negative_dataset:
            rows_to_insert.append(
                {
                    "doc_id": item["doc_id"],
                    "text": item["text"],
                    "entities": json.dumps(
                        item.get("entities", []), ensure_ascii=False
                    ),
                    "triplets": json.dumps(
                        item.get("triplets", []), ensure_ascii=False
                    ),
                    "triplets_count": item.get("triplets_count", 0),
                    label_column: item[label_column],
                    "processing_timestamp": datetime.now().isoformat(),
                }
            )

        return rows_to_insert

    # Use batch processor
    processor = BigQueryBatchProcessor(client.client, batch_size=batch_size)
    processing_stats = processor.process_table(
        table_id=input_table,
        processor_func=process_batch,
        output_table_id=output_table,
        where_clause="text IS NOT NULL AND triplets IS NOT NULL AND triplets_count > 0",
        select_columns=["doc_id", "text", "tags", "triplets", "triplets_count"],
    )

    # Final statistics
    sampler_stats = sampler.get_sampling_stats()
    logger.info(f"\n✅ Negative sampling complete!")
    logger.info(f"📊 Batch Processing Statistics:")
    logger.info(f"   Total processed: {processing_stats['total_processed']}")
    logger.info(f"   Successfully inserted: {processing_stats['success_count']}")
    logger.info(f"   Errors: {processing_stats['error_count']}")

    logger.info(f"\n📊 Sampling Statistics:")
    logger.info(f"   Negative samples created: {sampler_stats['corrupted']}")
    logger.info(f"   Original samples kept: {sampler_stats['skipped']}")

    if sampler_stats.get("modifications_by_type"):
        logger.info(f"📝 Modifications by type:")
        for mod_type, count in sampler_stats["modifications_by_type"].items():
            logger.info(f"   {mod_type}: {count}")


def show_strategies():
    """Show available sampling strategies"""
    print("\n📋 Available Sampling Strategies")
    print("=" * 60)

    strategies_info = [
        ("article", "Modify article numbers by small offset (±1-5)"),
        ("part", "Modify part numbers by small offset (±1-2)"),
        ("combined", "Modify both article and part numbers"),
        ("swap", "Swap article and part numbers (creates invalid refs)"),
        ("random", "Replace with random valid components"),
        ("mixed", "Combine multiple strategies (default)"),
        ("conservative", "Conservative mix (small offsets)"),
        ("aggressive", "Aggressive mix (all strategies)"),
    ]

    for name, description in strategies_info:
        print(f"  {name:15s} - {description}")

    print("\n💡 Examples:")
    print("  python -m legal_graph.processors.negative_sampling sample-csv \\")
    print("      input.csv output.csv --strategy article --probability 0.3")
    print()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Legal Document Negative Sampling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create negative samples from CSV
  python -m legal_graph.processors.negative_sampling sample-csv \\
      input.csv output.csv --strategy mixed --probability 0.5

  # Create negative samples from JSON
  python -m legal_graph.processors.negative_sampling sample-json \\
      input.json output.json --strategy article

  # Create negative samples from BigQuery
  python -m legal_graph.processors.negative_sampling sample-bigquery \\
      project.dataset.input_table project.dataset.output_table \\
      --strategy aggressive --probability 0.7

  # Show available strategies
  python -m legal_graph.processors.negative_sampling strategies
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # CSV sampling command
    csv_parser = subparsers.add_parser(
        "sample-csv", help="Create negative samples from CSV"
    )
    csv_parser.add_argument("input_csv", help="Input CSV file path")
    csv_parser.add_argument("output_csv", help="Output CSV file path")
    csv_parser.add_argument(
        "--strategy",
        default="mixed",
        choices=[
            "article",
            "part",
            "combined",
            "swap",
            "random",
            "mixed",
            "conservative",
            "aggressive",
        ],
        help="Sampling strategy (default: mixed)",
    )
    csv_parser.add_argument(
        "--probability",
        type=float,
        default=0.5,
        help="Sampling probability 0.0-1.0 (default: 0.5)",
    )
    csv_parser.add_argument(
        "--label-column",
        default="is_valid",
        help="Label column name (default: is_valid)",
    )

    # JSON sampling command
    json_parser = subparsers.add_parser(
        "sample-json", help="Create negative samples from JSON"
    )
    json_parser.add_argument("input_json", help="Input JSON file path")
    json_parser.add_argument("output_json", help="Output JSON file path")
    json_parser.add_argument(
        "--strategy",
        default="mixed",
        choices=[
            "article",
            "part",
            "combined",
            "swap",
            "random",
            "mixed",
            "conservative",
            "aggressive",
        ],
        help="Sampling strategy (default: mixed)",
    )
    json_parser.add_argument(
        "--probability",
        type=float,
        default=0.5,
        help="Sampling probability 0.0-1.0 (default: 0.5)",
    )
    json_parser.add_argument(
        "--label-column",
        default="is_valid",
        help="Label column name (default: is_valid)",
    )

    # BigQuery sampling command
    bq_parser = subparsers.add_parser(
        "sample-bigquery", help="Create negative samples from BigQuery"
    )
    bq_parser.add_argument(
        "input_table", help="Input BigQuery table ID (project.dataset.table)"
    )
    bq_parser.add_argument("output_table", help="Output BigQuery table ID")
    bq_parser.add_argument(
        "--strategy",
        default="mixed",
        choices=[
            "article",
            "part",
            "combined",
            "swap",
            "random",
            "mixed",
            "conservative",
            "aggressive",
        ],
        help="Sampling strategy (default: mixed)",
    )
    bq_parser.add_argument(
        "--probability",
        type=float,
        default=0.5,
        help="Sampling probability 0.0-1.0 (default: 0.5)",
    )
    bq_parser.add_argument("--project", help="Google Cloud Project ID")
    bq_parser.add_argument("--key-path", help="Path to service account key file")
    bq_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    bq_parser.add_argument(
        "--label-column",
        default="is_valid",
        help="Label column name (default: is_valid)",
    )

    # Strategies info command
    strategies_parser = subparsers.add_parser(
        "strategies", help="Show available strategies"
    )

    args = parser.parse_args()

    if args.command == "sample-csv":
        asyncio.run(
            sample_csv_data(
                input_csv=args.input_csv,
                output_csv=args.output_csv,
                strategy=args.strategy,
                sampling_probability=args.probability,
                label_column=args.label_column,
            )
        )

    elif args.command == "sample-json":
        asyncio.run(
            sample_json_data(
                input_json=args.input_json,
                output_json=args.output_json,
                strategy=args.strategy,
                sampling_probability=args.probability,
                label_column=args.label_column,
            )
        )

    elif args.command == "sample-bigquery":
        asyncio.run(
            sample_bigquery_data(
                input_table=args.input_table,
                output_table=args.output_table,
                strategy=args.strategy,
                sampling_probability=args.probability,
                project=args.project,
                key_path=args.key_path,
                batch_size=args.batch_size,
                label_column=args.label_column,
            )
        )

    elif args.command == "strategies":
        show_strategies()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
