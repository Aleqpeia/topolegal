#!/usr/bin/env python3
"""
Example: Complete Dataset Reformatting Workflow

This script demonstrates the complete workflow for reformatting a BigQuery
legal graph dataset and creating stratified train/test/val splits.

Usage:
    python example_dataset_reformatting.py
"""

import json
from bigquery_dataset_reformatter import BigQueryDatasetReformatter


def main():
    """Run complete reformatting workflow"""

    # Configuration
    PROJECT_ID = "lab-test-project-1-305710"
    DATASET_ID = "court_data_2022"
    SOURCE_TABLE = f"{PROJECT_ID}.{DATASET_ID}.processing_doc_links"
    REFORMATTED_TABLE = f"{PROJECT_ID}.{DATASET_ID}.reformatted_triplets"
    TRAIN_TABLE = f"{PROJECT_ID}.{DATASET_ID}.train"
    TEST_TABLE = f"{PROJECT_ID}.{DATASET_ID}.test"
    VAL_TABLE = f"{PROJECT_ID}.{DATASET_ID}.val"
    GROUPING_VIEW = f"{PROJECT_ID}.{DATASET_ID}.grouping_stats"

    # Initialize reformatter
    print("🚀 Initializing BigQuery Dataset Reformatter")
    print("=" * 60)

    reformatter = BigQueryDatasetReformatter(
        project_id=PROJECT_ID,
        credentials_path="lab-test-project-1-305710-30eed237388b.json"
    )

    # Step 1: Analyze source data
    print("\n📊 Step 1: Analyzing source data...")
    print("-" * 60)

    stats = reformatter.analyze_triplet_statistics(SOURCE_TABLE)

    print(f"\nTotal triplets: {stats['total_triplets']}")
    print(f"Unique legal references: {stats['unique_legal_references']}")
    print(f"Unique relations: {stats['unique_relations']}")

    print("\nTop 10 Legal References:")
    for item in stats['top_legal_references'][:10]:
        print(f"  {item['legal_reference']}: {item['total_occurrences_legal_ref']} occurrences")

    print("\nTop 10 Relations:")
    for item in stats['top_relations'][:10]:
        print(f"  {item['relation']}: {item['total_occurrences_relation']} occurrences")

    # Save statistics
    with open('dataset_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("\n✅ Statistics saved to dataset_statistics.json")

    # Step 2: Reformat dataset
    print("\n🔄 Step 2: Reformatting dataset...")
    print("-" * 60)

    reformatter.reformat_dataset(
        source_table=SOURCE_TABLE,
        dest_table=REFORMATTED_TABLE
    )

    print(f"✅ Dataset reformatted: {REFORMATTED_TABLE}")

    # Step 3: Create grouping view
    print("\n📈 Step 3: Creating grouping view...")
    print("-" * 60)

    reformatter.create_grouping_view(
        source_table=REFORMATTED_TABLE,
        view_name=GROUPING_VIEW,
        group_by=['legal_reference', 'relation']
    )

    print(f"✅ Grouping view created: {GROUPING_VIEW}")

    # Step 4: Create stratified splits
    print("\n✂️  Step 4: Creating stratified train/test/val splits...")
    print("-" * 60)

    split_stats = reformatter.stratified_split(
        source_table=REFORMATTED_TABLE,
        train_table=TRAIN_TABLE,
        test_table=TEST_TABLE,
        val_table=VAL_TABLE,
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1,
        stratify_by='legal_reference',
        random_seed=42
    )

    print("\n📊 Split Statistics:")
    for split_name, split_info in split_stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Table: {split_info['table']}")
        print(f"  Unique documents: {split_info['unique_docs']}")
        print(f"  Total rows (triplets): {split_info['total_rows']}")

    # Step 5: Validate splits
    print("\n✅ Step 5: Validating splits...")
    print("-" * 60)

    validation_results = reformatter.validate_split(
        train_table=TRAIN_TABLE,
        test_table=TEST_TABLE,
        val_table=VAL_TABLE
    )

    print(f"\nValidation Results:")
    print(f"  Valid: {'✅ YES' if validation_results['valid'] else '❌ NO'}")
    print(f"  Overlapping documents: {validation_results['overlapping_docs']}")

    print("\n📊 Distribution Across Splits:")
    for split_name, stats in validation_results['splits_stats'].items():
        print(f"\n{split_name.upper()}:")
        print(f"  Unique documents: {stats['unique_docs']}")
        print(f"  Total triplets: {stats['total_triplets']}")
        print(f"  Unique legal references: {stats['unique_legal_refs']}")
        print(f"  Unique relations: {stats['unique_relations']}")

    # Step 6: Get stratification groups for inspection
    print("\n🔍 Step 6: Analyzing stratification groups...")
    print("-" * 60)

    groups = reformatter.get_stratification_groups(
        source_table=REFORMATTED_TABLE,
        group_by='legal_reference',
        min_samples=10
    )

    print(f"\nTotal stratification groups (≥10 docs): {len(groups)}")
    print("\nTop 10 groups by document count:")
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (group_key, doc_ids) in enumerate(sorted_groups[:10], 1):
        print(f"  {i}. {group_key}: {len(doc_ids)} documents")

    # Summary
    print("\n" + "=" * 60)
    print("🎉 REFORMATTING COMPLETE!")
    print("=" * 60)
    print("\n📋 Summary:")
    print(f"  Source table: {SOURCE_TABLE}")
    print(f"  Reformatted table: {REFORMATTED_TABLE}")
    print(f"  Grouping view: {GROUPING_VIEW}")
    print(f"\n  Train table: {TRAIN_TABLE}")
    print(f"  Test table: {TEST_TABLE}")
    print(f"  Val table: {VAL_TABLE}")

    print("\n💡 Next Steps:")
    print("  1. Review statistics in dataset_statistics.json")
    print("  2. Query grouping view for detailed analysis:")
    print(f"     SELECT * FROM `{GROUPING_VIEW}` LIMIT 100")
    print("  3. Use splits with PyTorch Geometric:")
    print("     from legal_graph.models import GraphDataset")
    print(f"     train_dataset = GraphDataset(table_id='{TRAIN_TABLE}')")
    print("  4. Export to CSV if needed:")
    print(f"     bq extract --destination_format=CSV {TRAIN_TABLE} gs://bucket/train.csv")

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
