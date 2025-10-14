#!/usr/bin/env python
"""
BigQuery Integration Example for Negative Sampling

This script demonstrates how to use the negative sampling module
with Google BigQuery for large-scale legal document processing.
"""

import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def example_1_basic_bigquery_setup():
    """Example 1: Basic BigQuery connection setup"""
    print("=" * 70)
    print("Example 1: Basic BigQuery Setup")
    print("=" * 70)
    print()

    setup_instructions = """
    Prerequisites:
    1. Install BigQuery client:
       pip install google-cloud-bigquery

    2. Authenticate with Google Cloud (choose one):
       a) User credentials:
          gcloud auth application-default login

       b) Service account:
          export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

       c) In code:
          from google.cloud import bigquery
          client = bigquery.Client(
              project="your-project-id",
              credentials=credentials
          )

    3. Ensure your BigQuery table has these columns:
       - doc_id        (STRING)
       - text          (STRING)
       - tags          (JSON or STRING) - entities
       - triplets      (JSON or STRING) - knowledge graph triplets
       - triplets_count (INTEGER) - number of triplets
    """

    print(setup_instructions)
    print()


def example_2_cli_bigquery():
    """Example 2: Using CLI for BigQuery processing"""
    print("=" * 70)
    print("Example 2: CLI Interface for BigQuery")
    print("=" * 70)
    print()

    cli_examples = """
    Basic Usage:
    -----------
    python -m legal_graph.processors.negative_sampling sample-bigquery \\
        project.dataset.input_table \\
        project.dataset.output_table \\
        --strategy mixed \\
        --probability 0.5

    With Authentication:
    -------------------
    python -m legal_graph.processors.negative_sampling sample-bigquery \\
        project.dataset.input_table \\
        project.dataset.output_table \\
        --strategy aggressive \\
        --probability 0.7 \\
        --project "your-project-id" \\
        --key-path "/path/to/service-account.json"

    Advanced Options:
    ----------------
    python -m legal_graph.processors.negative_sampling sample-bigquery \\
        project.dataset.input_table \\
        project.dataset.output_table \\
        --strategy conservative \\
        --probability 0.3 \\
        --batch-size 50 \\
        --label-column "is_valid"

    Show Available Strategies:
    -------------------------
    python -m legal_graph.processors.negative_sampling strategies
    """

    print(cli_examples)
    print()


def example_3_python_api():
    """Example 3: Python API for BigQuery"""
    print("=" * 70)
    print("Example 3: Python API for BigQuery Integration")
    print("=" * 70)
    print()

    code_example = '''
from google.cloud import bigquery
from legal_graph.processors.negative_sampling import NegativeSampler
import json

# Initialize BigQuery client
client = bigquery.Client(project="your-project-id")

# Query documents from BigQuery
query = """
    SELECT doc_id, text, tags, triplets, triplets_count
    FROM `project.dataset.legal_documents`
    WHERE triplets_count > 0
    LIMIT 1000
"""

results = client.query(query).to_dataframe()

# Convert to dataset format
dataset = []
for _, row in results.iterrows():
    dataset.append({
        'doc_id': str(row.doc_id),
        'text': str(row.text),
        'entities': json.loads(row.tags) if row.tags else [],
        'triplets': json.loads(row.triplets) if row.triplets else [],
        'triplets_count': int(row.triplets_count)
    })

# Create negative samples
sampler = NegativeSampler(
    sampling_probability=0.5,
    strategy='mixed'
)

negative_samples = sampler.create_negative_samples(dataset)

# Show statistics
stats = sampler.get_sampling_stats()
print(f"Total processed: {stats['total_processed']}")
print(f"Negative samples: {stats['corrupted']}")
print(f"Original samples: {stats['skipped']}")

# Insert back to BigQuery
output_table = "project.dataset.output_table"
rows_to_insert = []

for sample in negative_samples:
    row = {
        'doc_id': sample['doc_id'],
        'text': sample['text'],
        'entities': json.dumps(sample.get('entities', []), ensure_ascii=False),
        'triplets': json.dumps(sample.get('triplets', []), ensure_ascii=False),
        'triplets_count': sample.get('triplets_count', 0),
        'is_valid': sample['is_valid']
    }
    rows_to_insert.append(row)

# Insert in batches
batch_size = 100
for i in range(0, len(rows_to_insert), batch_size):
    batch = rows_to_insert[i:i + batch_size]
    errors = client.insert_rows_json(output_table, batch)
    if errors:
        print(f"Errors in batch {i//batch_size}: {errors}")
    else:
        print(f"Batch {i//batch_size} inserted successfully")
    '''

    print(code_example)
    print()


def example_4_async_processing():
    """Example 4: Async processing for large datasets"""
    print("=" * 70)
    print("Example 4: Async Processing for Large Datasets")
    print("=" * 70)
    print()

    async_example = '''
import asyncio
from google.cloud import bigquery
from legal_graph.processors.negative_sampling import NegativeSampler
import json
from datetime import datetime

async def process_bigquery_batch(
    client,
    input_table: str,
    output_table: str,
    batch_size: int = 100,
    strategy: str = 'mixed'
):
    """Process BigQuery data in batches asynchronously"""

    sampler = NegativeSampler(
        sampling_probability=0.5,
        strategy=strategy
    )

    processed = 0
    total_negative = 0

    while True:
        # Fetch batch
        query = f"""
            SELECT doc_id, text, tags, triplets, triplets_count
            FROM `{input_table}`
            WHERE text IS NOT NULL
            LIMIT {batch_size}
            OFFSET {processed}
        """

        results = client.query(query).to_dataframe()

        if results.empty:
            break

        # Convert to dataset format
        dataset = []
        for _, row in results.iterrows():
            dataset.append({
                'doc_id': str(row.doc_id),
                'text': str(row.text),
                'entities': json.loads(row.tags) if pd.notna(row.tags) else [],
                'triplets': json.loads(row.triplets) if pd.notna(row.triplets) else [],
                'triplets_count': int(row.triplets_count) if pd.notna(row.triplets_count) else 0
            })

        # Create negative samples
        negative_samples = sampler.create_negative_samples(dataset)

        # Insert to output table
        rows_to_insert = []
        for sample in negative_samples:
            rows_to_insert.append({
                'doc_id': sample['doc_id'],
                'text': sample['text'],
                'entities': json.dumps(sample.get('entities', []), ensure_ascii=False),
                'triplets': json.dumps(sample.get('triplets', []), ensure_ascii=False),
                'triplets_count': sample.get('triplets_count', 0),
                'is_valid': sample['is_valid'],
                'processing_timestamp': datetime.now().isoformat()
            })

        errors = client.insert_rows_json(output_table, rows_to_insert)
        if not errors:
            processed += len(results)
            batch_negative = sum(1 for s in negative_samples if not s['is_valid'])
            total_negative += batch_negative
            print(f"Processed {processed} documents, {total_negative} negative samples")
        else:
            print(f"Errors: {errors}")
            break

    # Final statistics
    stats = sampler.get_sampling_stats()
    return {
        'total_processed': processed,
        'negative_samples': total_negative,
        'stats': stats
    }

# Usage
async def main():
    client = bigquery.Client(project="your-project-id")

    result = await process_bigquery_batch(
        client,
        input_table="project.dataset.legal_documents",
        output_table="project.dataset.negative_samples",
        batch_size=100,
        strategy='aggressive'
    )

    print(f"\\nFinal Results:")
    print(f"  Total processed: {result['total_processed']}")
    print(f"  Negative samples: {result['negative_samples']}")
    print(f"  Stats: {result['stats']}")

# Run
asyncio.run(main())
    '''

    print(async_example)
    print()


def example_5_bigquery_schema():
    """Example 5: BigQuery table schema"""
    print("=" * 70)
    print("Example 5: BigQuery Table Schema")
    print("=" * 70)
    print()

    schema_info = """
    Input Table Schema:
    ------------------
    CREATE TABLE `project.dataset.legal_documents` (
        doc_id STRING NOT NULL,
        text STRING NOT NULL,
        tags JSON,           -- or STRING containing JSON
        triplets JSON,       -- or STRING containing JSON
        triplets_count INT64
    );

    Output Table Schema (with negative sampling):
    --------------------------------------------
    CREATE TABLE `project.dataset.negative_samples` (
        doc_id STRING NOT NULL,
        text STRING NOT NULL,
        entities JSON,
        triplets JSON,
        triplets_count INT64,
        is_valid BOOL NOT NULL,           -- True = original, False = negative sample
        processing_timestamp TIMESTAMP
    );

    Example Query to Check Results:
    -------------------------------
    SELECT
        COUNT(*) as total,
        COUNTIF(is_valid) as original_samples,
        COUNTIF(NOT is_valid) as negative_samples,
        ROUND(COUNTIF(NOT is_valid) / COUNT(*) * 100, 2) as negative_percentage
    FROM `project.dataset.negative_samples`;

    Example Query to Compare References:
    ------------------------------------
    SELECT
        o.doc_id,
        o.text as original_text,
        n.text as modified_text,
        o.text = n.text as is_same
    FROM `project.dataset.legal_documents` o
    LEFT JOIN `project.dataset.negative_samples` n
        ON o.doc_id = n.doc_id
    WHERE NOT n.is_valid
    LIMIT 10;
    """

    print(schema_info)
    print()


def example_6_monitoring_statistics():
    """Example 6: Monitoring and statistics"""
    print("=" * 70)
    print("Example 6: Monitoring and Statistics")
    print("=" * 70)
    print()

    monitoring_example = '''
from legal_graph.processors.negative_sampling import NegativeSampler

# Initialize sampler
sampler = NegativeSampler(strategy='mixed', sampling_probability=0.5)

# Process dataset
negative_samples = sampler.create_negative_samples(dataset)

# Get detailed statistics
stats = sampler.get_sampling_stats()

print(f"Processing Statistics:")
print(f"  Total processed: {stats['total_processed']}")
print(f"  Negative samples created: {stats['corrupted']}")
print(f"  Original samples kept: {stats['skipped']}")
print(f"  Errors: {stats['errors']}")
print(f"  Strategy used: {stats['strategy']}")

print(f"\\nModifications by type:")
for mod_type, count in stats['modifications_by_type'].items():
    percentage = (count / stats['corrupted'] * 100) if stats['corrupted'] > 0 else 0
    print(f"  {mod_type:40s}: {count:4d} ({percentage:5.1f}%)")

# Export statistics to BigQuery
import json
from datetime import datetime

stats_row = {
    'run_timestamp': datetime.now().isoformat(),
    'strategy': stats['strategy'],
    'total_processed': stats['total_processed'],
    'negative_samples': stats['corrupted'],
    'original_samples': stats['skipped'],
    'errors': stats['errors'],
    'modifications': json.dumps(stats['modifications_by_type'], ensure_ascii=False)
}

# Insert to statistics table
client.insert_rows_json('project.dataset.sampling_statistics', [stats_row])
    '''

    print(monitoring_example)
    print()


def example_7_best_practices():
    """Example 7: Best practices"""
    print("=" * 70)
    print("Example 7: Best Practices for BigQuery Integration")
    print("=" * 70)
    print()

    best_practices = """
    1. Batch Processing:
       - Use batch_size=100-500 for optimal performance
       - Process incrementally to handle interruptions
       - Use OFFSET/LIMIT for pagination

    2. Error Handling:
       - Always check insert_rows_json() return value
       - Log failed batches for retry
       - Use transactions when possible

    3. Cost Optimization:
       - Filter data in SQL (WHERE clauses) before processing
       - Use partitioned tables for large datasets
       - Consider costs of scanning vs. processing

    4. Performance:
       - Create indexes on doc_id for joins
       - Use streaming inserts for real-time processing
       - Batch inserts for bulk processing

    5. Data Quality:
       - Validate JSON before insertion
       - Check for NULL values
       - Ensure legal references are well-formed

    6. Monitoring:
       - Track processing rates
       - Monitor error rates
       - Log statistics for analysis

    7. Testing:
       - Start with small batches (LIMIT 100)
       - Test on sample data first
       - Verify output quality before full run

    8. Strategy Selection:
       Strategy          Use Case                      Sampling Rate
       -------------------------------------------------------------
       conservative      Subtle modifications          0.3-0.5
       mixed             General purpose               0.5-0.7
       aggressive        Maximum diversity             0.7-0.9
       article           Article-focused training      0.4-0.6
       part              Part-focused training         0.3-0.5

    9. Resource Management:
       - Close BigQuery connections properly
       - Use context managers (with statements)
       - Implement timeouts for long queries

    10. Security:
        - Use service accounts with minimal permissions
        - Never commit credentials to version control
        - Use Secret Manager for sensitive data
    """

    print(best_practices)
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("BigQuery Integration Guide for Negative Sampling")
    print("=" * 70 + "\n")

    examples = [
        example_1_basic_bigquery_setup,
        example_2_cli_bigquery,
        example_3_python_api,
        example_4_async_processing,
        example_5_bigquery_schema,
        example_6_monitoring_statistics,
        example_7_best_practices,
    ]

    for example in examples:
        example()
        input("Press Enter to continue to next example...")
        print("\n")

    print("=" * 70)
    print("End of BigQuery Integration Guide")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - src/legal_graph/processors/negative_sampling/README.md")
    print("  - src/legal_graph/processors/negative_sampling/ARCHITECTURE.md")
    print()


if __name__ == "__main__":
    main()