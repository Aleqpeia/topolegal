-- ============================================================================
-- BigQuery Dataset Reformatting Queries
-- ============================================================================
-- This file contains SQL queries for extracting triplet JSON columns into
-- separate fields for legal reference, source, target, and relation.
--
-- Usage: Run these queries in BigQuery console or via bq command-line tool
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. BASIC TRIPLET EXTRACTION
-- Extract all triplet fields from JSON array into separate columns
-- ----------------------------------------------------------------------------

CREATE OR REPLACE TABLE `project.dataset.reformatted_table` AS
WITH triplet_data AS (
    SELECT
        doc_id,
        text,
        tags,
        triplets,
        triplets_count,
        JSON_EXTRACT_ARRAY(triplets) AS triplet_array
    FROM `project.dataset.source_table`
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
    REGEXP_EXTRACT(legal_reference, r'ст\.\s*(\d+)') AS article_number,
    REGEXP_EXTRACT(legal_reference, r'ч\.\s*(\d+)') AS part_number,
    REGEXP_EXTRACT(legal_reference, r'(КК|КПК|ЦК|КоАП|СК|КЗпП)\s*України') AS legal_code,
    CURRENT_TIMESTAMP() AS reformatted_timestamp
FROM exploded_triplets
ORDER BY doc_id, triplet_index;

-- ----------------------------------------------------------------------------
-- 2. STATISTICS QUERY
-- Analyze distribution of legal references and relations
-- ----------------------------------------------------------------------------

WITH triplet_stats AS (
    SELECT
        legal_reference,
        relation,
        COUNT(DISTINCT doc_id) AS unique_docs,
        COUNT(*) AS total_occurrences
    FROM `project.dataset.reformatted_table`
    GROUP BY legal_reference, relation
)
SELECT
    legal_reference,
    relation,
    unique_docs,
    total_occurrences,
    ROUND(100.0 * total_occurrences / SUM(total_occurrences) OVER (), 2) AS percentage
FROM triplet_stats
ORDER BY total_occurrences DESC
LIMIT 100;

-- ----------------------------------------------------------------------------
-- 3. GROUPING CONDITIONS - Legal Reference Distribution
-- Define grouping for stratified splitting based on legal references
-- ----------------------------------------------------------------------------

CREATE OR REPLACE VIEW `project.dataset.legal_reference_groups` AS
WITH legal_ref_stats AS (
    SELECT
        legal_reference,
        legal_code,
        article_number,
        part_number,
        COUNT(DISTINCT doc_id) AS unique_docs,
        COUNT(*) AS total_triplets,
        AVG(confidence) AS avg_confidence
    FROM `project.dataset.reformatted_table`
    GROUP BY legal_reference, legal_code, article_number, part_number
)
SELECT
    legal_reference,
    legal_code,
    article_number,
    part_number,
    unique_docs,
    total_triplets,
    avg_confidence,
    -- Categorize by frequency
    CASE
        WHEN unique_docs >= 100 THEN 'high_frequency'
        WHEN unique_docs >= 20 THEN 'medium_frequency'
        WHEN unique_docs >= 5 THEN 'low_frequency'
        ELSE 'rare'
    END AS frequency_category,
    -- Categorize by legal code
    CASE
        WHEN legal_code = 'КК' THEN 'criminal'
        WHEN legal_code = 'КПК' THEN 'criminal_procedure'
        WHEN legal_code = 'ЦК' THEN 'civil'
        WHEN legal_code = 'КоАП' THEN 'administrative'
        WHEN legal_code = 'СК' THEN 'family'
        WHEN legal_code = 'КЗпП' THEN 'labor'
        ELSE 'other'
    END AS code_category
FROM legal_ref_stats
ORDER BY total_triplets DESC;

-- ----------------------------------------------------------------------------
-- 4. GROUPING CONDITIONS - Relation Distribution
-- Define grouping for stratified splitting based on relations
-- ----------------------------------------------------------------------------

CREATE OR REPLACE VIEW `project.dataset.relation_groups` AS
WITH relation_stats AS (
    SELECT
        relation,
        COUNT(DISTINCT doc_id) AS unique_docs,
        COUNT(*) AS total_triplets,
        AVG(confidence) AS avg_confidence,
        COUNT(DISTINCT legal_reference) AS unique_legal_refs
    FROM `project.dataset.reformatted_table`
    GROUP BY relation
)
SELECT
    relation,
    unique_docs,
    total_triplets,
    avg_confidence,
    unique_legal_refs,
    -- Categorize by frequency
    CASE
        WHEN unique_docs >= 100 THEN 'high_frequency'
        WHEN unique_docs >= 20 THEN 'medium_frequency'
        WHEN unique_docs >= 5 THEN 'low_frequency'
        ELSE 'rare'
    END AS frequency_category,
    -- Categorize by semantic type (adjust based on your relations)
    CASE
        WHEN relation LIKE '%винним%' OR relation LIKE '%засудж%' THEN 'conviction'
        WHEN relation LIKE '%звільн%' OR relation LIKE '%виправда%' THEN 'acquittal'
        WHEN relation LIKE '%застосува%' OR relation LIKE '%призначен%' THEN 'sentencing'
        WHEN relation LIKE '%порушен%' THEN 'violation'
        ELSE 'other'
    END AS semantic_category
FROM relation_stats
ORDER BY total_triplets DESC;

-- ----------------------------------------------------------------------------
-- 5. COMBINED GROUPING VIEW
-- Combine legal reference and relation statistics for comprehensive analysis
-- ----------------------------------------------------------------------------

CREATE OR REPLACE VIEW `project.dataset.combined_groups` AS
SELECT
    legal_reference,
    relation,
    COUNT(DISTINCT doc_id) AS unique_docs,
    COUNT(*) AS total_triplets,
    AVG(confidence) AS avg_confidence,
    MIN(confidence) AS min_confidence,
    MAX(confidence) AS max_confidence,
    -- Legal reference metadata
    MAX(legal_code) AS legal_code,
    MAX(article_number) AS article_number,
    MAX(part_number) AS part_number,
    -- Aggregate text samples for inspection
    STRING_AGG(DISTINCT source, '; ' LIMIT 5) AS sample_sources,
    STRING_AGG(DISTINCT target, '; ' LIMIT 5) AS sample_targets,
    CURRENT_TIMESTAMP() AS view_created_at
FROM `project.dataset.reformatted_table`
GROUP BY legal_reference, relation
ORDER BY total_triplets DESC;

-- ----------------------------------------------------------------------------
-- 6. STRATIFIED SPLIT - Training Set
-- Create training set with stratified sampling (70%)
-- ----------------------------------------------------------------------------

CREATE OR REPLACE TABLE `project.dataset.train` AS
WITH doc_assignment AS (
    SELECT DISTINCT
        doc_id,
        legal_reference,
        relation,
        -- Use hash for deterministic random assignment
        MOD(ABS(FARM_FINGERPRINT(CONCAT(doc_id, 'salt42'))), 100) AS hash_value
    FROM `project.dataset.reformatted_table`
)
SELECT r.*
FROM `project.dataset.reformatted_table` r
INNER JOIN doc_assignment da
    ON r.doc_id = da.doc_id
WHERE da.hash_value < 70  -- 70% for training
ORDER BY r.doc_id, r.triplet_index;

-- ----------------------------------------------------------------------------
-- 7. STRATIFIED SPLIT - Test Set
-- Create test set with stratified sampling (20%)
-- ----------------------------------------------------------------------------

CREATE OR REPLACE TABLE `project.dataset.test` AS
WITH doc_assignment AS (
    SELECT DISTINCT
        doc_id,
        legal_reference,
        relation,
        MOD(ABS(FARM_FINGERPRINT(CONCAT(doc_id, 'salt42'))), 100) AS hash_value
    FROM `project.dataset.reformatted_table`
)
SELECT r.*
FROM `project.dataset.reformatted_table` r
INNER JOIN doc_assignment da
    ON r.doc_id = da.doc_id
WHERE da.hash_value >= 70 AND da.hash_value < 90  -- 20% for testing
ORDER BY r.doc_id, r.triplet_index;

-- ----------------------------------------------------------------------------
-- 8. STRATIFIED SPLIT - Validation Set
-- Create validation set with stratified sampling (10%)
-- ----------------------------------------------------------------------------

CREATE OR REPLACE TABLE `project.dataset.val` AS
WITH doc_assignment AS (
    SELECT DISTINCT
        doc_id,
        legal_reference,
        relation,
        MOD(ABS(FARM_FINGERPRINT(CONCAT(doc_id, 'salt42'))), 100) AS hash_value
    FROM `project.dataset.reformatted_table`
)
SELECT r.*
FROM `project.dataset.reformatted_table` r
INNER JOIN doc_assignment da
    ON r.doc_id = da.doc_id
WHERE da.hash_value >= 90  -- 10% for validation
ORDER BY r.doc_id, r.triplet_index;

-- ----------------------------------------------------------------------------
-- 9. VALIDATION QUERY - Check for document overlap
-- Verify that train/test/val splits have no overlapping documents
-- ----------------------------------------------------------------------------

WITH all_docs AS (
    SELECT doc_id, 'train' as split FROM `project.dataset.train`
    UNION ALL
    SELECT doc_id, 'test' as split FROM `project.dataset.test`
    UNION ALL
    SELECT doc_id, 'val' as split FROM `project.dataset.val`
),
doc_counts AS (
    SELECT
        doc_id,
        COUNT(DISTINCT split) as split_count,
        STRING_AGG(DISTINCT split, ', ') as splits
    FROM all_docs
    GROUP BY doc_id
    HAVING split_count > 1
)
SELECT
    COUNT(*) as overlapping_docs,
    STRING_AGG(doc_id, ', ' LIMIT 10) as sample_overlapping_docs
FROM doc_counts;

-- ----------------------------------------------------------------------------
-- 10. SPLIT STATISTICS
-- Compare distribution across train/test/val splits
-- ----------------------------------------------------------------------------

WITH split_stats AS (
    SELECT
        'train' as split,
        COUNT(DISTINCT doc_id) as unique_docs,
        COUNT(*) as total_triplets,
        COUNT(DISTINCT legal_reference) as unique_legal_refs,
        COUNT(DISTINCT relation) as unique_relations,
        AVG(confidence) as avg_confidence
    FROM `project.dataset.train`

    UNION ALL

    SELECT
        'test' as split,
        COUNT(DISTINCT doc_id) as unique_docs,
        COUNT(*) as total_triplets,
        COUNT(DISTINCT legal_reference) as unique_legal_refs,
        COUNT(DISTINCT relation) as unique_relations,
        AVG(confidence) as avg_confidence
    FROM `project.dataset.test`

    UNION ALL

    SELECT
        'val' as split,
        COUNT(DISTINCT doc_id) as unique_docs,
        COUNT(*) as total_triplets,
        COUNT(DISTINCT legal_reference) as unique_legal_refs,
        COUNT(DISTINCT relation) as unique_relations,
        AVG(confidence) as avg_confidence
    FROM `project.dataset.val`
)
SELECT
    split,
    unique_docs,
    total_triplets,
    unique_legal_refs,
    unique_relations,
    ROUND(avg_confidence, 3) as avg_confidence,
    ROUND(100.0 * unique_docs / SUM(unique_docs) OVER (), 2) as doc_percentage,
    ROUND(100.0 * total_triplets / SUM(total_triplets) OVER (), 2) as triplet_percentage
FROM split_stats
ORDER BY
    CASE split
        WHEN 'train' THEN 1
        WHEN 'test' THEN 2
        WHEN 'val' THEN 3
    END;

-- ----------------------------------------------------------------------------
-- 11. DISTRIBUTION CHECK - Legal Reference Coverage
-- Ensure all legal references are represented across splits
-- ----------------------------------------------------------------------------

WITH legal_ref_coverage AS (
    SELECT
        legal_reference,
        COUNTIF(split = 'train') as in_train,
        COUNTIF(split = 'test') as in_test,
        COUNTIF(split = 'val') as in_val,
        COUNT(DISTINCT split) as num_splits
    FROM (
        SELECT doc_id, legal_reference, 'train' as split FROM `project.dataset.train`
        UNION ALL
        SELECT doc_id, legal_reference, 'test' as split FROM `project.dataset.test`
        UNION ALL
        SELECT doc_id, legal_reference, 'val' as split FROM `project.dataset.val`
    )
    GROUP BY legal_reference
)
SELECT
    COUNT(*) as total_legal_refs,
    COUNTIF(num_splits = 3) as in_all_three_splits,
    COUNTIF(num_splits = 2) as in_two_splits,
    COUNTIF(num_splits = 1) as in_one_split_only,
    ROUND(100.0 * COUNTIF(num_splits = 3) / COUNT(*), 2) as pct_in_all_splits
FROM legal_ref_coverage;

-- ----------------------------------------------------------------------------
-- 12. DISTRIBUTION CHECK - Relation Coverage
-- Ensure all relations are represented across splits
-- ----------------------------------------------------------------------------

WITH relation_coverage AS (
    SELECT
        relation,
        COUNTIF(split = 'train') as in_train,
        COUNTIF(split = 'test') as in_test,
        COUNTIF(split = 'val') as in_val,
        COUNT(DISTINCT split) as num_splits
    FROM (
        SELECT doc_id, relation, 'train' as split FROM `project.dataset.train`
        UNION ALL
        SELECT doc_id, relation, 'test' as split FROM `project.dataset.test`
        UNION ALL
        SELECT doc_id, relation, 'val' as split FROM `project.dataset.val`
    )
    GROUP BY relation
)
SELECT
    COUNT(*) as total_relations,
    COUNTIF(num_splits = 3) as in_all_three_splits,
    COUNTIF(num_splits = 2) as in_two_splits,
    COUNTIF(num_splits = 1) as in_one_split_only,
    ROUND(100.0 * COUNTIF(num_splits = 3) / COUNT(*), 2) as pct_in_all_splits
FROM relation_coverage;

-- ----------------------------------------------------------------------------
-- 13. EXPORT TO CSV (for local analysis)
-- Use bq command-line tool to export results
-- ----------------------------------------------------------------------------

-- Run this in terminal, not in BigQuery console:
-- bq extract --destination_format=CSV \
--   project:dataset.train \
--   gs://your-bucket/train.csv

-- bq extract --destination_format=CSV \
--   project:dataset.test \
--   gs://your-bucket/test.csv

-- bq extract --destination_format=CSV \
--   project:dataset.val \
--   gs://your-bucket/val.csv
