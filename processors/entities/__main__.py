#!/usr/bin/env python3
"""
Entry point for legal-text NER extraction using custom spaCy components
and BigQuery flag checking with service account authentication.

Usage:
    python -m processors.entities [--test-bq] [INPUT_FILE] [DOC_ID]
    cat file.txt | python -m processors.entities [--test-bq] [DOC_ID]

Options:
    --test-bq      Run a simple BigQuery SELECT 1 to verify connectivity.

Authentication:
    Set GOOGLE_APPLICATION_CREDENTIALS to your service key JSON path.
    Optionally place in a `.env` file to load automatically.
"""
import sys
import os
import argparse
from dotenv import load_dotenv
import spacy
from google.cloud import bigquery
from google.oauth2 import service_account

# Load .env
load_dotenv(dotenv_path="/home/efyis/RustroverProjects/topolegal/processors/entities/.env")
print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
# Register custom components
import processors.entities.person
import processors.entities.date
import processors.entities.case
import processors.entities.role
import processors.entities.doctype
import processors.entities.crime


def build_pipeline(use_pretrained: bool = False):
    nlp = spacy.load("uk_core_news_sm") if use_pretrained else spacy.blank("uk")
    for name in [
        "person_component",
        "date_component",
        "case_component",
        "role_component",
        "doctype_component",
        "crime_component",
    ]:
        nlp.add_pipe(name, last=True)
    return nlp


def get_bigquery_client():
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    creds = service_account.Credentials.from_service_account_file(key_path)
    return bigquery.Client(credentials=creds, project=creds.project_id)


def test_bigquery(client):
    try:
        query_job = client.query("SELECT 1 AS test_field")
        result = next(iter(query_job.result()))
        print(f"BigQuery test succeeded: {result.test_field}")
        sys.exit(0)
    except Exception as e:
        sys.exit(f"BigQuery test failed: {e}")


def check_flag_present(table_ref: str, doc_id: str, client) -> bool:
    query = f"SELECT flag FROM `{table_ref}` WHERE doc_id = @doc_id LIMIT 1"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id)]
    )
    rows = client.query(query, job_config=job_config).result()
    return any(row.flag for row in rows)


def main():
    parser = argparse.ArgumentParser(description="NER + BigQuery flag checker")
    parser.add_argument("input", nargs="?", help="Input file path")
    parser.add_argument("doc_id", help="Document identifier (or skip if --test-bq)")
    parser.add_argument("--test-bq", action="store_true", help="Test BigQuery connectivity and exit")
    args = parser.parse_args()

    client = get_bigquery_client()
    if args.test_bq:
        test_bigquery(client)

    # Read input
    if args.input and os.path.isfile(args.input):
        text = open(args.input, encoding="utf-8").read()
    else:
        text = sys.stdin.read()
    doc_id = args.doc_id

    table_ref = os.getenv("BQ_TABLE", "project.dataset.flags_table")
    if check_flag_present(table_ref, doc_id, client):
        sys.exit(f"FLAG_PRESENT for doc_id={doc_id}, skipping...")

    nlp = build_pipeline(use_pretrained=False)
    doc = nlp(text)
    for ent in doc.ents:
        print(f"{ent.label_}\t{ent.text}\t{ent.start_char}\t{ent.end_char}")


if __name__ == "__main__":
    main()