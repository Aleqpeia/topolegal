"""
Entry point for legal-text NER extraction using custom spaCy components.

Usage:
    python -m processors.entities [INPUT_FILE]
    cat file.txt | python -m processors.entities
"""
import sys
import os
from google.cloud import bigquery
import spacy

# Register all custom components (module-level factories)
import processors.entities.person
import processors.entities.date
import processors.entities.case
import processors.entities.role
import processors.entities.doctype
import processors.entities.crime


def build_pipeline(use_pretrained: bool = False):
    if use_pretrained:
        # Load a pretrained model (ensure it's installed)
        nlp = spacy.load("uk_core_news_sm")
    else:
        nlp = spacy.blank("uk")
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


def check_flag_present(table_ref: str, doc_id: str) -> bool:
    client = bigquery.Client()
    query = f"SELECT flag FROM `{table_ref}` WHERE doc_id = @doc_id LIMIT 1"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id)
        ]
    )
    query_job = client.query(query, job_config=job_config)
    result = query_job.result()
    for row in result:
        return bool(row.flag)
    return False


def main():
    # Parse arguments
    args = sys.argv[1:]
    doc_id = None
    input_text = None

    if len(args) == 0:
        print("Usage: python -m processors.entities [INPUT_FILE] [DOC_ID]")
        sys.exit(1)

    # Determine doc_id (last arg) and optional input file
    if len(args) >= 2 and os.path.isfile(args[0]):
        input_text = open(args[0], encoding="utf-8").read()
        doc_id = args[1]
    else:
        # read from stdin, first arg is doc_id
        input_text = sys.stdin.read()
        doc_id = args[0]

    # BigQuery table reference from env or default
    table_ref = os.getenv("BQ_TABLE", "project.dataset.flags_table")

    # Skip processing if flag present
    if check_flag_present(table_ref, doc_id):
        print(f"FLAG_PRESENT for doc_id={doc_id}, skipping...")
        sys.exit(0)

    # Process NLP
    nlp = build_pipeline(use_pretrained=False)
    doc = nlp(input_text)

    # Output TSV: label, text, start, end
    for ent in doc.ents:
        print(f"{ent.label_}\t{ent.text}\t{ent.start_char}\t{ent.end_char}")


if __name__ == "__main__":
    main()
