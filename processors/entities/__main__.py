#!/usr/bin/env python3
"""
Entry point for legal-text NER extraction from metadata CSV and local docs directory,
writes per-document tag CSVs.

Usage:
    python -m processors.entities \
        --input-csv METADATA_CSV \
        [--docs-dir DOCS_DIR] \
        [--use-pretrained]

The metadata CSV must include a 'doc_id' column. For each doc_id, the script will load
"{docs_dir}/{doc_id}.txt" and write tags to "{docs_dir}/{doc_id}_tags.csv".

Logging:
    Logs are written to the file specified by LOG_FILE env (default: ner_processing.log).
"""
import sys
import os
import argparse
import csv
import logging
import spacy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
LOG_FILE = os.getenv('LOG_FILE', 'ner_processing.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(module)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Register custom components
import processors.entities.person
import processors.entities.date
import processors.entities.case
import processors.entities.role
import processors.entities.doctype
import processors.entities.crime


def build_pipeline(use_pretrained: bool = False):
    model = "uk_core_news_sm" if use_pretrained else None
    nlp = spacy.load(model) if model else spacy.blank("uk")
    components = [
        "person_component",
        "date_component",
        "case_component",
        "role_component",
        "doctype_component",
        "crime_component",
    ]
    for name in components:
        nlp.add_pipe(name, last=True)
    logger.info("Pipeline built (pretrained=%s) with components: %s", use_pretrained, nlp.pipe_names)
    return nlp


def process_document(nlp, docs_dir: str, doc_id: str, text: str):
    """Process a single document text and write tags CSV."""
    logger.info("Processing doc_id=%s (length=%d)", doc_id, len(text))
    doc = nlp(text)
    tags_path = os.path.join(docs_dir, f"{doc_id}_tags.csv")
    try:
        with open(tags_path, 'w', newline='', encoding='utf-8') as tagfile:
            writer = csv.writer(tagfile)
            writer.writerow(['label', 'text', 'start_char', 'end_char'])
            for ent in doc.ents:
                logger.info("Tagging %s: '%s' [%d-%d] in doc_id=%s",
                            ent.label_, ent.text, ent.start_char, ent.end_char, doc_id)
                writer.writerow([ent.label_, ent.text, ent.start_char, ent.end_char])
    except Exception as e:
        logger.error("Failed writing tags for %s: %s", doc_id, e)


def main():
    parser = argparse.ArgumentParser(description="NER from metadata and docs directory, per-doc CSV output")
    parser.add_argument("--input-csv", required=True,
                        help="Path to metadata CSV with 'doc_id' column")
    parser.add_argument("--docs-dir", default="docs",
                        help="Directory containing text files named '<doc_id>.txt' and for tags output")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="Use pretrained spaCy model instead of blank model")
    args = parser.parse_args()

    meta_path = args.input_csv
    docs_dir = args.docs_dir
    if not os.path.isfile(meta_path):
        logger.error("Metadata CSV not found: %s", meta_path)
        sys.exit(f"Error: metadata CSV not found at {meta_path}")
    if not os.path.isdir(docs_dir):
        logger.error("Docs directory not found: %s", docs_dir)
        sys.exit(f"Error: docs directory not found at {docs_dir}")

    nlp = build_pipeline(use_pretrained=args.use_pretrained)

    with open(meta_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if 'doc_id' not in reader.fieldnames:
            logger.error("'doc_id' column missing in metadata CSV")
            sys.exit("Error: 'doc_id' column missing in metadata CSV")
        for row in reader:
            doc_id = row['doc_id']
            text_file = os.path.join(docs_dir, f"{doc_id}.txt")
            if not os.path.isfile(text_file):
                logger.warning("Text file missing for doc_id=%s: %s", doc_id, text_file)
                continue
            try:
                with open(text_file, encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                logger.error("Failed reading %s: %s", text_file, e)
                continue
            process_document(nlp, docs_dir, doc_id, text)

if __name__ == "__main__":
    main()
