#!/usr/bin/env python3
"""NER runner that writes results back into SQLite
=================================================

* Reads **documents.db** (table `docs` must have `doc_id`, `text`).
* Adds column `ner_tags` (TEXT) if missing.
* Runs spaCy pipeline (statistical model + rule‑based matcher components).
* Serialises extracted entities as compact JSON and stores in `ner_tags`.

Example:
    python -m processors.entities --db documents.db --use-pretrained
"""
import argparse
import json
import logging
import os
import sqlite3
import sys

import spacy
from spacy.util import filter_spans
from dotenv import load_dotenv

# ─────────────────────────── logging / env ──────────────────────────────────
load_dotenv()
logging.basicConfig(
    filename=os.getenv("LOG_FILE", "ner_processing.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(module)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────── register rule components ──────────────────────────
import processors.entities.person   # noqa: F401
import processors.entities.date     # noqa: F401
import processors.entities.case     # noqa: F401
import processors.entities.role     # noqa: F401
import processors.entities.doctype  # noqa: F401
import processors.entities.crime    # noqa: F401

RULE_COMPONENTS = [
    "person_component",
    "date_component",
    "case_component",
    "role_component",
    "doctype_component",
    "crime_component",
]
LABELS = ["PERSON", "DATE", "CASE_NUMBER", "ROLE", "DOCTYPE", "CRIME"]

# ────────────────────────── pipeline builder ───────────────────────────────

def build_pipeline(pretrained: bool) -> spacy.language.Language:
    if pretrained:
        nlp = spacy.load("uk_core_news_sm")
        logger.info("Loaded pretrained uk_core_news_sm: pipes=%s", nlp.pipe_names)
    else:
        nlp = spacy.blank("uk")
        ner = nlp.add_pipe("ner", last=False)
        for lb in LABELS:
            ner.add_label(lb)
        logger.info("Blank model + NER labels initialised")

    for name in RULE_COMPONENTS:
        if name not in nlp.pipe_names:
            nlp.add_pipe(name, last=True)
    logger.info("Pipeline ready: %s", nlp.pipe_names)
    return nlp

# ────────────────────────── DB helpers ──────────────────────────────────────

def ensure_ner_column(cur: sqlite3.Cursor):
    cur.execute("PRAGMA table_info(docs)")
    cols = [row[1] for row in cur.fetchall()]
    if "ner_tags" not in cols:
        cur.execute("ALTER TABLE docs ADD COLUMN ner_tags TEXT")
        logger.info("Added ner_tags column to docs table")

# ─────────────────────────────── main ───────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("NER → SQLite column updater")
    ap.add_argument("--db", required=True, help="path to documents.db")
    ap.add_argument("--use-pretrained", action="store_true", help="load uk_core_news_sm")
    args = ap.parse_args()

    if not os.path.isfile(args.db):
        sys.exit(f"SQLite DB not found: {args.db}")

    nlp = build_pipeline(args.use_pretrained)

    con = sqlite3.connect(args.db)
    cur = con.cursor()
    ensure_ner_column(cur)

    rows = cur.execute("SELECT doc_id, text FROM docs").fetchall()
    total = len(rows)
    logger.info("Processing %d documents", total)

    for doc_id, text in rows:
        logger.info("doc_id=%s length=%d", doc_id, len(text))
        doc = nlp(text)
        doc.ents = filter_spans(doc.ents)
        tags = [
            {
                "label": e.label_,
                "text": e.text,
                "start": e.start_char,
                "end": e.end_char,
            }
            for e in doc.ents
        ]
        cur.execute(
            "UPDATE docs SET ner_tags = ? WHERE doc_id = ?",
            (json.dumps(tags, ensure_ascii=False), doc_id),
        )
    con.commit()
    con.close()
    logger.info("NER tags stored for %d docs", total)


if __name__ == "__main__":
    main()
