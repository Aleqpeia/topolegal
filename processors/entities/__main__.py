#!/usr/bin/env python3
"""Topolegal NER runner – selectable backend
===========================================

Modes
-----
1. **postgres**  – stream rows from a PostgreSQL table, fill `text` & `ner_tags`.
2. **bigquery**  – same logic but uses Google BigQuery (*UPDATE … WHERE text IS NULL*).
3. **test**      – single document (URL or local .rtf/.txt) rendered with **displacy** to HTML for manual review.

Usage examples
--------------
```bash
# PostgreSQL
python -m processors.entities --mode postgres \
       --dsn "dbname=legal user=etl password=*** host=127.0.0.1" \
       --table documents --batch 1000 --use-pretrained

# BigQuery (service account picked up from $GOOGLE_APPLICATION_CREDENTIALS)
python -m processors.entities --mode bigquery \
       --bq-table project.dataset.documents --batch 1000 --use-pretrained

# Local test of one RTF via URL
python -m processors.entities --mode test \
       --source "https://bucket/path/102383644.rtf" --html out.html
```
Dependencies
------------
```bash
pip install psycopg[binary] google-cloud-bigquery requests striprtf ftfy spacy \
            python-dotenv
```
"""
from __future__ import annotations
import argparse, json, logging, os, re, unicodedata, html, sys, tempfile, pathlib
from typing import Iterable

import ftfy, requests, spacy
from striprtf.striprtf import rtf_to_text as striprtf_to_text
from spacy.util import filter_spans
from dotenv import load_dotenv
from spacy import displacy
from gliner_spacy.pipeline import GlinerSpacy

# --------------------------- cleaning helpers ------------------------------
_RE = lambda p: re.compile(p)
NBSP = "\u00A0"
RE_HYPH  = _RE(r"-[\r\n]+")
RE_EOL   = _RE(r"[\r\n]+")
RE_MULTI = _RE(r"[ \t]{2,}")
RE_ZW    = _RE(r"[\u200B-\u200D\u2060]")
RE_GLUE  = _RE(r"(?<=[а-яіїєґ])(?=[А-ЯІЇЄҐ])")
RE_NUMAL = _RE(r"(?<=\d)(?=[A-Za-zА-Яа-яІЇЄҐ])")

_DEF_TIMEOUT = (10, 30)


def clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = RE_HYPH.sub("", text)
    text = RE_EOL.sub(" ", text)
    text = text.replace(NBSP, " ")
    text = RE_ZW.sub("", text)
    text = RE_GLUE.sub(" ", text)
    text = RE_NUMAL.sub(" ", text)
    text = RE_MULTI.sub(" ", text)
    return text.strip()

# --------------------------- spaCy pipeline -------------------------------
import processors.entities.person   # noqa: F401
import processors.entities.date     # noqa: F401
import processors.entities.case     # noqa: F401
import processors.entities.role     # noqa: F401
import processors.entities.doctype  # noqa: F401
import processors.entities.crime    # noqa: F401

RULE_COMPONENTS = [
    "person_component", "date_component",
    "role_component", "doctype_component", "crime_component"]

LABELS = ["PERSON", "DATE",  "ROLE", "DOCTYPE", "CRIME"]


def build_nlp(use_pretrained: bool):
    nlp = spacy.load("uk_core_news_trf") if use_pretrained else spacy.blank("uk")
    nlp.add_pipe("gliner_spacy")
    if not use_pretrained:
        ner = nlp.add_pipe("ner", last=False)
        for label in LABELS:
            ner.add_label(label)
    for name in RULE_COMPONENTS:
        if name not in nlp.pipe_names:
            nlp.add_pipe(name, last=True)
    return nlp

# --------------------------- IO helpers ------------------------------------

def rtf_to_plain(data: bytes) -> str:
    try:
        return striprtf_to_text(data.decode("latin-1", errors="ignore"))
    except Exception:
        return ""


def fetch_rtf(url_or_path: str) -> bytes:
    if url_or_path.startswith("http"):
        r = requests.get(url_or_path, timeout=_DEF_TIMEOUT)
        r.raise_for_status()
        return r.content
    return pathlib.Path(url_or_path).read_bytes()

# --------------------------- backend: PostgreSQL ---------------------------

def run_postgres(dsn: str, table: str, batch: int, nlp):
    import psycopg

    fetch = f"SELECT doc_id, doc_url FROM {table} WHERE text IS NULL LIMIT %s"
    update = f"UPDATE {table} SET text = %s, ner_tags = %s WHERE doc_id = %s"

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            session = requests.Session(); session.headers["UA"] = "topolegal"
            total = 0
            while True:
                cur.execute(fetch, (batch,))
                rows = cur.fetchall()
                if not rows:
                    break
                for doc_id, url in rows:
                    try:
                        plain = clean(rtf_to_plain(fetch_rtf(url)))
                        tags_json = analyse(plain, nlp)
                        cur.execute(update, (plain, tags_json, doc_id))
                        total += 1
                    except Exception as e:
                        logging.error("doc %s: %s", doc_id, e)
                conn.commit(); logging.info("PG processed %d", total)

# --------------------------- backend: BigQuery -----------------------------

def run_bigquery(table: str, batch: int, nlp):
    from google.cloud import bigquery
    client = bigquery.Client()
    total  = 0
    while True:
        rows = client.query(f"""
            SELECT doc_id, doc_url FROM `{table}`
            WHERE text IS NULL
            LIMIT {batch}
        """).result().to_dataframe()
        if rows.empty:
            break
        for _, r in rows.iterrows():
            try:
                plain = clean(rtf_to_plain(fetch_rtf(r.doc_url)))
                tags_json = analyse(plain, nlp)
                client.query(
                    f"""
                    UPDATE `{table}`
                       SET text = @text,
                           ner_tags = @tags
                     WHERE doc_id = @id
                    """,
                    job_config=bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter("text", "STRING", plain),
                            bigquery.ScalarQueryParameter("tags", "STRING", tags_json),
                            bigquery.ScalarQueryParameter("id",   "STRING", r.doc_id),
                        ]
                    ),
                ).result()
                total += 1
            except Exception as e:
                logging.error("BQ doc %s: %s", r.doc_id, e)
        logging.info("BQ processed %d", total)

# --------------------------- backend: test mode ---------------------------

def run_test(source: str, nlp, html_out: str | None):
    plain = clean(rtf_to_plain(fetch_rtf(source)))
    doc   = nlp(plain); doc.ents = filter_spans(doc.ents)
    print("Entities:\n" + "\n".join(f"{e.label_}\t{e.text}" for e in doc.ents))
    if html_out:
        html = displacy.render(doc, style="ent", page=True)
        pathlib.Path(html_out).write_text(html, encoding="utf-8")
        logging.info("HTML saved → %s", html_out)

# --------------------------- common analyse -------------------------------

def analyse(text: str, nlp):
    doc = nlp(text); doc.ents = filter_spans(doc.ents)
    return json.dumps([
        {"label": e.label_, "text": e.text, "start": e.start_char, "end": e.end_char}
        for e in doc.ents
    ], ensure_ascii=False)

# --------------------------- CLI ------------------------------------------

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser("topolegal multipurpose NER runner")
    sub = ap.add_subparsers(dest="mode", required=True)

    pg = sub.add_parser("postgres")
    pg.add_argument("--dsn", required=True)
    pg.add_argument("--table", required=True)
    pg.add_argument("--batch", type=int, default=1000)

    bq = sub.add_parser("bigquery")
    bq.add_argument("--bq-table", required=True)
    bq.add_argument("--batch", type=int, default=1000)

    ts = sub.add_parser("test")
    ts.add_argument("--source", required=True, help="RTF URL or local path")
    ts.add_argument("--html", help="optional output html")

    ap.add_argument("--use-pretrained", action="store_true")
    args = ap.parse_args()

    nlp = build_nlp(args.use_pretrained)

    if args.mode == "postgres":
        run_postgres(args.dsn, args.table, args.batch, nlp)
    elif args.mode == "bigquery":
        run_bigquery(args.bq_table, args.batch, nlp)
    else:  # test
        run_test(args.source, nlp, args.html)

if __name__ == "__main__":
    main()
