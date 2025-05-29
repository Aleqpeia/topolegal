#!/usr/bin/env python3
"""Topolegal NER runner – selectable backend  (v0.1.1)
====================================================

* postgres → batch update using psycopg
* bigquery → batch update using GCP BigQuery
* test      → one‑off preview served by **displaCy** on localhost

Article‑linker now recognises *any* code abbreviation (КПК, КК, ЦК, ЦПК …).
"""
from __future__ import annotations
import argparse, json, logging, html, unicodedata, pathlib, re
from typing import Dict, Counter
import os
import ftfy, requests, spacy
from dotenv import load_dotenv
from spacy import displacy
from spacy.util import filter_spans
from striprtf.striprtf import rtf_to_text as striprtf_to_text

# ───────────────────────── regex helpers ───────────────────────────────────

def _RE(p: str, f: int = 0) -> re.Pattern:  # short alias
    return re.compile(p, f)

# general cleaning
NBSP = "\u00A0"
RE_HYPH  = _RE(r"-[\r\n]+")
RE_EOL   = _RE(r"[\r\n]+")
RE_MULTI = _RE(r"[ \t]{2,}")
RE_ZW    = _RE(r"[\u200B-\u200D\u2060]")
RE_GLUE  = _RE(r"(?<=[а-яіїєґ])(?=[А-ЯІЇЄҐ])")
RE_NUMAL = _RE(r"(?<=\d)(?=[A-Za-zА-Яа-яІЇЄҐ])")

# link tokens & patterns
START, STOP = "⟪L|", "⟪/L⟫"
RE_URL  = _RE(r"(?:(?:https?://|www\.)[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)", re.I)
RE_ERDR = _RE(r"№\s*([0-9]{14,})")
# plural: ст.ст. 12, 13, 76-82, 280 ЦПК України
RE_MULTI_ART = _RE(r"ст\.ст\.\s*([0-9, \-–]+)\s+([A-ZА-ЯІЇЄҐ]{2,4})\s+України", re.I)
# singular (optional п./ч.)
RE_ART = _RE(r"(?:п\.\s*(\d+)\s+)?(?:ч\.\s*(\d+)\s+)?ст\.\s*(\d+)\s+([A-ZА-ЯІЇЄҐ]{2,4})\s+України", re.I)

# zakon.rada.gov.ua anchors – extend if needed
LAW_CODE_URLS: Dict[str, str] = {
    "КПК": "4651-17",  # кримінальний процесуальний
    "КК":  "2341-14",  # кримінальний
    "ЦК":  "435-15",   # цивільний
    "ЦПК": "1618-15",  # цивільний процесуальний
}


_DEF_TIMEOUT = (10, 30)

# ───────────────────────── text cleaning  ─────────────────────────────────

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




def _law_url(code: str, art: str) -> str:
    code = code.upper()
    law_id = LAW_CODE_URLS.get(code)
    if law_id:
        return f"https://zakon.rada.gov.ua/laws/show/{law_id}#n{art}"
    return f"https://zakon.rada.gov.ua/laws/main?query=ст.{art}%20{code}%20України"


def _insert_tokens(text: str) -> str:
    """Wrap URLs, ЄРДР ids, single & plural article refs with ⟪L|…⟫ … ⟪/L⟫."""

    # 1) bare URLs
    text = RE_URL.sub(lambda m: f"{START}{m[0]}⟫{m[0]}{STOP}", text)

    # 2) ЄРДР review
    text = RE_ERDR.sub(lambda m: f"{START}https://reyestr.court.gov.ua/Review/{m[1]}⟫№{m[1]}{STOP}", text)

    # 3) plural article list (run *before* singular to prevent overlap)
    def _multi(match: re.Match) -> str:
        raw_list, code = match.groups()
        segments = [s.strip() for s in raw_list.replace("–", "-").split(",")]
        linked = []
        for seg in segments:
            if "-" in seg:  # range 76-82
                first, last = [s.strip() for s in seg.split("-", 1)]
                url = _law_url(code, first)
                linked.append(f"{START}{url}⟫{seg}{STOP}")
            else:
                url = _law_url(code, seg)
                linked.append(f"{START}{url}⟫{seg}{STOP}")
        return f"ст.ст. {' , '.join(linked)} {code} України"

    text = RE_MULTI_ART.sub(_multi, text)

    # 4) singular article
    def _single(m: re.Match) -> str:
        p, ch, art, code = m.groups()
        url = _law_url(code, art)
        span_parts = []
        if p: span_parts.append(f"п.{p}")
        if ch: span_parts.append(f"ч.{ch}")
        span_parts.append(f"ст.{art}")
        span = " ".join(span_parts) + f" {code} України"
        return f"{START}{url}⟫{span}{STOP}"

    text = RE_ART.sub(_single, text)
    return text

# ───────────────────────── text utils ─────────────────────────────────────-

TOK_RE = re.compile(fr"{re.escape(START)}([^⟫]+)⟫(.*?){re.escape(STOP)}")
ART_ANCHOR_RE = re.compile(r"/laws/show/(?P<law>[0-9\-]+)#n(?P<art>[0-9]+)")

def annotate_links(raw: str) -> tuple[str, list[dict]]:
    """Return *(plain_text, links)* with inline code/art if applicable."""
    marked = _insert_tokens(raw)
    plain_parts: list[str] = []
    links: list[dict] = []
    p_idx = s_idx = 0
    for m in TOK_RE.finditer(marked):
        pre = marked[s_idx:m.start()]; plain_parts.append(pre); p_idx += len(pre)
        url, span = m[1], m[2]; start, end = p_idx, p_idx+len(span)
        rec = {"url": url, "text": span, "start": start, "end": end}
        ac = ART_ANCHOR_RE.search(url)
        if ac:
            law, art = ac.group("law"), ac.group("art")
            for code,lid in LAW_CODE_URLS.items():
                if lid==law:
                    rec.update({"code": code, "art": art}); break
        links.append(rec); plain_parts.append(span); p_idx=end; s_idx=m.end()
    plain_parts.append(marked[s_idx:])
    return "".join(plain_parts), links


# ───────────────────────── spaCy pipeline  ────────────────────────────────
import processors.entities.person    # noqa: F401
import processors.entities.date      # noqa: F401
import processors.entities.case      # noqa: F401
import processors.entities.role      # noqa: F401
import processors.entities.doctype   # noqa: F401
import processors.entities.crime     # noqa: F401
import processors.entities.adress
import processors.entities.info
import processors.entities.number


RULE_COMPONENTS = [
    "person_component", "date_component", "role_component",
    "doctype_component", "crime_component", "adress_component",
    "info_component", "number_component",
]
LABELS = ["INFO", "DATE", "LOC" "ROLE", "DOCTYPE", "CRIME", "NUM"]

def build_nlp(use_pretrained: bool):
    nlp = spacy.load("uk_core_news_trf") if use_pretrained else spacy.blank("uk")
    if not use_pretrained:
        ner = nlp.add_pipe("ner", last=False)
        for label in LABELS:
            ner.add_label(label)
    for comp in RULE_COMPONENTS:
        if comp not in nlp.pipe_names:
            nlp.add_pipe(comp, last=True)
    return nlp

# ───────────────────────── IO helpers  ────────────────────────────────────

def fetch_rtf(src: str) -> bytes:
    if src.startswith("http"):
        r = requests.get(src, timeout=_DEF_TIMEOUT)
        r.raise_for_status()
        return r.content
    return pathlib.Path(src).read_bytes()


def rtf_to_plain(data: bytes) -> str:
    try:
        return striprtf_to_text(data.decode("latin-1", errors="ignore"))
    except Exception:
        return ""

# --------------------------- backend: PostgreSQL ---------------------------

def run_postgres(dsn: str, table: str, batch: int, nlp):
    import psycopg

    fetch_q = f"SELECT doc_id, doc_url FROM {table} WHERE text IS NULL LIMIT %s"
    upd_q   = f"UPDATE {table} SET text = %s, links = %s, ner_tags = %s WHERE doc_id = %s"

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            total = 0
            while True:
                cur.execute(fetch_q, (batch,))
                rows = cur.fetchall()
                if not rows:
                    break
                for doc_id, url in rows:
                    try:
                        plain, links = annotate_links(clean(rtf_to_plain(fetch_rtf(url))))
                        tags_json = analyse(plain, nlp)
                        cur.execute(upd_q, (plain, links, tags_json, doc_id))
                        total += 1
                    except Exception as e:
                        logging.error("PG doc %s: %s", doc_id, e)
                conn.commit(); logging.info("PG processed %d", total)

# --------------------------- backend: BigQuery -----------------------------

from google.cloud import bigquery
import logging

DEST = "lab-test-project-1-305710.court_data_2022.sentences_only"   # new table
SRC  = "lab-test-project-1-305710.court_data_2022.document_data"             # original

def run_bigquery(src: str, dest: str, batch: int, nlp,
                 project: str | None = None, key_path: str | None = None):
    # honour CLI / .env overrides
    if key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    client = bigquery.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"], location="us-central1")

    total  = 0
    while True:
        # Pull the next batch of unprocessed rows
        sql = f"""
            SELECT  src.doc_id,
                    src.doc_url   
            FROM    `{SRC}`  AS src
            LEFT JOIN `{DEST}` AS dst
                ON dst.doc_id = src.doc_id
            WHERE dst.doc_id IS NULL                      -- not processed yet
            LIMIT {batch}
        """
        job = client.query(sql)
        logging.info("Batch job %s submitted", job.job_id)
        rows = job.result().to_dataframe()

        if rows.empty:
            break
        for _, r in rows.iterrows():
            try:
                plain, links = annotate_links(clean(rtf_to_plain(fetch_rtf(r.doc_url))))
                tags    = analyse(plain, nlp)

                # Stream the processed record into the destination table
                client.insert_rows_json(
                    DEST,
                    [{
                        "doc_id": r.doc_id,
                        "text":  plain,
                        "links": json.dumps(links, ensure_ascii=False),
                        "tags":  tags,
                    }],
                    row_ids=[str(r.doc_id)]          # idempotent insert
                )

                # Optionally mark the source row as done so we never re-process it
                # client.query(
                #     f"""
                #     UPDATE `{DEST}`
                #        SET text = @text AND doc_id = @r.doc_id
                #        WHERE doc_id is NULL
                #     """,
                #     job_config=bigquery.QueryJobConfig(
                #         query_parameters=[
                #             bigquery.ScalarQueryParameter("tags", "STRING", tags),
                #             bigquery.ScalarQueryParameter("links", "STRING", links),
                #             bigquery.ScalarQueryParameter("text", "STRING", plain),
                #             bigquery.ScalarQueryParameter("doc_id",   "INTEGER", r.doc_id),
                #         ]
                #     ),
                # ).result()

                total += 1

            except Exception as e:
                logging.error("BQ doc %s: %s", r.doc_id, e)

        logging.info("processed %d rows", total)
        


# --------------------------- backend: test mode ---------------------------
# 234 CK - text + pos

def run_test(source: str, nlp, port: int):
    """Serve a single annotated document on http://127.0.0.1:<port>."""
    plain, links = annotate_links(clean(rtf_to_plain(fetch_rtf(source))))
    logging.info(plain, links)
    print(plain,links)
    doc = nlp(plain); doc.ents = filter_spans(doc.ents)

    logging.info("Starting displaCy at http://127.0.0.1:%d — press Ctrl+C to stop", port)
    displacy.serve(doc, style="ent", page=True, host="127.0.0.1", port=port)

# ───────────────────────── common NER JSON  ───────────────────────────────

def analyse(text: str, nlp):
    doc = nlp(text); doc.ents = filter_spans(doc.ents)
    return json.dumps([
        {"label": e.label_, "text": e.text, "start": e.start_char, "end": e.end_char}
        for e in doc.ents
    ], ensure_ascii=False)

# ───────────────────────── CLI  ───────────────────────────────────────────


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser("topolegal multipurpose NER runner")
    sub = ap.add_subparsers(dest="mode", required=True)

    # ─────────── Postgres ───────────
    pg = sub.add_parser("postgres", help="Process a PostgreSQL table")
    pg.add_argument("--dsn",
                    default=os.getenv("PG_DSN"),
                    required=os.getenv("PG_DSN") is None,
                    help="PostgreSQL DSN (falls back to PG_DSN)")
    pg.add_argument("--table",
                    default=os.getenv("PG_TABLE"),
                    required=os.getenv("PG_TABLE") is None,
                    help="Table name (falls back to PG_TABLE)")
    pg.add_argument("--batch", type=int, default=1000)

    # ─────────── BigQuery ───────────
    bq = sub.add_parser("bigquery", help="Process a BigQuery table")
    bq.add_argument("--bq-table",
                    default=os.getenv("BQ_TABLE"),
                    required=os.getenv("BQ_TABLE") is None,
                    help="project.dataset.table (falls back to BQ_TABLE)")
    bq.add_argument("--gcp-project",
                    default=os.getenv("GOOGLE_CLOUD_PROJECT"),
                    help="GCP project id (falls back to GOOGLE_CLOUD_PROJECT)")
    bq.add_argument("--gcp-key",
                    default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                    help="Path to service-account JSON key "
                         "(falls back to GOOGLE_APPLICATION_CREDENTIALS)")
    bq.add_argument("--batch", type=int, default=1000)

    # ─────────── Test server ───────────
    ts = sub.add_parser("test")
    ts.add_argument("--source", required=True, help="RTF/TXT path or URL")
    ts.add_argument("--port", type=int, default=8000)

    ap.add_argument("--use-pretrained", action="store_true")
    args = ap.parse_args()
    print(args)
    nlp = build_nlp(args.use_pretrained)

    if args.mode == "postgres":
        run_postgres(args.dsn, args.table, args.batch, nlp)
        raise NotImplementedError("Postgres backend kept unchanged in snippet")
    elif args.mode == "bigquery":
        run_bigquery(SRC, DEST, args.batch, nlp)

    else:  # test
        run_test(args.source, nlp, args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
