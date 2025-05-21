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
from typing import Dict

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

def annotate_links(raw: str) -> tuple[str, list[dict]]:
    """Return (plain_text, links).  links = [{url,text,start,end}, …]."""
    marked = _insert_tokens(raw)          # reuse your old logic

    out, links, i_plain, i_src = [], [], 0, 0
    for m in TOK_RE.finditer(marked):
        pre = marked[i_src:m.start()]
        out.append(pre); i_plain += len(pre)

        span = m[2]
        out.append(span)
        links.append({"url": m[1], "text": span,
                      "start": i_plain, "end": i_plain + len(span)})
        i_plain += len(span); i_src = m.end()

    out.append(marked[i_src:])
    return "".join(out), links



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
                plain, links = annotate_links(clean(rtf_to_plain(fetch_rtf(r.doc_url))))
                tags_json = analyse(plain, nlp)
                client.query(
                    f"""
                    UPDATE `{table}`
                       SET text = @text,
                           links = @links,
                           ner_tags = @tags
                     WHERE doc_id = @id
                    """,
                    job_config=bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter("text", "STRING", plain),
                            bigquery.ScalarQueryParameter("links", "STRING", links),
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

    pg = sub.add_parser("postgres")
    pg.add_argument("--dsn", required=True)
    pg.add_argument("--table", required=True)
    pg.add_argument("--batch", type=int, default=1000)

    bq = sub.add_parser("bigquery")
    bq.add_argument("--bq-table", required=True)
    bq.add_argument("--batch", type=int, default=1000)

    ts = sub.add_parser("test")
    ts.add_argument("--source", required=True, help="RTF/TXT path or URL")
    ts.add_argument("--port", type=int, default=8000, help="local port (default 8000)")

    ap.add_argument("--use-pretrained", action="store_true")
    args = ap.parse_args()

    nlp = build_nlp(args.use_pretrained)

    if args.mode == "postgres":
        # run_postgres(args.dsn, args.table, args.batch, nlp)
        raise NotImplementedError("Postgres backend kept unchanged in snippet")
    elif args.mode == "bigquery":
        # run_bigquery(args.bq_table, args.batch, nlp)
        raise NotImplementedError("BigQuery backend kept unchanged in snippet")
    else:  # test
        run_test(args.source, nlp, args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
