#!/usr/bin/env python3
"""preproc.py – robust CSV+RTF→SQLite pre-processor
====================================================

Fixes earlier issues (duplicate columns, bad RTF decoding) and handles
source files that look like:
    {\rtf1\ansi... \'d1\'ef\'f0...}

Key points
----------
* **RTF to UTF-8 text**
  1. `striprtf` first (pure-python, understands codepages)
  2. `pypandoc` fallback
  3. `unrtf --text` fallback
  4. raw read as last resort.
* **TXT input** – detect encoding with `chardet` and decode correctly.
* **Cleaning** – joins hyphen breaks, normalises spaces, inserts missing
  space between glued Cyrillic/Latin/digit runs, removes RTF artefacts &
  control chars.
* **SQLite schema**
  `doc_id` PRIMARY KEY, all **other** CSV columns, plus `text`.
* **Upsert** with `INSERT OR REPLACE`.

Install deps once:
    pip install striprtf chardet ftfy pypandoc python-dotenv
    sudo apt-get install unrtf        # optional fallback

Run:
    python preproc.py --csv document_data.csv --docs docs --db documents.db
"""
from __future__ import annotations
import argparse, csv, logging, re, unicodedata, html, subprocess, sqlite3, sys, os, io
from pathlib import Path
from typing import List

import ftfy               # fix mojibake
import chardet            # detect txt encodings
from striprtf.striprtf import rtf_to_text as striprtf_to_text  # pure-python

# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
NBSP = "\u00A0"
RE_CTRL  = re.compile(r"[\x00-\x08\x0b-\x1f]")
RE_RTF   = re.compile(r"{\\rtf[^}]+}", re.S)
RE_HYPH  = re.compile(r"-[\r\n]+")
RE_EOL   = re.compile(r"[\r\n]+")
RE_MULTI = re.compile(r"[ \t]{2,}")
RE_ZW    = re.compile(r"[\u200B-\u200D\u2060]")
RE_GLUE  = re.compile(r"(?<=[а-яіїєґ])(?=[А-ЯІЇЄҐ])")
RE_NUMAL = re.compile(r"(?<=\d)(?=[A-Za-zА-Яа-яІЇЄҐ])")

def clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = RE_RTF.sub(" ", text)           # stray RTF blocks after bad convert
    text = RE_HYPH.sub("", text)
    text = RE_EOL.sub(" ", text)
    text = text.replace(NBSP, " ")
    text = RE_ZW.sub("", text)
    text = RE_GLUE.sub(" ", text)
    text = RE_NUMAL.sub(" ", text)
    text = RE_MULTI.sub(" ", text)
    text = RE_CTRL.sub("", text)
    return text.strip()

# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_txt_with_detect(path: Path) -> str:
    data = path.read_bytes()
    guess = chardet.detect(data)
    enc = guess["encoding"] or "utf-8"
    try:
        return data.decode(enc, errors="ignore")
    except LookupError:
        return data.decode("utf-8", errors="ignore")

def rtf_to_text(path: Path) -> str:
    raw = path.read_bytes()
    # try striprtf first (handles cp1251 escape sequences)
    try:
        return striprtf_to_text(raw.decode("latin-1", errors="ignore"))
    except Exception:
        pass
    # fallback pypandoc
    try:
        import pypandoc
        return pypandoc.convert_text(raw, "plain", format="rtf")
    except Exception:
        pass
    # fallback unrtf
    try:
        res = subprocess.run(["unrtf", "--text", str(path)], capture_output=True, text=True, check=True)
        return res.stdout
    except Exception:
        pass
    # last resort: decode escape sequences manually (very slow but safe)
    try:
        text = raw.decode("latin-1", errors="ignore")
        # replace \'hh with bytes
        def sub(match):
            byte = bytes.fromhex(match.group(1))
            return byte.decode("cp1251", errors="ignore")
        text = re.sub(r"\\'([0-9a-fA-F]{2})", sub, text)
        # strip remaining RTF controls
        return striprtf_to_text(text)
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def ensure_table(cur: sqlite3.Cursor, csv_columns: List[str]):
    extra_cols = [c for c in csv_columns if c != "doc_id"]
    col_defs   = ",\n        ".join([f'"{c}" TEXT' for c in extra_cols])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS docs (
            doc_id TEXT PRIMARY KEY,
            {col_defs},
            text   TEXT
        )""")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build SQLite DB from CSV + docs (RTF/TXT)")
    ap.add_argument("--csv",  required=True, help="metadata CSV")
    ap.add_argument("--docs", default="docs", help="directory with <doc_id>.(rtf|txt)")
    ap.add_argument("--db",   default="documents.db", help="output sqlite file")
    args = ap.parse_args()

    csv_path = Path(args.csv); docs_dir = Path(args.docs); db_path = Path(args.db)
    if not csv_path.is_file():
        sys.exit(f"CSV not found: {csv_path}")
    docs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        ensure_table(cur, fieldnames)

        for i, row in enumerate(reader, 1):
            doc_id = row.get("doc_id")
            src = None
            for ext in (".rtf", ".txt"):
                p = docs_dir / f"{doc_id}{ext}"
                if p.is_file():
                    src = p; break
            if not src:
                logging.warning("[%d] missing source for %s", i, doc_id); continue

            try:
                raw = rtf_to_text(src) if src.suffix.lower() == ".rtf" else read_txt_with_detect(src)
                text = clean(raw)
            except Exception as e:
                logging.error("[%d] failed convert %s: %s", i, src.name, e); continue

            # build insert
            extra = [c for c in fieldnames if c != "doc_id"]
            placeholders = ", ".join(["?"] * (len(extra) + 1))  # +1 for text
            values = [row.get(c, "") for c in extra] + [text]
            cur.execute(
                f"INSERT OR REPLACE INTO docs (doc_id, {', '.join(extra)}, text) VALUES (?, {placeholders})",
                [doc_id] + values
            )
            if i % 100 == 0:
                con.commit()
                logging.info("processed %d rows", i)
        con.commit()
    con.close()
    logging.info("Done → %s", db_path)

if __name__ == "__main__":
    main()
