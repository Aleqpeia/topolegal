from pathlib import Path
from google.cloud import bigquery, storage


PROJECT   = "my-ml-demo"
DATASET   = "my_dataset"
TABLE     = "documents"
BUCKET    = "my-doc-bucket"
PREFIX    = "docs"               # gs://my-doc-bucket/docs/{doc_id}.rtf
OUT_DIR   = Path("data/raw")     # local cache

# ----- BigQuery --------------------------------------------------------------
bq = bigquery.Client(project=PROJECT)
sql = f"SELECT doc_id FROM `{PROJECT}.{DATASET}.{TABLE}`"
doc_ids = [row.doc_id for row in bq.query(sql).result()]

# ----- Cloud Storage ---------------------------------------------------------
gs = storage.Client(project=PROJECT)
bucket = gs.bucket(BUCKET)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for doc_id in doc_ids:
    blob = bucket.blob(f"{PREFIX}/{doc_id}.rtf")
    dest = OUT_DIR / f"{doc_id}.rtf"
    if not dest.exists():
        blob.download_to_filename(dest)
        print("✔", dest)
