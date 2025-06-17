# Court Document Processing

This project processes Ukrainian court documents as RTF files using spaCy NLP to extract entities and analyze text.



## Setup

1. Install dependencies using uv (recommended) or pip:
    ```bash
    uv install
    ```
2. Install transformer model via spacy:
    ```bash
    python -m spacy download uk_core_news_trf
    ```
3. Run BigQuery workload:
    ```bash
    python -m processors.entities bigquery --use-pretrained --batch 100 
    ```
## Environment variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to the Google Cloud service account JSON file.
- `GOOGLE_CLOUD_PROJECT`: ID of the Google Cloud project.
- `BQ_TABLE`: ID of the BigQuery table.

prefered to use dotenv via local processors.entities .env file to set these variables.



