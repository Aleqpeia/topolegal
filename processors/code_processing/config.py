import os
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class Config:
    """Configuration for legal code parser"""
    
    # BigQuery settings
    BIGQUERY_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "your-project-id")
    BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "legal_codes")
    BIGQUERY_TABLE_ID = os.getenv("BIGQUERY_TABLE_ID", "criminal_process_code")
    
    # Paths
    BASE_PATH = Path(__file__).parent
    CODES_PATH = BASE_PATH / "codes"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Spacy model (placeholder)
    SPACY_MODEL = os.getenv("SPACY_MODEL", "uk_core_news_sm") 