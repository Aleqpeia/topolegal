"""
CSV Custom Parser Module

This module processes CSV files with any structure and ingests them into BigQuery.
Table names are automatically generated from filenames, and schemas are inferred from data.
"""

from .parser import GenericCsvParser
from .models import CsvRecord, SchemaInference

__all__ = ['GenericCsvParser', 'CsvRecord', 'SchemaInference'] 