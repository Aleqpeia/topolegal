"""
Generic data models for CSV parsing
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import re


@dataclass
class CsvRecord:
    """Generic CSV record that can handle any structure"""
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for BigQuery insertion"""
        return self.data.copy()
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> 'CsvRecord':
        """Create CsvRecord from CSV row data"""
        cleaned_data = {}
        
        for key, value in row.items():
            if key and value is not None:
                # Clean column name for BigQuery compatibility
                clean_key = cls._clean_column_name(key)
                # Clean and convert value
                clean_value = cls._clean_value(value)
                cleaned_data[clean_key] = clean_value
        
        return cls(data=cleaned_data)
    
    @staticmethod
    def _clean_column_name(name: str) -> str:
        """Clean column name to be BigQuery compatible"""
        # Remove quotes and whitespace
        cleaned = name.strip().strip('"').strip("'")
        
        # Replace spaces and special characters with underscores
        cleaned = re.sub(r'[^\w\d]', '_', cleaned)
        
        # Ensure it starts with a letter or underscore
        if cleaned and not cleaned[0].isalpha() and cleaned[0] != '_':
            cleaned = f"col_{cleaned}"
        
        # Ensure it's not empty
        if not cleaned:
            cleaned = "unnamed_column"
            
        return cleaned.lower()
    
    @staticmethod
    def _clean_value(value: str) -> Any:
        """Clean and convert value to appropriate type"""
        if not value or value.strip() == '':
            return None
            
        # Remove quotes and whitespace
        cleaned = value.strip().strip('"').strip("'")
        
        # Try to convert to integer
        try:
            return int(cleaned)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(cleaned)
        except ValueError:
            pass
        
        # Return as string
        return cleaned


class SchemaInference:
    """Helper class to infer BigQuery schema from CSV data"""
    
    @staticmethod
    def infer_column_type(values: List[Any]) -> str:
        """Infer BigQuery column type from a list of values"""
        non_null_values = [v for v in values if v is not None]
        
        if not non_null_values:
            return "STRING"
        
        # Check if all values are integers
        if all(isinstance(v, int) for v in non_null_values):
            return "INTEGER"
        
        # Check if all values are numbers (int or float)
        if all(isinstance(v, (int, float)) for v in non_null_values):
            return "FLOAT"
        
        # Default to string
        return "STRING"
    
    @classmethod
    def infer_schema(cls, records: List[CsvRecord]) -> List[Dict[str, str]]:
        """Infer BigQuery schema from a list of CSV records"""
        if not records:
            return []
        
        # Get all column names
        all_columns = set()
        for record in records:
            all_columns.update(record.data.keys())
        
        schema = []
        for column in sorted(all_columns):
            # Collect all values for this column
            column_values = [record.data.get(column) for record in records]
            
            # Infer type
            column_type = cls.infer_column_type(column_values)
            
            # Determine if column can be nullable
            has_nulls = any(v is None for v in column_values)
            mode = "NULLABLE" if has_nulls else "REQUIRED"
            
            schema.append({
                "name": column,
                "type": column_type,
                "mode": mode
            })
        
        return schema 