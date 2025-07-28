from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel
import json
import logging
import re
import random
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class CorruptionResult(BaseModel):
    """Result model for corruption operation"""
    original_data: Dict
    corrupted_data: Dict
    corruption_applied: bool
    corruption_type: str
    corruption_details: Dict


class DataCorruptor(ABC):
    """
    Abstract base class for data corruption operations.
    Used to create datasets with corrupted samples for binary classification tasks.
    """
    
    def __init__(self, corruption_probability: float = 0.5):
        """
        Initialize the data corruptor.
        
        Args:
            corruption_probability: Probability of corrupting each sample (0.0 to 1.0)
        """
        if not 0.0 <= corruption_probability <= 1.0:
            raise ValueError("Corruption probability must be between 0.0 and 1.0")
        
        self.corruption_probability = corruption_probability
        self.corruption_stats = {
            'total_processed': 0,
            'corrupted': 0,
            'skipped': 0,
            'errors': 0
        }
    
    @abstractmethod
    def corrupt_sample(self, sample: Dict) -> CorruptionResult:
        """
        Corrupt a single data sample.
        
        Args:
            sample: Original data sample
            
        Returns:
            CorruptionResult containing original and corrupted data
        """
        pass
    
    @abstractmethod
    def can_corrupt(self, sample: Dict) -> bool:
        """
        Check if a sample can be corrupted.
        
        Args:
            sample: Data sample to check
            
        Returns:
            True if sample can be corrupted, False otherwise
        """
        pass
    
    def corrupt_dataset(self, dataset: List[Dict], label_column: str = 'is_corrupted') -> List[Dict]:
        """
        Corrupt a dataset by applying corruption to selected samples.
        
        Args:
            dataset: List of data samples
            label_column: Column name for corruption labels
            
        Returns:
            List of samples with corruption labels
        """
        corrupted_dataset = []
        
        for sample in dataset:
            self.corruption_stats['total_processed'] += 1
            
            try:
                # Decide whether to corrupt this sample
                should_corrupt = (
                    self.can_corrupt(sample) and 
                    random.random() < self.corruption_probability
                )
                
                if should_corrupt:
                    # Apply corruption
                    result = self.corrupt_sample(sample)
                    
                    if result.corruption_applied:
                        # Add corrupted version with label
                        corrupted_sample = result.corrupted_data.copy()
                        corrupted_sample[label_column] = False  # Corrupted = False
                        corrupted_dataset.append(corrupted_sample)
                        self.corruption_stats['corrupted'] += 1
                    else:
                        # Corruption failed, keep original
                        original_sample = sample.copy()
                        original_sample[label_column] = True  # Original = True
                        corrupted_dataset.append(original_sample)
                        self.corruption_stats['skipped'] += 1
                else:
                    # Keep original with label
                    original_sample = sample.copy()
                    original_sample[label_column] = True  # Original = True
                    corrupted_dataset.append(original_sample)
                    self.corruption_stats['skipped'] += 1
                    
            except Exception as e:
                logger.error(f"Error corrupting sample {sample.get('doc_id', 'unknown')}: {e}")
                # Keep original on error
                original_sample = sample.copy()
                original_sample[label_column] = True
                corrupted_dataset.append(original_sample)
                self.corruption_stats['errors'] += 1
        
        return corrupted_dataset
    
    def get_corruption_stats(self) -> Dict:
        """Get corruption statistics"""
        return self.corruption_stats.copy()


class LegalReferenceCorruptor(DataCorruptor):
    """
    Concrete implementation for corrupting legal references in Ukrainian legal documents.
    Changes article numbers in legal references (e.g., "ст. 537 КПК України" -> "ст. 191 КПК України").
    """
    
    def __init__(self, corruption_probability: float = 0.5):
        super().__init__(corruption_probability)
        
        # Ukrainian legal code patterns
        self.legal_patterns = {
            'КК України': r'(ст\.\s*)(\d+)(\s*(?:ч\.\s*\d+)?\s*КК\s*України)',
            'КПК України': r'(ст\.\s*)(\d+)(\s*(?:ч\.\s*\d+)?\s*КПК\s*України)',
            'ЦК України': r'(ст\.\s*)(\d+)(\s*(?:ч\.\s*\d+)?\s*ЦК\s*України)',
            'КоАП України': r'(ст\.\s*)(\d+)(\s*(?:ч\.\s*\d+)?\s*КоАП\s*України)',
            'СК України': r'(ст\.\s*)(\d+)(\s*(?:ч\.\s*\d+)?\s*СК\s*України)',
            'КЗпП України': r'(ст\.\s*)(\d+)(\s*(?:ч\.\s*\d+)?\s*КЗпП\s*України)'
        }
        
        # Article number ranges for different codes (realistic ranges)
        self.article_ranges = {
            'КК України': (1, 446),      # Criminal Code
            'КПК України': (1, 615),     # Criminal Procedure Code
            'ЦК України': (1, 1308),     # Civil Code
            'КоАП України': (1, 328),    # Administrative Code
            'СК України': (1, 270),      # Family Code
            'КЗпП України': (1, 265)     # Labor Code
        }
    
    def can_corrupt(self, sample: Dict) -> bool:
        """Check if sample contains legal references that can be corrupted"""
        text = sample.get('text', '')
        if not text:
            return False
        
        # Check if text contains any legal references
        for pattern in self.legal_patterns.values():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def corrupt_sample(self, sample: Dict) -> CorruptionResult:
        """Corrupt legal references in a sample"""
        try:
            corrupted_sample = sample.copy()
            corruption_details = {
                'changes': [],
                'patterns_found': 0,
                'patterns_corrupted': 0
            }
            
            # Corrupt text field
            corrupted_text, text_changes = self._corrupt_text(sample.get('text', ''))
            if text_changes:
                corrupted_sample['text'] = corrupted_text
                corruption_details['changes'].extend(text_changes)
            
            # Corrupt triplets if they exist
            if 'triplets' in sample:
                corrupted_triplets, triplet_changes = self._corrupt_triplets(sample['triplets'])
                if triplet_changes:
                    corrupted_sample['triplets'] = corrupted_triplets
                    corruption_details['changes'].extend(triplet_changes)
            
            # Corrupt entities/tags if they contain legal references
            if 'entities' in sample:
                corrupted_entities, entity_changes = self._corrupt_entities(sample['entities'])
                if entity_changes:
                    corrupted_sample['entities'] = corrupted_entities
                    corruption_details['changes'].extend(entity_changes)
            
            corruption_applied = len(corruption_details['changes']) > 0
            corruption_details['patterns_corrupted'] = len(corruption_details['changes'])
            
            return CorruptionResult(
                original_data=sample,
                corrupted_data=corrupted_sample,
                corruption_applied=corruption_applied,
                corruption_type='legal_reference_corruption',
                corruption_details=corruption_details
            )
            
        except Exception as e:
            logger.error(f"Error in corrupt_sample: {e}")
            return CorruptionResult(
                original_data=sample,
                corrupted_data=sample,
                corruption_applied=False,
                corruption_type='legal_reference_corruption',
                corruption_details={'error': str(e)}
            )
    
    def _corrupt_text(self, text: str) -> Tuple[str, List[Dict]]:
        """Corrupt legal references in text"""
        if not text:
            return text, []
        
        corrupted_text = text
        changes = []
        
        for code_name, pattern in self.legal_patterns.items():
            matches = list(re.finditer(pattern, corrupted_text, re.IGNORECASE))
            
            for match in reversed(matches):  # Process in reverse to maintain positions
                original_article = int(match.group(2))
                
                # Generate a different article number
                min_article, max_article = self.article_ranges.get(code_name, (1, 500))
                new_article = self._generate_different_article(original_article, min_article, max_article)
                
                # Replace the article number
                original_ref = match.group(0)
                new_ref = match.group(1) + str(new_article) + match.group(3)
                
                corrupted_text = (
                    corrupted_text[:match.start()] + 
                    new_ref + 
                    corrupted_text[match.end():]
                )
                
                changes.append({
                    'field': 'text',
                    'original': original_ref,
                    'corrupted': new_ref,
                    'code_type': code_name,
                    'position': match.start()
                })
        
        return corrupted_text, changes
    
    def _corrupt_triplets(self, triplets: Union[str, List]) -> Tuple[Union[str, List], List[Dict]]:
        """Corrupt legal references in triplets"""
        changes = []
        
        try:
            # Parse triplets if they're in string format
            if isinstance(triplets, str):
                triplets_data = json.loads(triplets)
                is_string = True
            else:
                triplets_data = triplets
                is_string = False
            
            corrupted_triplets = []
            
            for triplet in triplets_data:
                corrupted_triplet = triplet.copy()
                
                # Corrupt legal_reference field if it exists
                if 'legal_reference' in triplet and triplet['legal_reference']:
                    corrupted_ref, ref_changes = self._corrupt_text(triplet['legal_reference'])
                    if ref_changes:
                        corrupted_triplet['legal_reference'] = corrupted_ref
                        changes.extend([{**change, 'field': 'triplet_legal_reference'} for change in ref_changes])
                
                # Corrupt source, relation, target fields if they contain legal references
                for field in ['source', 'relation', 'target']:
                    if field in triplet and triplet[field]:
                        corrupted_field, field_changes = self._corrupt_text(triplet[field])
                        if field_changes:
                            corrupted_triplet[field] = corrupted_field
                            changes.extend([{**change, 'field': f'triplet_{field}'} for change in field_changes])
                
                corrupted_triplets.append(corrupted_triplet)
            
            # Return in original format
            if is_string:
                return json.dumps(corrupted_triplets, ensure_ascii=False), changes
            else:
                return corrupted_triplets, changes
                
        except Exception as e:
            logger.error(f"Error corrupting triplets: {e}")
            return triplets, []
    
    def _corrupt_entities(self, entities: Union[str, List]) -> Tuple[Union[str, List], List[Dict]]:
        """Corrupt legal references in entities"""
        changes = []
        
        try:
            # Parse entities if they're in string format
            if isinstance(entities, str):
                entities_data = json.loads(entities)
                is_string = True
            else:
                entities_data = entities
                is_string = False
            
            corrupted_entities = []
            
            for entity in entities_data:
                corrupted_entity = entity.copy()
                
                # Corrupt text field if it contains legal references
                if 'text' in entity and entity['text']:
                    corrupted_text, text_changes = self._corrupt_text(entity['text'])
                    if text_changes:
                        corrupted_entity['text'] = corrupted_text
                        changes.extend([{**change, 'field': 'entity_text'} for change in text_changes])
                
                corrupted_entities.append(corrupted_entity)
            
            # Return in original format
            if is_string:
                return json.dumps(corrupted_entities, ensure_ascii=False), changes
            else:
                return corrupted_entities, changes
                
        except Exception as e:
            logger.error(f"Error corrupting entities: {e}")
            return entities, []
    
    def _generate_different_article(self, original: int, min_val: int, max_val: int) -> int:
        """Generate a different article number within valid range"""
        # Try to generate a number that's different from original
        attempts = 0
        while attempts < 10:
            new_article = random.randint(min_val, max_val)
            if new_article != original:
                return new_article
            attempts += 1
        
        # Fallback: add or subtract 1 within bounds
        if original > min_val:
            return original - 1
        elif original < max_val:
            return original + 1
        else:
            return original  # Should rarely happen


# Convenience class for backward compatibility
class BinaryClassificationCorruptor(LegalReferenceCorruptor):
    """Backward compatibility alias for LegalReferenceCorruptor"""
    
    def __init__(self, legal_entities: List, triplets: List, legal_reference: str, content: str, 
                 corruption_probability: float = 0.5):
        super().__init__(corruption_probability)
        
        # Convert old-style parameters to new format for processing
        self.legacy_data = {
            'legal_entities': legal_entities,
            'triplets': triplets,
            'legal_reference': legal_reference,
            'content': content
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core ABC and result classes
    'DataCorruptor',
    'CorruptionResult',

    # Main implementation
    'LegalReferenceCorruptor',
    
    # Backward compatibility
    'BinaryClassificationCorruptor',
]