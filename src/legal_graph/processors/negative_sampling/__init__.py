"""
Negative Sampling Module for Legal References

This module provides functionality for creating negative samples by intelligently
modifying legal reference components (article numbers and part numbers) in
Ukrainian legal documents.

Unlike the balance module which replaces entire article numbers randomly,
this module provides fine-grained control over modifications, creating
"hard negative" samples useful for training graph neural networks.
"""

from abc import ABC
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel
import json
import logging
import re
import random
from pathlib import Path

# Import base classes from balance module
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from balance import DataCorruptor, CorruptionResult

# Import strategies
from .strategies import (
    SamplingStrategy,
    ArticleOffsetStrategy,
    PartOffsetStrategy,
    CombinedOffsetStrategy,
    IndexSwapStrategy,
    RandomComponentStrategy,
    MixedStrategy,
    DEFAULT_STRATEGY,
    CONSERVATIVE_STRATEGY,
    AGGRESSIVE_STRATEGY,
    create_strategy,
)

logger = logging.getLogger(__name__)


class NegativeSampleResult(BaseModel):
    """Result model for negative sampling operation"""

    original_data: Dict
    sampled_data: Dict
    sampling_applied: bool
    modifications: List[Dict]
    strategy_used: str


class NegativeSampler(BaseModel):
    """
    Create negative samples by modifying legal reference components.

    This sampler extends DataCorruptor to provide fine-grained control over
    legal reference modifications. It can modify article numbers (ст.#),
    part numbers (ч.#), or both, using various strategies.

    Key differences from LegalReferenceCorruptor:
    - Component-level modifications (article ±offset, part ±offset)
    - Multiple sampling strategies (offset, swap, random, mixed)
    - Can create syntactically invalid references for hard negatives
    - Better control over modification granularity

    Example:
        >>> sampler = NegativeSampler(
        ...     strategy='article',
        ...     sampling_probability=0.5
        ... )
        >>> dataset = [{'text': 'згідно з ч.2 ст.185 КК України', ...}]
        >>> negative_samples = sampler.create_negative_samples(dataset)
        >>> # Result: 'згідно з ч.2 ст.187 КК України' (article modified)
    """

    def __init__(
        self,
        sampling_probability: float = 0.5,
        strategy: Union[str, SamplingStrategy] = "mixed",
        article_ranges: Optional[Dict[str, Tuple[int, int]]] = {
            "КК України": (1, 446),
            "КПК України": (1, 615),
            "ЦК України": (1, 1308),
            "КоАП України": (1, 328),
            "СК України": (1, 270),
            "КЗпП України": (1, 265),
        },
        max_part: int = 10,
        preserve_format: bool = True,
    ):
        """
        Initialize the negative sampler.

        Args:
            sampling_probability: Probability of sampling each document (0.0 to 1.0)
            strategy: Sampling strategy ('article', 'part', 'combined', 'swap',
                     'random', 'mixed', 'conservative', 'aggressive') or
                     SamplingStrategy instance
            article_ranges: Dict mapping code names to (min, max) article numbers
                          If None, uses default ranges from balance module
            max_part: Maximum part number (default 10)
            preserve_format: If True, preserve original formatting (spaces, etc.)
        """
        # Initialize base class with corruption_probability
        super().__init__(corruption_probability=sampling_probability)

        # Set up strategy
        if isinstance(strategy, str):
            self.strategy = create_strategy(strategy)
        else:
            self.strategy = strategy

        # Article ranges for different Ukrainian legal codes
        self.article_ranges = article_ranges

        self.max_part = max_part
        self.preserve_format = preserve_format

        # Statistics
        self.sampling_stats = {
            "total_processed": 0,
            "sampled": 0,
            "skipped": 0,
            "errors": 0,
            "modifications_by_type": {},
        }

        # Regex patterns for legal references
        # Pattern 1: "ч.X ст.Y КК України" (part + article with "ст.")
        self.part_article_pattern = re.compile(
            r"(ч\.\s*)(\d+)(\s+ст\.\s*)(\d+)(\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)",
            re.IGNORECASE,
        )

        # Pattern 1b: "ч.X п.Y КК України" (part + article with "п." - punkt)
        # Some documents use "п." (punkt) instead of "ст." (stattya) for articles
        self.part_punkt_pattern = re.compile(
            r"(ч\.\s*)(\d+)(\s+п\.\s*)(\d+)(\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)",
            re.IGNORECASE,
        )

        # Pattern 2: "ст.Y КК України" (article without part, but may have part later)
        self.article_pattern = re.compile(
            r"(ст\.\s*)(\d+)(\s*(?:ч\.\s*\d+)?\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)",
            re.IGNORECASE,
        )

        # Pattern 2b: "п.Y КК України" (punkt without part)
        self.punkt_pattern = re.compile(
            r"(п\.\s*)(\d+)(\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)", re.IGNORECASE
        )

        # Pattern 3: Standalone "ч.X" or "п.X"
        self.part_only_pattern = re.compile(r"(ч\.|п\.)(\s*)(\d+)", re.IGNORECASE)

        logger.info(
            f"Initialized NegativeSampler with strategy: {self.strategy.get_strategy_name()}"
        )

    def can_corrupt(self, sample: Dict) -> bool:
        """Check if sample contains legal references that can be modified"""
        text = sample.get("text", "")
        if not text:
            return False

        # Check if text contains any legal references
        if self.part_article_pattern.search(text):
            return True
        if self.part_punkt_pattern.search(text):
            return True
        if self.article_pattern.search(text):
            return True
        if self.punkt_pattern.search(text):
            return True

        return False

    def corrupt_sample(self, sample: Dict) -> CorruptionResult:
        """
        Create negative sample by modifying legal references.

        This method overrides DataCorruptor.corrupt_sample to provide
        component-level modifications using the configured strategy.

        IMPORTANT: All modifications to legal references in the text are
        consistently applied to the same references in triplets and entities
        to maintain data consistency.
        """
        try:
            sampled_sample = sample.copy()
            modifications = []

            # Step 1: Modify text field and collect reference mappings
            sampled_text, text_mods = self._modify_text(sample.get("text", ""))
            if text_mods:
                sampled_sample["text"] = sampled_text
                modifications.extend(text_mods)

            # Build a mapping of original → modified references from text modifications
            # This ensures consistency across text, triplets, and entities
            reference_mapping = {}
            for mod in text_mods:
                if "original" in mod and "modified" in mod:
                    reference_mapping[mod["original"]] = mod["modified"]

            # Step 2: Apply the SAME modifications to triplets
            # This ensures triplets match the modified text
            if "triplets" in sample:
                sampled_triplets, triplet_mods = self._modify_triplets_with_mapping(
                    sample["triplets"], reference_mapping
                )
                if triplet_mods:
                    sampled_sample["triplets"] = sampled_triplets
                    modifications.extend(triplet_mods)

            # Step 3: Apply the SAME modifications to entities
            # This ensures entities match the modified text
            if "entities" in sample:
                sampled_entities, entity_mods = self._modify_entities_with_mapping(
                    sample["entities"], reference_mapping
                )
                if entity_mods:
                    sampled_sample["entities"] = sampled_entities
                    modifications.extend(entity_mods)

            sampling_applied = len(modifications) > 0

            # Update statistics
            if sampling_applied:
                for mod in modifications:
                    mod_type = mod.get("modification_type", "unknown")
                    self.sampling_stats["modifications_by_type"][mod_type] = (
                        self.sampling_stats["modifications_by_type"].get(mod_type, 0)
                        + 1
                    )

            return CorruptionResult(
                original_data=sample,
                corrupted_data=sampled_sample,
                corruption_applied=sampling_applied,
                corruption_type="negative_sampling",
                corruption_details={
                    "modifications": modifications,
                    "strategy": self.strategy.get_strategy_name(),
                    "num_modifications": len(modifications),
                    "reference_mapping": reference_mapping,  # Track the mapping for debugging
                },
            )

        except Exception as e:
            logger.error(f"Error in corrupt_sample: {e}")
            return CorruptionResult(
                original_data=sample,
                corrupted_data=sample,
                corruption_applied=False,
                corruption_type="negative_sampling",
                corruption_details={"error": str(e)},
            )

    def _modify_text(self, text: str) -> Tuple[str, List[Dict]]:
        """Modify legal references in text"""
        if not text:
            return text, []

        modified_text = text
        modifications = []

        # Process "ч.X ст.Y" patterns (most specific first)
        matches = list(self.part_article_pattern.finditer(text))

        for match in reversed(matches):  # Process in reverse to maintain positions
            try:
                part_prefix = match.group(1)  # "ч."
                part_num = int(match.group(2))  # X
                middle = match.group(3)  # " ст."
                article_num = int(match.group(4))  # Y
                suffix = match.group(5)  # " КК України"

                # Determine code name
                code_name = self._extract_code_name(suffix)
                min_article, max_article = self.article_ranges.get(code_name, (1, 500))

                # Apply strategy
                new_article, new_part, mod_type = self.strategy.modify_reference(
                    article=article_num,
                    part=part_num,
                    code_name=code_name,
                    min_article=min_article,
                    max_article=max_article,
                    max_part=self.max_part,
                )

                # Only apply if something changed
                if new_article != article_num or new_part != part_num:
                    original_ref = match.group(0)
                    new_ref = f"{part_prefix}{new_part}{middle}{new_article}{suffix}"

                    modified_text = (
                        modified_text[: match.start()]
                        + new_ref
                        + modified_text[match.end() :]
                    )

                    modifications.append(
                        {
                            "field": "text",
                            "original": original_ref,
                            "modified": new_ref,
                            "modification_type": mod_type,
                            "position": match.start(),
                            "original_article": article_num,
                            "new_article": new_article,
                            "original_part": part_num,
                            "new_part": new_part,
                            "code_name": code_name,
                        }
                    )

            except Exception as e:
                logger.warning(f"Error processing match: {e}")
                continue

        # Process "ч.X п.Y" patterns (punkt instead of stattya)
        # These are processed separately to avoid conflicts
        matches_punkt = list(self.part_punkt_pattern.finditer(modified_text))

        for match in reversed(
            matches_punkt
        ):  # Process in reverse to maintain positions
            try:
                part_prefix = match.group(1)  # "ч."
                part_num = int(match.group(2))  # X
                middle = match.group(3)  # " п."
                artikel_num = int(match.group(4))  # Y (punkt number)
                suffix = match.group(5)  # " КК України"

                # Determine code name
                code_name = self._extract_code_name(suffix)
                min_article, max_article = self.article_ranges.get(code_name, (1, 500))

                # Apply strategy (punkt is treated like article number)
                new_artikel, new_part, mod_type = self.strategy.modify_reference(
                    article=artikel_num,
                    part=part_num,
                    code_name=code_name,
                    min_article=min_article,
                    max_article=max_article,
                    max_part=self.max_part,
                )

                # Only apply if something changed
                if new_artikel != artikel_num or new_part != part_num:
                    original_ref = match.group(0)
                    new_ref = f"{part_prefix}{new_part}{middle}{new_artikel}{suffix}"

                    modified_text = (
                        modified_text[: match.start()]
                        + new_ref
                        + modified_text[match.end() :]
                    )

                    modifications.append(
                        {
                            "field": "text",
                            "original": original_ref,
                            "modified": new_ref,
                            "modification_type": mod_type,
                            "position": match.start(),
                            "original_article": artikel_num,
                            "new_article": new_artikel,
                            "original_part": part_num,
                            "new_part": new_part,
                            "code_name": code_name,
                            "format": "ч.X п.Y",
                        }
                    )

            except Exception as e:
                logger.warning(f"Error processing ч.X п.Y match: {e}")
                continue

        # Process standalone "ст.Y" patterns (without preceding part)
        # Only process if not already covered by previous patterns
        if not modifications:  # Only if we haven't found ч.X ст.Y or ч.X п.Y patterns
            matches = list(self.article_pattern.finditer(modified_text))

            for match in reversed(matches):
                try:
                    article_prefix = match.group(1)  # "ст."
                    article_num = int(match.group(2))  # Y
                    suffix = match.group(3)  # " КК України"

                    # Check if this pattern has a part number in suffix
                    part_in_suffix = re.search(r"ч\.\s*(\d+)", suffix)
                    part_num = int(part_in_suffix.group(1)) if part_in_suffix else None

                    code_name = self._extract_code_name(suffix)
                    min_article, max_article = self.article_ranges.get(
                        code_name, (1, 500)
                    )

                    # Apply strategy
                    new_article, new_part, mod_type = self.strategy.modify_reference(
                        article=article_num,
                        part=part_num,
                        code_name=code_name,
                        min_article=min_article,
                        max_article=max_article,
                        max_part=self.max_part,
                    )

                    if new_article != article_num:
                        original_ref = match.group(0)

                        # Reconstruct with modified article
                        if part_in_suffix and new_part != part_num:
                            # Also update part in suffix if modified
                            new_suffix = re.sub(r"ч\.\s*\d+", f"ч.{new_part}", suffix)
                            new_ref = f"{article_prefix}{new_article}{new_suffix}"
                        else:
                            new_ref = f"{article_prefix}{new_article}{suffix}"

                        modified_text = (
                            modified_text[: match.start()]
                            + new_ref
                            + modified_text[match.end() :]
                        )

                        modifications.append(
                            {
                                "field": "text",
                                "original": original_ref,
                                "modified": new_ref,
                                "modification_type": mod_type,
                                "position": match.start(),
                                "original_article": article_num,
                                "new_article": new_article,
                                "code_name": code_name,
                            }
                        )

                except Exception as e:
                    logger.warning(f"Error processing article pattern: {e}")
                    continue

            # Also process standalone "п.Y КК України" patterns (punkt without part)
            matches_punkt_alone = list(self.punkt_pattern.finditer(modified_text))

            for match in reversed(matches_punkt_alone):
                try:
                    punkt_prefix = match.group(1)  # "п."
                    punkt_num = int(match.group(2))  # Y
                    suffix = match.group(3)  # " КК України"

                    code_name = self._extract_code_name(suffix)
                    min_article, max_article = self.article_ranges.get(
                        code_name, (1, 500)
                    )

                    # Apply strategy (punkt treated as article, no part)
                    new_punkt, _, mod_type = self.strategy.modify_reference(
                        article=punkt_num,
                        part=None,
                        code_name=code_name,
                        min_article=min_article,
                        max_article=max_article,
                        max_part=self.max_part,
                    )

                    if new_punkt != punkt_num:
                        original_ref = match.group(0)
                        new_ref = f"{punkt_prefix}{new_punkt}{suffix}"

                        modified_text = (
                            modified_text[: match.start()]
                            + new_ref
                            + modified_text[match.end() :]
                        )

                        modifications.append(
                            {
                                "field": "text",
                                "original": original_ref,
                                "modified": new_ref,
                                "modification_type": mod_type,
                                "position": match.start(),
                                "original_article": punkt_num,
                                "new_article": new_punkt,
                                "code_name": code_name,
                                "format": "п.Y",
                            }
                        )

                except Exception as e:
                    logger.warning(f"Error processing punkt pattern: {e}")
                    continue

        return modified_text, modifications

    def _modify_triplets(
        self, triplets: Union[str, List]
    ) -> Tuple[Union[str, List], List[Dict]]:
        """Modify legal references in triplets"""
        modifications = []

        try:
            # Parse triplets if string format
            if isinstance(triplets, str):
                triplets_data = json.loads(triplets)
                is_string = True
            else:
                triplets_data = triplets
                is_string = False

            modified_triplets = []

            for triplet in triplets_data:
                modified_triplet = triplet.copy()

                # Modify legal_reference field
                if "legal_reference" in triplet and triplet["legal_reference"]:
                    modified_ref, ref_mods = self._modify_text(
                        triplet["legal_reference"]
                    )
                    if ref_mods:
                        modified_triplet["legal_reference"] = modified_ref
                        modifications.extend(
                            [
                                {**mod, "field": "triplet_legal_reference"}
                                for mod in ref_mods
                            ]
                        )

                # Modify other fields if they contain legal references
                for field in ["source", "relation", "target"]:
                    if field in triplet and triplet[field]:
                        modified_field, field_mods = self._modify_text(triplet[field])
                        if field_mods:
                            modified_triplet[field] = modified_field
                            modifications.extend(
                                [
                                    {**mod, "field": f"triplet_{field}"}
                                    for mod in field_mods
                                ]
                            )

                modified_triplets.append(modified_triplet)

            # Return in original format
            if is_string:
                return json.dumps(modified_triplets, ensure_ascii=False), modifications
            else:
                return modified_triplets, modifications

        except Exception as e:
            logger.error(f"Error modifying triplets: {e}")
            return triplets, []

    def _modify_entities(
        self, entities: Union[str, List]
    ) -> Tuple[Union[str, List], List[Dict]]:
        """Modify legal references in entities"""
        modifications = []

        try:
            # Parse entities if string format
            if isinstance(entities, str):
                entities_data = json.loads(entities)
                is_string = True
            else:
                entities_data = entities
                is_string = False

            modified_entities = []

            for entity in entities_data:
                modified_entity = entity.copy()

                # Modify text field if it contains legal references
                if "text" in entity and entity["text"]:
                    modified_text, text_mods = self._modify_text(entity["text"])
                    if text_mods:
                        modified_entity["text"] = modified_text
                        modifications.extend(
                            [{**mod, "field": "entity_text"} for mod in text_mods]
                        )

                modified_entities.append(modified_entity)

            # Return in original format
            if is_string:
                return json.dumps(modified_entities, ensure_ascii=False), modifications
            else:
                return modified_entities, modifications

        except Exception as e:
            logger.error(f"Error modifying entities: {e}")
            return entities, []

    def _apply_reference_mapping(
        self, text: str, reference_mapping: Dict[str, str]
    ) -> Tuple[str, List[Dict]]:
        """
        Apply a reference mapping to text.

        This method replaces all occurrences of original references with their
        modified versions according to the mapping. This ensures consistency
        across text, triplets, and entities.

        Args:
            text: Text to modify
            reference_mapping: Dict mapping original references to modified ones

        Returns:
            Tuple of (modified_text, list of modification records)
        """
        if not text or not reference_mapping:
            return text, []

        modified_text = text
        modifications = []

        # Apply each mapping (in reverse order of length to handle overlaps correctly)
        # e.g., "ч.2 ст.185 КК України" should be matched before "ст.185"
        for original, modified in sorted(
            reference_mapping.items(), key=lambda x: -len(x[0])
        ):
            if original in modified_text:
                count = modified_text.count(original)
                modified_text = modified_text.replace(original, modified)
                modifications.append(
                    {
                        "original": original,
                        "modified": modified,
                        "occurrences": count,
                        "modification_type": "reference_mapping_applied",
                    }
                )

        return modified_text, modifications

    def _modify_triplets_with_mapping(
        self, triplets: Union[str, List], reference_mapping: Dict[str, str]
    ) -> Tuple[Union[str, List], List[Dict]]:
        """
        Apply reference mapping to triplets to ensure consistency with modified text.

        Args:
            triplets: Triplets data (str or list)
            reference_mapping: Dict mapping original references to modified ones

        Returns:
            Tuple of (modified_triplets, list of modification records)
        """
        if not reference_mapping:
            return triplets, []

        modifications = []

        try:
            # Parse triplets if string format
            if isinstance(triplets, str):
                triplets_data = json.loads(triplets)
                is_string = True
            else:
                triplets_data = triplets
                is_string = False

            modified_triplets = []

            for triplet in triplets_data:
                modified_triplet = triplet.copy()

                # Apply mapping to legal_reference field
                if "legal_reference" in triplet and triplet["legal_reference"]:
                    modified_ref, ref_mods = self._apply_reference_mapping(
                        triplet["legal_reference"], reference_mapping
                    )
                    if ref_mods:
                        modified_triplet["legal_reference"] = modified_ref
                        modifications.extend(
                            [
                                {**mod, "field": "triplet_legal_reference"}
                                for mod in ref_mods
                            ]
                        )

                # Apply mapping to other fields if they contain legal references
                for field in ["source", "relation", "target"]:
                    if field in triplet and triplet[field]:
                        modified_field, field_mods = self._apply_reference_mapping(
                            str(triplet[field]), reference_mapping
                        )
                        if field_mods:
                            modified_triplet[field] = modified_field
                            modifications.extend(
                                [
                                    {**mod, "field": f"triplet_{field}"}
                                    for mod in field_mods
                                ]
                            )

                modified_triplets.append(modified_triplet)

            # Return in original format
            if is_string:
                return json.dumps(modified_triplets, ensure_ascii=False), modifications
            else:
                return modified_triplets, modifications

        except Exception as e:
            logger.error(f"Error applying reference mapping to triplets: {e}")
            return triplets, []

    def _modify_entities_with_mapping(
        self, entities: Union[str, List], reference_mapping: Dict[str, str]
    ) -> Tuple[Union[str, List], List[Dict]]:
        """
        Apply reference mapping to entities to ensure consistency with modified text.

        Args:
            entities: Entities data (str or list)
            reference_mapping: Dict mapping original references to modified ones

        Returns:
            Tuple of (modified_entities, list of modification records)
        """
        if not reference_mapping:
            return entities, []

        modifications = []

        try:
            # Parse entities if string format
            if isinstance(entities, str):
                entities_data = json.loads(entities)
                is_string = True
            else:
                entities_data = entities
                is_string = False

            modified_entities = []

            for entity in entities_data:
                modified_entity = entity.copy()

                # Apply mapping to text field if it contains legal references
                if "text" in entity and entity["text"]:
                    modified_text, text_mods = self._apply_reference_mapping(
                        entity["text"], reference_mapping
                    )
                    if text_mods:
                        modified_entity["text"] = modified_text
                        modifications.extend(
                            [{**mod, "field": "entity_text"} for mod in text_mods]
                        )

                modified_entities.append(modified_entity)

            # Return in original format
            if is_string:
                return json.dumps(modified_entities, ensure_ascii=False), modifications
            else:
                return modified_entities, modifications

        except Exception as e:
            logger.error(f"Error applying reference mapping to entities: {e}")
            return entities, []

    def _extract_code_name(self, text: str) -> str:
        """Extract legal code name from text"""
        for code_name in self.article_ranges.keys():
            if code_name in text:
                return code_name
        return "КК України"  # Default

    def create_negative_samples(
        self, dataset: List[Dict], label_column: str = "is_valid"
    ) -> List[Dict]:
        """
        Create negative samples from a dataset.

        This is an alias for corrupt_dataset that's more semantically appropriate
        for negative sampling.

        Args:
            dataset: List of data samples
            label_column: Column name for validity labels

        Returns:
            List of samples with validity labels (True = original, False = negative)
        """
        return self.corrupt_dataset(dataset, label_column=label_column)

    def get_sampling_stats(self) -> Dict:
        """Get negative sampling statistics"""
        stats = self.get_corruption_stats()
        stats["modifications_by_type"] = self.sampling_stats["modifications_by_type"]
        stats["strategy"] = self.strategy.get_strategy_name()
        return stats


def create_negative_samples(
    dataset: List[Dict],
    strategy: Union[str, SamplingStrategy] = "mixed",
    sampling_probability: float = 0.5,
    label_column: str = "is_valid",
) -> List[Dict]:
    """
    Convenience function to create negative samples from a dataset.

    Args:
        dataset: List of data samples
        strategy: Sampling strategy name or instance
        sampling_probability: Probability of sampling each sample
        label_column: Column name for validity labels

    Returns:
        List of samples with validity labels
    """
    sampler = NegativeSampler(
        sampling_probability=sampling_probability, strategy=strategy
    )
    return sampler.create_negative_samples(dataset, label_column=label_column)


__all__ = [
    # Main classes
    "NegativeSampler",
    "NegativeSampleResult",
    # Strategies (re-export from strategies module)
    "SamplingStrategy",
    "ArticleOffsetStrategy",
    "PartOffsetStrategy",
    "CombinedOffsetStrategy",
    "IndexSwapStrategy",
    "RandomComponentStrategy",
    "MixedStrategy",
    "DEFAULT_STRATEGY",
    "CONSERVATIVE_STRATEGY",
    "AGGRESSIVE_STRATEGY",
    "create_strategy",
    # Convenience functions
    "create_negative_samples",
]
