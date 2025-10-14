"""
Negative Sampling Strategies for Legal Reference Modification

This module provides different strategies for creating negative samples by
modifying legal references in Ukrainian legal documents.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import random
import re
import logging

logger = logging.getLogger(__name__)


class SamplingStrategy(ABC):
    """Abstract base class for negative sampling strategies"""

    @abstractmethod
    def modify_reference(
        self,
        article: int,
        part: Optional[int],
        code_name: str,
        min_article: int,
        max_article: int,
        max_part: int = 10
    ) -> Tuple[int, Optional[int], str]:
        """
        Modify legal reference components to create a negative sample.

        Args:
            article: Original article number
            part: Original part number (can be None)
            code_name: Legal code name (e.g., 'КК України')
            min_article: Minimum valid article number for this code
            max_article: Maximum valid article number for this code
            max_part: Maximum valid part number (default 10)

        Returns:
            Tuple of (new_article, new_part, modification_type)
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging/tracking"""
        pass


class ArticleOffsetStrategy(SamplingStrategy):
    """
    Modify article numbers by a small offset.

    This creates "near-miss" negative samples that are syntactically valid
    but semantically incorrect. Useful for training models to distinguish
    between similar but different legal references.
    """

    def __init__(self, offset_range: Tuple[int, int] = (1, 5)):
        """
        Args:
            offset_range: Min and max offset values (positive or negative)
        """
        self.min_offset, self.max_offset = offset_range

    def modify_reference(
        self,
        article: int,
        part: Optional[int],
        code_name: str,
        min_article: int,
        max_article: int,
        max_part: int = 10
    ) -> Tuple[int, Optional[int], str]:
        """Modify article number by small offset"""
        # Generate offset (exclude 0)
        offset = random.choice(
            list(range(-self.max_offset, -self.min_offset + 1)) +
            list(range(self.min_offset, self.max_offset + 1))
        )

        new_article = article + offset

        # Ensure within valid range
        new_article = max(min_article, min(max_article, new_article))

        # If we couldn't change (at boundary), try opposite direction
        if new_article == article:
            offset = -offset
            new_article = max(min_article, min(max_article, article + offset))

        return new_article, part, f"article_offset_{offset:+d}"

    def get_strategy_name(self) -> str:
        return f"ArticleOffset(±{self.min_offset}-{self.max_offset})"


class PartOffsetStrategy(SamplingStrategy):
    """
    Modify part numbers by a small offset.

    Creates negative samples by changing the part number (ч.) while keeping
    the article number the same. Useful when different parts of the same
    article have different legal meanings.
    """

    def __init__(self, offset_range: Tuple[int, int] = (1, 2)):
        """
        Args:
            offset_range: Min and max offset values for part numbers
        """
        self.min_offset, self.max_offset = offset_range

    def modify_reference(
        self,
        article: int,
        part: Optional[int],
        code_name: str,
        min_article: int,
        max_article: int,
        max_part: int = 10
    ) -> Tuple[int, Optional[int], str]:
        """Modify part number by small offset"""
        if part is None:
            # No part to modify, return unchanged
            return article, part, "no_part"

        # Generate offset (exclude 0)
        offset = random.choice(
            list(range(-self.max_offset, -self.min_offset + 1)) +
            list(range(self.min_offset, self.max_offset + 1))
        )

        new_part = part + offset

        # Ensure within valid range (parts typically 1-10)
        new_part = max(1, min(max_part, new_part))

        # If we couldn't change (at boundary), try opposite direction
        if new_part == part:
            offset = -offset
            new_part = max(1, min(max_part, part + offset))

        return article, new_part, f"part_offset_{offset:+d}"

    def get_strategy_name(self) -> str:
        return f"PartOffset(±{self.min_offset}-{self.max_offset})"


class CombinedOffsetStrategy(SamplingStrategy):
    """
    Modify both article and part numbers.

    Creates more diverse negative samples by changing both components
    of the legal reference.
    """

    def __init__(
        self,
        article_offset_range: Tuple[int, int] = (1, 3),
        part_offset_range: Tuple[int, int] = (1, 2),
        probability_both: float = 0.5
    ):
        """
        Args:
            article_offset_range: Offset range for article numbers
            part_offset_range: Offset range for part numbers
            probability_both: Probability of modifying both vs just one
        """
        self.article_strategy = ArticleOffsetStrategy(article_offset_range)
        self.part_strategy = PartOffsetStrategy(part_offset_range)
        self.probability_both = probability_both

    def modify_reference(
        self,
        article: int,
        part: Optional[int],
        code_name: str,
        min_article: int,
        max_article: int,
        max_part: int = 10
    ) -> Tuple[int, Optional[int], str]:
        """Modify both article and part numbers"""
        modify_both = random.random() < self.probability_both and part is not None

        if modify_both:
            # Modify both
            new_article, _, mod_type_1 = self.article_strategy.modify_reference(
                article, part, code_name, min_article, max_article, max_part
            )
            _, new_part, mod_type_2 = self.part_strategy.modify_reference(
                article, part, code_name, min_article, max_article, max_part
            )
            return new_article, new_part, f"combined_{mod_type_1}_{mod_type_2}"
        else:
            # Modify only one (randomly choose which)
            if part is not None and random.random() < 0.5:
                return self.part_strategy.modify_reference(
                    article, part, code_name, min_article, max_article, max_part
                )
            else:
                return self.article_strategy.modify_reference(
                    article, part, code_name, min_article, max_article, max_part
                )

    def get_strategy_name(self) -> str:
        return "CombinedOffset"


class IndexSwapStrategy(SamplingStrategy):
    """
    Swap article and part numbers to create syntactically invalid references.

    Creates negative samples like "ч.185 ст.2" from "ч.2 ст.185".
    These are useful for testing if the model can detect structurally invalid
    legal references.
    """

    def modify_reference(
        self,
        article: int,
        part: Optional[int],
        code_name: str,
        min_article: int,
        max_article: int,
        max_part: int = 10
    ) -> Tuple[int, Optional[int], str]:
        """Swap article and part numbers"""
        if part is None:
            # Can't swap if no part number
            return article, part, "no_swap"

        # Swap the numbers (this creates invalid reference)
        new_article = part
        new_part = article

        return new_article, new_part, "index_swap"

    def get_strategy_name(self) -> str:
        return "IndexSwap"


class RandomComponentStrategy(SamplingStrategy):
    """
    Replace with completely random but valid components.

    Creates negative samples by selecting random valid article/part numbers
    from the legal code range. This creates more diverse negative samples
    that may be semantically unrelated to the original.
    """

    def modify_reference(
        self,
        article: int,
        part: Optional[int],
        code_name: str,
        min_article: int,
        max_article: int,
        max_part: int = 10
    ) -> Tuple[int, Optional[int], str]:
        """Replace with random valid components"""
        # Generate random article (different from original)
        attempts = 0
        new_article = article
        while new_article == article and attempts < 10:
            new_article = random.randint(min_article, max_article)
            attempts += 1

        # Generate random part if original had one
        new_part = None
        if part is not None:
            attempts = 0
            new_part = part
            while new_part == part and attempts < 10:
                new_part = random.randint(1, max_part)
                attempts += 1

        return new_article, new_part, "random_component"

    def get_strategy_name(self) -> str:
        return "RandomComponent"


class MixedStrategy(SamplingStrategy):
    """
    Combine multiple strategies with configurable weights.

    Allows creating diverse negative samples by randomly selecting
    from different modification strategies.
    """

    def __init__(
        self,
        strategies: Optional[List[SamplingStrategy]] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            strategies: List of strategy instances to use
            weights: Probability weights for each strategy (must sum to 1.0)
        """
        if strategies is None:
            # Default mix of strategies
            self.strategies = [
                ArticleOffsetStrategy(offset_range=(1, 5)),
                PartOffsetStrategy(offset_range=(1, 2)),
                CombinedOffsetStrategy(),
                RandomComponentStrategy()
            ]
        else:
            self.strategies = strategies

        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(self.strategies)] * len(self.strategies)
        else:
            if len(weights) != len(self.strategies):
                raise ValueError("Number of weights must match number of strategies")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights

    def modify_reference(
        self,
        article: int,
        part: Optional[int],
        code_name: str,
        min_article: int,
        max_article: int,
        max_part: int = 10
    ) -> Tuple[int, Optional[int], str]:
        """Select and apply a random strategy"""
        # Select strategy based on weights
        strategy = random.choices(self.strategies, weights=self.weights, k=1)[0]

        new_article, new_part, mod_type = strategy.modify_reference(
            article, part, code_name, min_article, max_article, max_part
        )

        # Add strategy name to modification type
        mod_type = f"{strategy.get_strategy_name()}:{mod_type}"

        return new_article, new_part, mod_type

    def get_strategy_name(self) -> str:
        strategy_names = [s.get_strategy_name() for s in self.strategies]
        return f"Mixed({', '.join(strategy_names)})"


# Default strategies
DEFAULT_STRATEGY = MixedStrategy()
CONSERVATIVE_STRATEGY = MixedStrategy(
    strategies=[
        ArticleOffsetStrategy(offset_range=(1, 3)),
        PartOffsetStrategy(offset_range=(1, 1))
    ],
    weights=[0.7, 0.3]
)
AGGRESSIVE_STRATEGY = MixedStrategy(
    strategies=[
        ArticleOffsetStrategy(offset_range=(1, 10)),
        PartOffsetStrategy(offset_range=(1, 5)),
        CombinedOffsetStrategy(),
        IndexSwapStrategy(),
        RandomComponentStrategy()
    ],
    weights=[0.3, 0.2, 0.2, 0.1, 0.2]
)


def create_strategy(strategy_name: str) -> SamplingStrategy:
    """
    Factory function to create strategies by name.

    Args:
        strategy_name: One of 'article', 'part', 'combined', 'swap', 'random',
                      'mixed', 'conservative', 'aggressive'

    Returns:
        SamplingStrategy instance
    """
    strategy_map = {
        'article': ArticleOffsetStrategy(),
        'part': PartOffsetStrategy(),
        'combined': CombinedOffsetStrategy(),
        'swap': IndexSwapStrategy(),
        'random': RandomComponentStrategy(),
        'mixed': DEFAULT_STRATEGY,
        'conservative': CONSERVATIVE_STRATEGY,
        'aggressive': AGGRESSIVE_STRATEGY
    }

    if strategy_name.lower() not in strategy_map:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(strategy_map.keys())}"
        )

    return strategy_map[strategy_name.lower()]


__all__ = [
    'SamplingStrategy',
    'ArticleOffsetStrategy',
    'PartOffsetStrategy',
    'CombinedOffsetStrategy',
    'IndexSwapStrategy',
    'RandomComponentStrategy',
    'MixedStrategy',
    'DEFAULT_STRATEGY',
    'CONSERVATIVE_STRATEGY',
    'AGGRESSIVE_STRATEGY',
    'create_strategy'
]
