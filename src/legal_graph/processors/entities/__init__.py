import spacy
from abc import ABC, abstractmethod, abstractproperty
from spacy.language import Language
import re
from spacy.matcher import Matcher
from spacy.util import filter_spans


class BaseMatcherComponent(ABC):
    """
    Sub-classes must define:
        • label      – spaCy entity label
        • patterns   – list[ list[dict] ]  (token patterns for Matcher)
    """

    def __init__(self, nlp: Language, name: str):
        self.name = name
        self.matcher = Matcher(nlp.vocab, validate=True)
        self.matcher.add(self.label, self.patterns)

    # --- must be provided by child -----------------------------------------
    @abstractproperty
    def label(self) -> str:      ...

    @abstractproperty
    def patterns(self):          ...
    # -----------------------------------------------------------------------

    def __call__(self, doc):
        spans = []
        occupied = []                       # char-level intervals to avoid overlaps
        for mid, start, end in self.matcher(doc):
            if any(not (end <= s or start >= e) for s, e in occupied):
                continue                    # overlapping – skip
            span = doc[start:end]
            span.label_ = self.label
            spans.append(span)
            occupied.append((start, end))

        if spans:
            doc.ents = filter_spans(list(doc.ents) + spans)
        return doc


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base component for creating custom entity matchers
    'BaseMatcherComponent',
]