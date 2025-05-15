import spacy
from abc import ABC, abstractmethod
from spacy.language import Language
import re

nlp = spacy.blank("uk")


class BaseRegexComponent(ABC):
    def __init__(self, nlp, name):
        self.name = name
        self.pattern = re.compile(self.pattern_str, flags=re.IGNORECASE)

    def __call__(self, doc):
        ents = list(doc.ents)
        for m in self.pattern.finditer(doc.text):
            span = doc.char_span(m.start(), m.end(), label=self.label)
            if span:
                ents.append(span)
        doc.ents = tuple(ents)
        return doc

    @property
    @abstractmethod
    def label(self) -> str:
        """Назва сутності, наприклад 'PERSON'."""
        pass

    @property
    @abstractmethod
    def pattern_str(self) -> str:
        """Regex-шаблон у вигляді рядка."""
        pass
