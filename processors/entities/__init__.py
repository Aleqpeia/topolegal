import spacy
from abc import ABC, abstractmethod
from spacy.language import Language
import re
from spacy.matcher import PhraseMatcher

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

class BaseListComponent:
    def __init__(self, nlp, name):
        self.name = name
        # використовуємо LOWER-атрибут для нечутливості до регістру
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(text) for text in self.terms]
        self.matcher.add(self.label, None, *patterns)

    def __call__(self, doc):
        ents = list(doc.ents)
        occupied = [(ent.start, ent.end) for ent in ents]
        for match_id, start, end in self.matcher(doc):
            # уникаємо перекриття з існуючими сутностями
            if any(not (end <= o_start or start >= o_end) for o_start, o_end in occupied):
                continue
            span = doc[start:end]
            span.label_ = self.label
            ents.append(span)
            occupied.append((start, end))
        doc.ents = tuple(ents)
        return doc