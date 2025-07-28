from spacy.language import Language
from src.legal_graph.processors.entities import BaseMatcherComponent

_DOC_KEYWORDS = [
    "ухвала",
    "постанова",
    "рішення",
    "скарга",
    "клопотання",
]

@Language.factory("doctype_component")
class DocTypeComponent(BaseMatcherComponent):
    @property
    def label(self):
        return "DTYPE"

    patterns = [[{"LOWER": kw}] for kw in _DOC_KEYWORDS]
