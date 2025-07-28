from src.legal_graph.processors.entities import BaseMatcherComponent
from spacy.language import Language

@Language.factory("date_component")
class DateComponent(BaseMatcherComponent):
    @property
    def label(self) -> str:
        return "DATE"

    patterns = [[{"TEXT": {"REGEX": r"\b\d{1,2} [а-яіїє]+ \d{4}\b"}}]]