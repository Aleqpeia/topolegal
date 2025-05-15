from processors.entities import BaseRegexComponent
from spacy.language import Language

@Language.factory("date_component")
class DateComponent(BaseRegexComponent):
    @property
    def label(self) -> str:
        return "DATE"

    @property
    def pattern_str(self) -> str:
        # matches “22 грудня 2021”
        return r"\b\d{1,2} [а-яіїє]+ \d{4}\b"