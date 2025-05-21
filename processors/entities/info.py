from processors.entities import BaseMatcherComponent
from spacy.language import Language

@Language.factory("info_component")
class PersonComponent(BaseMatcherComponent):
    @property
    def label(self):
        return "INFO"

    patterns = [[{"TEXT": {"REGEX": r"ІНФОРМАЦІЯ_\d+"}}]]