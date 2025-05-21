from processors.entities import BaseMatcherComponent
from spacy.language import Language

@Language.factory("adress_component")
class PersonComponent(BaseMatcherComponent):
    @property
    def label(self):
        return "LOC"

    patterns = [[{"TEXT": {"REGEX": r"АДРЕСА_\d+"}}]]