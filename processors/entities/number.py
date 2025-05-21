from processors.entities import BaseMatcherComponent
from spacy.language import Language

@Language.factory("number_component")
class PersonComponent(BaseMatcherComponent):
    @property
    def label(self):
        return "NUM"

    patterns = [[{"TEXT": {"REGEX": r"НОМЕР_\d+"}}]]