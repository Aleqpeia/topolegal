from src.legal_graph.processors.entities import BaseMatcherComponent
from spacy.language import Language

@Language.factory("person_component")
class PersonComponent(BaseMatcherComponent):
    @property
    def label(self):
        return "PER"

    patterns = [[{"TEXT": {"REGEX": r"ОСОБА_\d+"}}]]