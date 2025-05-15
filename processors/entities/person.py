from processors.entities import BaseRegexComponent
from spacy.language import Language

@Language.factory("person_component")
class PersonComponent(BaseRegexComponent):
    @property
    def label(self):
        return "PERSON"

    @property
    def pattern_str(self):
        return r"ОСОБА_\d+"