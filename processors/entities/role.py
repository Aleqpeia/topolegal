from processors.entities import BaseRegexComponent
from spacy.language import Language

@Language.factory("role_component")
class RoleComponent(BaseRegexComponent):
    @property
    def label(self) -> str:
        return "ROLE"

    @property
    def pattern_str(self) -> str:
        # приклад: “Слідчий суддя” або “секретар судового засідання”
        return r"(Слідчий суддя|секретар судового засідання)"