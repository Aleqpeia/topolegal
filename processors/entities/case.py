from processors.entities import BaseMatcherComponent
from spacy.language import Language


# some abstraction needed for span around № selection
_CASE_FORMS = [
    "провадження",   # nom / acc / gen
    "провадженню",   # dat
    "провадженні",   # loc
    "провадженням",  # instr
]

@Language.factory("case_component")
class CaseComponent(BaseMatcherComponent):
    @property
    def label(self) -> str:
        return "CASEID"

    patterns = [[
        {"LOWER": {"IN": _CASE_FORMS}},
        {"TEXT": {"REGEX": r"№\\d+"}}
    ]]
