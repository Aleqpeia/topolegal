from processors.entities import BaseRegexComponent
from spacy.language import Language

@Language.factory("case_component")
class CaseComponent(BaseRegexComponent):
    @property
    def label(self) -> str:
        return "CASE_NUMBER"

    @property
    def pattern_str(self) -> str:
        # matches all Ukrainian cases for 'провадження' plus number (№...)
        forms = [
            "провадження",    # nominative, genitive, accusative
            "провадженню",    # dative
            "провадженні",    # locative
            "провадженням",   # instrumental
        ]
        # escape each form for regex safety
        escaped = [rf"{form}" for form in forms]
        # join into a single group
        forms_regex = "|".join(escaped)
        # full pattern: word boundary, one of the forms, space, №digits
        return rf"\b(?:{forms_regex})\s+№\d+\b"