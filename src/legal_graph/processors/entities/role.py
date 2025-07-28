from src.legal_graph.processors.entities import BaseMatcherComponent
from spacy.language import Language


_SINGLE_NOUN_ROLES = [
    "прокурор",
    "адвокат",
    "слідчий",
    "суддя",          # when used alone (e.g. «суддя Петров»)
    "обвинувачений",
    "підозрюваний",
]

# token dict for a set of lemmas
_single_token_pat = [{"LEMMA": {"IN": _SINGLE_NOUN_ROLES}}]

# --------------- Component --------------------

@Language.factory("role_component")
class RoleComponent(BaseMatcherComponent):
    """Identify court‐related roles (judge, prosecutor, etc.)."""

    label = "ROLE"

    # list[list[dict]] — each inner list is a token pattern
    patterns = [
        # слідчий суддя   /   (optional adjective in-between)
        [{"LEMMA": "слідчий"}, {"LEMMA": "суддя", "OP": "?"}],

        # головуючий суддя
        [{"LEMMA": "головуючий"}, {"LEMMA": "суддя"}],

        # секретар … засідання   (allow any attrs between except verbs/adv)
        [
            {"LEMMA": "секретар"},
            {"OP": "*", "IS_PUNCT": False, "POS": {"NOT_IN": ["VERB", "ADV"]}},
            {"LEMMA": "засідання"},
        ],

        # single-word roles («прокурор», «адвокат», «слідчий», …)
        [{"LEMMA": {"IN": ["прокурор", "адвокат", "слідчий", "захисник", "експерт",
                           "суддя", "обвинувачений", "підозрюваний"]}}],
    ]