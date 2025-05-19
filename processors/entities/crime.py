from spacy.language import Language
from processors.entities import BaseMatcherComponent

_CRIME_WORDS = [
    "крадіжка",
    "бездіяльність",
    "шахрайство",
    "хуліганство",
]


@Language.factory("crime_component")
class CrimeComponent(BaseMatcherComponent):
    label = "CRIME"
    patterns = [[{"LOWER": kw}] for kw in _CRIME_WORDS]
