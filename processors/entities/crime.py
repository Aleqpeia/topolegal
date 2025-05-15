from spacy.language import Language
from processors.entities import BaseListComponent

@Language.factory("crime_component")
class CrimeComponent(BaseListComponent):
    label = "CRIME"
    terms = ["крадіжка", "бездіяльність", "шахрайство", "хуліганство"]
