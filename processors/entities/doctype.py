from spacy.language import Language
from processors.entities import BaseListComponent


@Language.factory("doctype_component")
class DocTypeComponent(BaseListComponent):
    label = "DOCTYPE"
    terms = ["скарга", "ухвала", "постанова", "рішення", "клопотання"]