# processors/entities/case.py
from __future__ import annotations
import re
from spacy.language import Language
from spacy.util import filter_spans
from processors.entities import BaseMatcherComponent

# base lemmas that can precede the sign «№»
_HEAD_BASES = {"справа", "провадження"}
# token text allowed inside identifier (digits / letters / slash / dash)
_ID_TOKEN_RE = re.compile(r"^[\-—/0-9a-zа-яіїєґ]+$", re.I)

@Language.factory("case_component")
class CaseComponent(BaseMatcherComponent):
    label = "CASEID"
    # seed pattern – just the sign «№»; BaseMatcherComponent will wrap
    patterns = [[{"TEXT": "№"}]]

    def __call__(self, doc):
        spans, occupied = [], []
        for _, start, end in self.matcher(doc):  # token index of «№»
            # ------- optional headword on the left ----------------------
            left = start
            if left - 1 >= 0 and doc[left - 1].lemma_.lower() in _HEAD_BASES:
                left -= 1

            # ------- expand to the right over id fragments --------------
            right = end
            # skip optional single space after №
            if right < len(doc) and doc[right].is_space:
                right += 1
            # absorb sequence of tokens that look like      947 / 40220 / 21  or 1-кс
            while right < len(doc):
                tok = doc[right]
                if _ID_TOKEN_RE.match(tok.text):
                    right += 1
                    # skip inter-token spaces
                    if right < len(doc) and doc[right].is_space:
                        right += 1
                    continue
                break
            # trim trailing spaces from span
            while right > left and doc[right - 1].is_space:
                right -= 1

            span = doc[left:right]
            span.label_ = self.label
            # skip if overlaps existing entity
            if any(not (right <= s or left >= e) for s, e in occupied):
                continue
            spans.append(span)
            occupied.append((left, right))

        if spans:
            doc.ents = filter_spans(list(doc.ents) + spans)
        return doc
