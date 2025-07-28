from __future__ import annotations
import re
from spacy.language import Language
from spacy.util import filter_spans
from src.legal_graph.processors.entities import BaseMatcherComponent

_HEAD_BASES = {"справа", "провадження"}
_ID_RE = re.compile(r"^[0-9a-zа-яіїєґ\-—/]+$", re.I)
_SEP = {"/", "-", "—"}

@Language.factory("case_component")
class CaseComponent(BaseMatcherComponent):
    label = "CASEID"
    patterns = [[{"TEXT": "№"}]]  # seed

    # ------------------------------------------------------------------
    # helpers7
    # ------------------------------------------------------------------
    @staticmethod
    def _is_id_token(tok):
        txt = tok.text.lower()
        if tok.like_num:
            return True
        if _ID_RE.fullmatch(txt):
            # allow short alpha pieces (кс) or mixed alphanum
            return True
        return False

    # ------------------------------------------------------------------
    def __call__(self, doc):
        spans, occupied = [], []
        for _, pivot, _ in self.matcher(doc):        # pivot index of «№»
            # ----- LEFT expansion (≤2 spaces) -------------------------
            left = pivot
            idx = pivot - 1
            spaces = 0
            while idx >= 0 and doc[idx].is_space and spaces < 2:
                idx -= 1; spaces += 1
            if idx >= 0 and doc[idx].lemma_.lower() in _HEAD_BASES:
                left = idx
            # otherwise keep original pivot as left

            # ----- RIGHT expansion (≤1 space before first id token) ----
            right = pivot + 1
            # optional single space
            if right < len(doc) and doc[right].is_space:
                right += 1
            # if another space immediately -> invalid pattern, skip
            if right < len(doc) and doc[right].is_space:
                continue
            # now capture identifier chain
            while right < len(doc):
                tok = doc[right]
                if tok.is_space:
                    break  # stop on any space inside id
                if tok.text in _SEP or self._is_id_token(tok):
                    right += 1
                    continue
                break
            # backtrack trailing separators
            while right > pivot and doc[right-1].text in _SEP:
                right -= 1
            if right <= pivot + 1:  # nothing captured
                continue

            span = doc[left:right]
            span.label_ = self.label
            if any(not (right <= s or left >= e) for s, e in occupied):
                continue
            spans.append(span); occupied.append((left, right))

        if spans:
            doc.ents = filter_spans(list(doc.ents) + spans)
        return doc
