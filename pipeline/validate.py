"""Validation layers applied to every parser output before it is stored.

1. Schema       - Pydantic (models.py) already rejected invalid codes, ranges, combinations.
2. Grounding    - evidence must actually appear in the report text; otherwise rejected.
3. Hedging      - "not sure", "maybe", questions -> needs review.
4. Safety       - fever is never auto-accepted from free text.
5. Cross-check  - LLM vs rules disagreement on polarity, or a symptom only one found -> review.
6. Confidence   - below 0.6 -> review.

Nothing is silently dropped: rejected items are returned so they can be audited.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .models import Extraction

HEDGE = re.compile(r"not sure|maybe|i think|hard to say|\?|might have|kind of|sort of")
SAFETY = {"fever"}
LOW_CONF = 0.6


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[\"“”]", "", s)
    return re.sub(r"\s+", " ", s).strip(" .,!;:")


def grounded(evidence: str, text: str) -> tuple[bool, float]:
    ev, tx = _norm(evidence), _norm(text)
    if not ev:
        return False, 0.0
    if ev in tx:
        return True, 1.0
    # tolerate tiny copy differences (dropped apostrophe, typo fix) via best window match
    n = len(ev)
    best = max((SequenceMatcher(None, ev, tx[i:i + n]).ratio()
                for i in range(0, max(1, len(tx) - n + 1))), default=0.0)
    return best >= 0.85, round(best, 2)


def validate(ext: Extraction, text: str, method: str, cross_check: Extraction | None = None):
    accepted, rejected = [], []
    for m in ext.symptoms:
        obs = m.model_dump()
        obs["method"] = method
        reasons = []
        ok, score = grounded(m.evidence, text)
        if not ok:
            rejected.append({**obs, "review_status": "rejected",
                             "review_reason": f"evidence not found in report (match {score})"})
            continue
        if m.confidence < LOW_CONF:
            reasons.append(f"low confidence ({m.confidence:.2f})")
        if HEDGE.search(m.evidence.lower()) or (HEDGE.search(text.lower()) and m.present):
            reasons.append("patient is unsure / hedged")
        if m.symptom in SAFETY and m.present:
            reasons.append("safety-relevant symptom: confirm before charting")
        if m.count is not None and m.count > 10:
            reasons.append("unusually high count")
        if cross_check is not None:
            same = [c for c in cross_check.symptoms if c.symptom == m.symptom]
            if same and all(c.present != m.present for c in same):
                reasons.append("rule parser disagrees on present/absent")
        obs["review_status"] = "needs_review" if reasons else "auto_accepted"
        obs["review_reason"] = "; ".join(reasons) or None
        accepted.append(obs)

    # symptoms the cross-check found that the primary parser missed
    if cross_check is not None:
        found = {o["symptom"] for o in accepted} | {o["symptom"] for o in rejected}
        for c in cross_check.symptoms:
            if c.symptom not in found:
                obs = c.model_dump()
                obs.update(method="rules", review_status="needs_review",
                           review_reason="found by rule parser only; LLM did not extract it")
                accepted.append(obs)
                found.add(c.symptom)
    return accepted, rejected
