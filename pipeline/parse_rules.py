"""Rule-based baseline parser: lexicon + clause-scoped negation + simple number/time cues.

It is fast, deterministic, needs no model, and is the offline fallback. It is also
the cross-check for the LLM: when the two disagree, the observation goes to review.
"""
from __future__ import annotations

import re

from .models import Extraction, MedicationMention, SymptomMention

A = "'?"  # optional apostrophe: didn't / didnt

SYMPTOM_PATTERNS = {
    "vomiting": rf"vomit\w*|vomité|threw up|throw(?:ing)? up|thrown up|puk\w*|got sick (?:once|twice|\w+ times)|kept (?:everything|food|it|anything) down|keep (?:anything|food) down",
    "nausea": rf"nause\w*|náusea\w*|queasy|sick to my stomach|stomach{A}?s? (?:has |is |feels )?(?:been )?(?:a little )?(?:unsettled|fine|upset)|(?:felt|feel|feeling) sick|sick all \w+",
    "mouth_sores": r"mouth sores?|\bsores?\b|\bsres\b|ulcers?|redness in my mouth|mouth is (?:a bit )?sore|mouth is raw|white patches",
    "pain": r"\bpain\b|\bhurts?\b|hurting|achy|aching|throat (?:is )?on fire",
    "swallowing": r"swallow\w*|soft foods?|soft stuff|only (?:managing |do |get )?(?:liquids|shakes|smoothies)|liquids down|smoothies|broth|get (?:food|water) down|water down",
    "dry_mouth": r"dry mouth|mouth (?:is |so )?(?:so )?dry|mouth is a desert|sipping water",
    "taste_change": r"\btastes?\b|metallic|like metal|cardboard",
    "thick_saliva": r"thick (?:saliva|spit|mucus)|spit is (?:super |really )?thick|\bmucus\b|phlegm",
    "fever": r"fever\w*|\btemp(?:erature)?\b|chills",
    "fatigue": r"\btired\b|exhausted|wiped out|fatigue\w*|no energy|drained",
    "general_unwell": r"kinda off|feel(?:ing)? off|not great|rough (?:day|one)|under the weather|\bblah\b",
}
NEG = re.compile(rf"\b(no|not|didn{A}t|did not|never|without|none|isn{A}t|wasn{A}t|zero|nothing)\b")
ABSENT_PHRASES = {
    "vomiting": re.compile(rf"kept (everything|food|it) down"),
    "nausea": re.compile(rf"stomach{A}?s? (has |is |feels )?(been )?(fine|ok|good)|nausea (is )?gone|feels fine"),
    "swallowing": re.compile(r"swallowing (is )?(fine|ok|normal)"),
}
NO_NEGATION = {"general_unwell", "taste_change"}   # "not great", "can't taste anything"

SEVERE = re.compile(rf"horrible|awful|terrible|worst|unbearable|killing|so bad|really bad|barely|can{A}t|couldn{A}t|on fire|excruciating|severe|everywhere")
MODERATE = re.compile(r"pretty|on and off|all (day|morning|evening|night)|most of|constant|getting worse|really hurts|worse")
MILD = re.compile(r"\blittle\b|\bbit\b|slightly|\bmild\b|small|starting|harder")
SWALLOW_LEVELS = [
    ("severe", re.compile(rf"can{A}t swallow|couldn{A}t (even )?get (water|anything) down|nothing down")),
    ("moderate", re.compile(r"liquids|shakes|smoothies|broth")),
    ("mild", re.compile(r"soft|harder")),
]
COUNT_WORDS = {"once": 1, "one time": 1, "una vez": 1, "twice": 2, "two times": 2,
               "three times": 3, "four times": 4, "five times": 5}
TIME_CUES = [
    (re.compile(r"two days ago"), -2, None),
    (re.compile(r"last night|overnight|in the night|all night|by bedtime"), -1, "night"),
    (re.compile(r"yesterday"), -1, None),
    (re.compile(r"this morning|this am\b|when i woke up|in the morning|all morning|toda la mañana"), 0, "morning"),
    (re.compile(r"after lunch|this afternoon"), 0, "afternoon"),
    (re.compile(r"after dinner|tonight|this evening|all evening"), 0, "evening"),
]
MED_PATTERNS = {
    "ondansetron": r"ondansetron|zofr[aie]n",
    "olanzapine": r"olanzapine|zyprexa|new nausea pill",
    "oxycodone": r"oxycodone|\boxy\b|pain pills?|pain meds",
    "acetaminophen": r"acetaminophen|tylenol",
    "lidocaine_rinse": r"magic mouthwash|lidocaine|numbing rinse",
}
CLAUSE_SPLIT = re.compile(r"(?<!\d)\.(?!\d)|[!?;,]+|\.\.\.|\s+-\s+|\bbut\b|\band\b|\bthen\b|\bsince\b")


def _clauses(text: str):
    """Yield (start, end) spans of clauses in the original text."""
    pos = 0
    for m in CLAUSE_SPLIT.finditer(text.lower()):
        if m.start() > pos:
            yield pos, m.start()
        pos = m.end()
    if pos < len(text):
        yield pos, len(text)


def _intensity(symptom: str, clause: str):
    if symptom == "swallowing":
        for level, rx in SWALLOW_LEVELS:
            if rx.search(clause):
                return level
        return None
    if symptom in ("vomiting", "dry_mouth", "taste_change", "thick_saliva", "fever",
                   "fatigue", "general_unwell"):
        return None
    if SEVERE.search(clause):
        return "severe"
    if MODERATE.search(clause):
        return "moderate"
    if MILD.search(clause):
        return "mild"
    return None


def _count(clause: str):
    m = re.search(r"(\d+)\s*(?:times|x)\b", clause)
    if m:
        return int(m.group(1))
    for w, n in COUNT_WORDS.items():
        if re.search(rf"\b{w}\b", clause):
            return n
    return None


def _value(symptom: str, clause: str):
    if symptom == "pain":
        m = re.search(r"(\d+(?:\.\d)?)\s*(?:/|out of)\s*10", clause) or \
            re.search(r"pain\s+(?:is\s+)?(?:like\s+)?(?:a\s+|around\s+|at\s+)?(?:like\s+)?(?:a\s+)?(\d+(?:\.\d)?)\b", clause)
        if m and 0 <= float(m.group(1)) <= 10:
            return float(m.group(1))
    if symptom == "mouth_sores":
        m = re.search(r"(\d)\s*(?:/|out of)\s*4", clause)
        if m:
            return float(m.group(1))
    if symptom == "fever":
        m = re.search(r"(\d{2,3}\.\d)", clause)
        if m:
            return float(m.group(1))
    return None


def _time(clause: str, full_text: str):
    for rx, rel, tod in TIME_CUES:
        if rx.search(clause):
            if rel == -1 and tod is None and re.search(r"evening", clause):
                return -1, "evening"
            return rel, tod
    return 0, None


def parse(text: str, submitted_at: str | None = None) -> Extraction:
    symptoms, meds, seen = [], [], set()
    low = text.lower()
    for s, e in _clauses(text):
        clause, raw = low[s:e], text[s:e].strip()
        if not raw:
            continue
        # mask medication names so "nausea pill" / "pain meds" don't read as symptoms
        sym_clause = clause
        for pat in MED_PATTERNS.values():
            sym_clause = re.sub(pat, lambda mm: " " * len(mm.group(0)), sym_clause)
        for sym, pat in SYMPTOM_PATTERNS.items():
            m = re.search(pat, sym_clause)
            if not m:
                continue
            # avoid double-counting "sick" phrases and "sores" inside "mouth sores"
            if sym == "nausea" and re.search(SYMPTOM_PATTERNS["vomiting"], clause) and "nause" not in clause:
                continue
            if sym == "fever" and "flu shot" in clause:
                continue
            present = True
            absent_rx = ABSENT_PHRASES.get(sym)
            if absent_rx and absent_rx.search(clause):
                present = False
            elif sym not in NO_NEGATION and NEG.search(clause[: m.start()]):
                present = False
            rel, tod = _time(clause, low)
            mention = SymptomMention(
                symptom=sym, present=present,
                intensity=("none" if not present else _intensity(sym, clause)),
                count=_count(clause) if (sym == "vomiting" and present) else None,
                value_num=_value(sym, clause) if present else None,
                relative_day=rel, time_of_day=tod, evidence=raw, confidence=0.7)
            key = (sym, rel, tod, present, mention.count, mention.value_num)
            if key not in seen:
                seen.add(key)
                symptoms.append(mention)
        # elliptical follow-up: "threw up twice last night and once this morning"
        if not any(m.evidence == raw for m in symptoms) and symptoms and \
                symptoms[-1].symptom == "vomiting" and symptoms[-1].present and _count(clause):
            rel, tod = _time(clause, low)
            symptoms.append(SymptomMention(symptom="vomiting", present=True, count=_count(clause),
                                           relative_day=rel, time_of_day=tod, evidence=raw,
                                           confidence=0.6))
        for med, pat in MED_PATTERNS.items():
            if re.search(pat, clause):
                if re.search(r"skip|missed|forgot", clause):
                    action = "missed"
                elif re.search(rf"barely help|didn{A}t help|not helping|doesn{A}t help", clause):
                    action = "did_not_help"
                elif re.search(r"help|working|edge off", clause):
                    action = "helped"
                elif re.search(r"start", clause):
                    action = "started"
                else:
                    action = "took"
                meds.append(MedicationMention(medication=med, action=action, evidence=raw))
    return Extraction(symptoms=symptoms, medications=meds)
