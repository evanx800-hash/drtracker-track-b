"""Controlled vocabulary shared by the generator, parsers, validator and dashboard.

Scored items mirror the Dr.Tracker oncology demo check-in (mouth sores 0-4,
pain 0-10, swallowing ladder, "also today" checkboxes, weight). Nausea (0-4) and
vomiting (episode count) are modeled as clinic-configured add-on questions,
since cisplatin is part of the regimen.
"""

# ---------------------------------------------------------------- symptoms
SCORED = {
    # code: (label, min, max, unit/labels)
    "mouth_sores": ("Mouth sores", 0, 4),
    "pain": ("Pain", 0, 10),
    "swallowing": ("Swallowing", 0, 3),
    "nausea": ("Nausea", 0, 4),
}
COUNTED = {"vomiting": ("Vomiting", "episodes")}
BOOLEAN = {
    "dry_mouth": "Dry mouth",
    "taste_change": "Taste change",
    "thick_saliva": "Thick saliva",
    "fever": "Fever",
    "fatigue": "Fatigue",
}
# Captured but never mapped to a specific symptom ("feel kinda off").
VAGUE = {"general_unwell": "Feeling unwell (unspecified)"}

SYMPTOMS = list(SCORED) + list(COUNTED) + list(BOOLEAN) + list(VAGUE)
LABELS = {**{k: v[0] for k, v in SCORED.items()},
          **{k: v[0] for k, v in COUNTED.items()},
          **BOOLEAN, **VAGUE}

MOUTH_SORES_LABELS = {
    0: "None",
    1: "Sore or red · eating normally",
    2: "Sores · soft food OK",          # wording from the demo
    3: "Sores · liquids only",
    4: "Can't eat or drink",            # wording from the demo
}
SWALLOWING_LABELS = {0: "Solids", 1: "Soft food", 2: "Liquids only", 3: "Nothing"}
NAUSEA_LABELS = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Very severe"}

# Free-text intensity words. Deliberately coarse: a patient's words are kept as
# words, and only mapped onto a scale for display, clearly marked as "from text".
INTENSITIES = ["none", "mild", "moderate", "severe"]
INTENSITY_FRACTION = {"none": 0.0, "mild": 0.33, "moderate": 0.66, "severe": 1.0}

TIMES_OF_DAY = ["morning", "afternoon", "evening", "night"]

# Synonyms -> canonical code (used by the rule parser and the prompt).
SYNONYMS = {
    "nausea": ["nausea", "nauseous", "nauseated", "queasy", "sick to my stomach",
               "stomach unsettled", "stomach a little unsettled", "stomach's been fine",
               "stomach has been fine", "felt sick", "feel sick", "feeling sick"],
    "vomiting": ["vomit", "vomited", "vomiting", "threw up", "throw up", "throwing up",
                 "puked", "puke", "puking", "got sick", "kept everything down", "kept food down"],
    "mouth_sores": ["mouth sores", "mouth sore", "sores", "ulcers", "mouth is sore",
                    "mouth is a bit sore", "redness in my mouth", "mouth is raw", "mouth hurts"],
    "pain": ["pain", "hurts", "hurting", "achy", "aching", "sore throat", "throat really hurts"],
    "swallowing": ["swallow", "swallowing", "soft food", "soft foods", "liquids", "only drink",
                   "get food down"],
    "dry_mouth": ["dry mouth", "mouth so dry", "mouth is dry", "mouth is so dry", "sipping water"],
    "taste_change": ["taste", "tastes", "metal", "metallic"],
    "thick_saliva": ["thick saliva", "saliva", "spit", "mucus", "phlegm"],
    "fever": ["fever", "feverish", "temperature", "temp of", "chills"],
    "fatigue": ["tired", "exhausted", "wiped out", "fatigue", "no energy", "drained"],
    "general_unwell": ["kinda off", "not great", "rough day", "rough one", "feel off",
                       "feeling off", "under the weather", "blah"],
}

# ---------------------------------------------------------------- medications
MEDICATIONS = {
    "cisplatin": {"class": "chemotherapy", "aliases": ["cisplatin", "chemo"]},
    "ondansetron": {"class": "antiemetic", "aliases": ["ondansetron", "zofran", "zofrin", "zofren"]},
    "olanzapine": {"class": "antiemetic", "aliases": ["olanzapine", "zyprexa", "new nausea pill", "new pill"]},
    "oxycodone": {"class": "opioid analgesic", "aliases": ["oxycodone", "oxy", "pain pill", "pain meds"]},
    "acetaminophen": {"class": "analgesic", "aliases": ["acetaminophen", "tylenol"]},
    "lidocaine_rinse": {"class": "topical anesthetic", "aliases": ["magic mouthwash", "lidocaine", "numbing rinse"]},
}
MED_ACTIONS = ["took", "missed", "started", "stopped", "helped", "did_not_help"]


def scale_fraction(symptom: str, value: float) -> float:
    """Map a questionnaire score to 0-1 for colouring only (hover shows the real value)."""
    if symptom in SCORED:
        _, lo, hi = SCORED[symptom]
        return (value - lo) / (hi - lo)
    if symptom == "vomiting":
        return min(value, 4) / 4
    return 1.0 if value else 0.0
