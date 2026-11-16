"""LLM extraction through a local Ollama model with JSON-schema-constrained output.

- Runs locally: report text never leaves the machine.
- Output is constrained to the `Extraction` JSON schema, then re-validated by Pydantic.
- temperature 0, one retry that feeds the validation error back to the model.
- Results are cached on disk, so rebuilding the database doesn't re-run the model.

Setup (once):  ollama pull qwen2.5:7b      (or qwen2.5:3b on an 8 GB laptop)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests
from pydantic import ValidationError

from .models import Extraction

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "llm_cache.jsonl"
PROMPT_VERSION = "v3"

SYSTEM = """You convert a cancer patient's daily check-in message into structured data.
The patient is receiving head & neck radiation with weekly cisplatin chemotherapy.

Extract ONLY what the patient states. Never diagnose, never infer causes, never guess.

Symptom codes:
- mouth_sores: sores, ulcers, redness, raw mouth, patches in the mouth
- pain: pain, hurting, aching, burning throat
- swallowing: difficulty swallowing or diet limited to soft food / liquids
- nausea: nausea, queasy, sick to the stomach ("felt sick" without throwing up = nausea)
- vomiting: threw up, vomited, puked, "got sick N times"; "kept everything down" = absent
- dry_mouth, taste_change, thick_saliva (thick spit/mucus), fever (fever, chills, temperature), fatigue
- general_unwell: vague statements like "feel off", "not great", "rough day". Do NOT map these to nausea.

Field rules:
- present=false only when the patient explicitly says the symptom is absent ("no vomiting").
- intensity: none | mild | moderate | severe, ONLY when words support it
  (mild: a little, slightly, a bit; moderate: pretty, on and off, all day, most of the day;
   severe: horrible, worst, unbearable, can't eat, killing me). Otherwise null.
  swallowing: mild = soft food only/harder, moderate = liquids only, severe = can't swallow anything.
- count: number of vomiting episodes, only if a specific number is stated ("a few" -> null).
- value_num: a number the patient gave: pain 0-10, mouth sores 0-4, temperature in F. Else null.
- relative_day: 0 = the day of the report; -1 = yesterday / last night / overnight; -2 = two days ago.
- time_of_day: morning | afternoon | evening | night, only if stated.
- If the same symptom is reported for two different days or times, output two entries.
- Symptoms of OTHER people (spouse, family) are not extracted.
- evidence: copy the exact words from the message that support the entry.
- confidence: 0-1, lower it when the statement is uncertain or hedged.
- medications: ondansetron (zofran), olanzapine (zyprexa, "new nausea pill"), oxycodone (oxy),
  acetaminophen (tylenol), lidocaine_rinse (magic mouthwash), cisplatin (chemo).
  action: took | missed | started | stopped | helped | did_not_help.
Return an empty list when nothing is reported."""

EXAMPLES = [  # written for the prompt; none of these appear in the evaluation sets
    ("Morning report. Message: \"queasy most of the day but no throwing up. mouth is sore, pain about 5/10\"",
     {"symptoms": [
         {"symptom": "nausea", "present": True, "intensity": "moderate", "count": None, "value_num": None,
          "relative_day": 0, "time_of_day": None, "evidence": "queasy most of the day", "confidence": 0.9},
         {"symptom": "vomiting", "present": False, "intensity": "none", "count": None, "value_num": None,
          "relative_day": 0, "time_of_day": None, "evidence": "no throwing up", "confidence": 0.9},
         {"symptom": "mouth_sores", "present": True, "intensity": None, "count": None, "value_num": None,
          "relative_day": 0, "time_of_day": None, "evidence": "mouth is sore", "confidence": 0.8},
         {"symptom": "pain", "present": True, "intensity": None, "count": None, "value_num": 5,
          "relative_day": 0, "time_of_day": None, "evidence": "pain about 5/10", "confidence": 0.95}],
      "medications": []}),
    ("Evening report. Message: \"forgot my zofran yesterday, threw up 3 times last night. feeling blah\"",
     {"symptoms": [
         {"symptom": "vomiting", "present": True, "intensity": None, "count": 3, "value_num": None,
          "relative_day": -1, "time_of_day": "night", "evidence": "threw up 3 times last night", "confidence": 0.9},
         {"symptom": "general_unwell", "present": True, "intensity": None, "count": None, "value_num": None,
          "relative_day": 0, "time_of_day": None, "evidence": "feeling blah", "confidence": 0.7}],
      "medications": [{"medication": "ondansetron", "action": "missed", "evidence": "forgot my zofran yesterday"}]}),
]


def _key(text, submitted_at, model):
    return hashlib.sha256(f"{PROMPT_VERSION}|{model}|{submitted_at}|{text}".encode()).hexdigest()


def _load_cache():
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["key"]] = rec
    return out


_CACHE = None


def available(model: str = DEFAULT_MODEL) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        names = [m["name"] for m in r.json().get("models", [])]
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    except Exception:
        return False


def _call(messages, model):
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=180, json={
        "model": model, "messages": messages, "stream": False,
        "format": Extraction.model_json_schema(),
        "options": {"temperature": 0, "seed": 7}})
    r.raise_for_status()
    return r.json()["message"]["content"]


def _when(submitted_at):
    if not submitted_at:
        return "Report"
    hour = int(submitted_at[11:13])
    return "Morning report" if hour < 12 else ("Afternoon report" if hour < 17 else "Evening report")


def parse(text: str, submitted_at: str | None = None, model: str = DEFAULT_MODEL,
          use_cache: bool = True) -> tuple[Extraction | None, dict]:
    """Returns (extraction or None, audit dict with raw output / error / latency)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_cache()
    key = _key(text, submitted_at, model)
    if use_cache and key in _CACHE:
        rec = _CACHE[key]
        try:
            return Extraction.model_validate_json(rec["raw"]), {**rec, "cached": True}
        except ValidationError:
            pass

    messages = [{"role": "system", "content": SYSTEM}]
    for user, answer in EXAMPLES:
        messages += [{"role": "user", "content": user},
                     {"role": "assistant", "content": json.dumps(answer)}]
    messages.append({"role": "user", "content": f"{_when(submitted_at)}. Message: \"{text}\""})

    audit = {"key": key, "model": model, "prompt_version": PROMPT_VERSION, "attempts": 0}
    t0 = time.time()
    raw, err = None, None
    for attempt in range(2):
        audit["attempts"] = attempt + 1
        try:
            raw = _call(messages, model)
            ext = Extraction.model_validate_json(raw)
            audit.update(raw=raw, latency_s=round(time.time() - t0, 2), error=None)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE, "a") as f:
                f.write(json.dumps(audit) + "\n")
            _CACHE[key] = audit
            return ext, audit
        except ValidationError as e:
            err = str(e)
            messages += [{"role": "assistant", "content": raw or ""},
                         {"role": "user", "content": f"That output failed validation: {err[:600]}. "
                                                     "Return corrected JSON only."}]
        except Exception as e:  # connection problems, timeouts
            err = f"{type(e).__name__}: {e}"
            break
    audit.update(raw=raw, latency_s=round(time.time() - t0, 2), error=err)
    return None, audit
