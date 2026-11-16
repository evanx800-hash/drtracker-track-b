# Dr.Tracker · Track B — Patient-reported data: store, manage & chart

A working prototype that takes what head-and-neck cancer patients report during radiation + weekly cisplatin (a check-in form and free-text messages), turns it into structured, validated rows in SQLite, and shows a provider what changed since the last visit in about 30 seconds.

Everything is synthetic. The scenario, scales and alert line are modeled on the public Dr.Tracker oncology demo (James: mouth sores 0–4, pain 0–10, a swallowing ladder, "also today" checkboxes, weight against a 165 lb alert line).

---

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pipeline.build_db            # builds data/drtracker.db with the rule parser (instant, offline)
streamlit run app/streamlit_app.py     # opens http://localhost:8501
```

### With the local LLM (recommended, free, nothing leaves the laptop)

```bash
# install Ollama from https://ollama.com (or: brew install ollama), then:
ollama pull qwen2.5:7b                 # ~4.7 GB; on an 8 GB machine use qwen2.5:3b and set OLLAMA_MODEL=qwen2.5:3b
python -m pipeline.build_db --backend ollama    # parses every message with the LLM, cross-checked by rules
python -m eval.evaluate                          # rules scores
python -m eval.evaluate --backend ollama         # LLM scores (results appear on the Evaluation page)
python -m pytest -q                              # unit tests
```

LLM outputs are cached in `data/cache/llm_cache.jsonl`, so rebuilding and the live demo don't re-run the model for messages it has already seen.

Optional: `python -m data.generate` regenerates the synthetic dataset (fixed seed, so it's reproducible).

---

## What's in the box

| Path | What it does |
|---|---|
| `data/generate.py` | Synthetic dataset: 5 patients, ~7 weeks, ~150 check-ins + ~105 free-text messages. **Truth is generated first, then rendered into patient-voice text**, so every message has gold labels. |
| `data/hard_cases.jsonl` | 30 hand-written edge cases (negation scope, vague counts, self-corrections, other people's symptoms, Spanish, hedging). |
| `data/holdout_cases.jsonl` | 20 more cases written *after* the parser rules and LLM prompt were frozen — the honest test. |
| `pipeline/schema.sql` | Event-based SQLite schema. |
| `pipeline/models.py` | Pydantic extraction contract (vocabulary, ranges, coherence rules). |
| `pipeline/parse_rules.py` | Rule-based parser: lexicon + clause-scoped negation. Offline fallback and cross-check. |
| `pipeline/parse_llm.py` | Ollama parser with JSON-schema-constrained output, temperature 0, one validated retry, disk cache. |
| `pipeline/validate.py` | Grounding, hedging, safety, confidence and cross-check rules that decide auto-accept vs review. |
| `pipeline/build_db.py` | Loads everything, parses messages, builds the `daily_symptoms` rollup with conflict detection. |
| `pipeline/queries.sql` | Seven named queries (vomiting per week, nausea around a medication change, nausea by days since cisplatin, weight vs alert line, adherence, changed-since-last-visit, review workload). |
| `eval/evaluate.py` | Precision/recall plus field-level accuracy, review-flag catch rate, error log. |
| `app/` | Streamlit dashboard: patient overview, live parser, review queue, queries, evaluation, schema. |

---

## Design decisions

### 1. Dataset: truth first, then text
The generator models each patient's daily state (mouth sores rising with radiation dose, pain following sores, nausea peaking 1–3 days after each cisplatin dose, olanzapine added on day 16 for one patient, weight falling with eating difficulty). It then writes the check-in form answers and renders some days into free text from phrase banks, adding typos, missing apostrophes, vague statements, negations, relative times ("last night"), skipped days and deliberate same-day conflicts between the form and a message. Because the labels describe exactly what each message says, the parser can be scored instead of eyeballed.

Patients have different profiles on purpose: James mirrors the demo (below the 165 lb alert line on day 25); Maria has severe mucositis and starts oxycodone; Robert is nausea-heavy; Linda is mild and consistent; David checks in irregularly, writes vaguely, has a 3-day gap and one fever.

### 2. Schema: event-based, append-only
- `reports` stores every submission exactly as received and is never edited.
- `symptom_observations` has **one row per symptom statement**, linked to its report, with the exact supporting words (`evidence`), the parser that produced it, its confidence and a review status.
- Clinic-side facts (`treatment_events`, `medication_events`) are kept apart from patient-reported medication behaviour (`medication_mentions`: took / missed / helped).
- `daily_symptoms` is a derived rollup for charting, rebuilt by documented rules and never edited by hand.
- `extraction_runs` and `rejected_extractions` keep an audit trail, including raw model output and failures.

An event model was chosen over one wide row per day because reports arrive at odd times, one message can describe two different days, and adding a symptom shouldn't require a migration.

### 3. Vague and conflicting reports
- **Unknown stays unknown.** Score, intensity and count are NULL unless the patient stated them ("threw up a few times" → present, count NULL).
- **Vague complaints** ("feel kinda off") are stored as `general_unwell` and never mapped to a specific symptom.
- **Absent is not the same as unmentioned.** "No vomiting" is a row with `present = 0`; silence is no row; a missed check-in is a gray gap on the chart, not a zero.
- **Scores vs words.** Form answers are scores. Free text keeps the patient's intensity word; a number is stored only if the patient typed one. Text-only days are drawn with a ○ so nobody mistakes them for a form score. This follows the demo console's principle that nothing shown is scored or inferred.
- **Conflicts are surfaced, not resolved.** If the form says nausea 1 and a message the same morning says "worst nausea so far", both rows stay. The chart keeps the form score, marks the cell ◆ and quotes both sources. Vomiting counts from the form (last 24 h) and messages overlap in time, so the rollup shows the larger number and notes the difference instead of adding them.

### 4. Parser and validation
The LLM (local `qwen2.5:7b` through Ollama) is constrained to the Pydantic JSON schema at decode time, re-validated afterwards, and retried once with the validation error. Every extraction then passes through `validate.py`:

1. **Schema**: unknown symptoms, out-of-range numbers, "absent but severe" and counts on non-vomiting symptoms are rejected.
2. **Grounding**: the quoted evidence must appear in the message; otherwise the item is rejected and logged. This catches hallucinated symptoms.
3. **Hedging**: "not sure", "maybe", questions → review.
4. **Safety**: fever from free text is never auto-accepted.
5. **Cross-check**: if the rule parser disagrees on present/absent, or finds a symptom the LLM missed, → review.
6. **Confidence** below 0.6 → review.

Anything flagged waits in the review queue and is not charted until a person confirms it.

### 5. Dashboard: the provider's 30 seconds
The overview answers "what changed since I last saw this patient?":
- A **since-last-visit panel** produced by deterministic rules, with no model output: weight vs alert line, fever, symptoms that worsened past a threshold, vomiting totals, new symptoms, form/message conflicts, prescription changes, missed check-ins, items awaiting review.
- A **symptom heatmap** (rows = symptoms, columns = days, darker = worse on that row's own scale) with cisplatin doses and prescription changes as vertical lines, so the post-chemo nausea pattern and the effect timing of a new medication are visible without reading numbers. Hover shows the actual answer and the patient's words.
- **Weight against the clinic's alert line**, on the same time axis.
- **What the patient wrote**, with what each message was captured as.

---

## Evaluation (rule parser; run the LLM eval to fill in the second column)

| | generated | hard_dev | holdout |
|---|---|---|---|
| F1 | 0.982 | 0.951 | **0.647** |
| Recall | 0.964 | 0.951 | **0.478** |
| Precision | 1.00 | 0.951 | 1.00 |

The rule parser looks excellent on generated text because it shares vocabulary with the generator, and on hard_dev because its lexicon was tuned on those cases. On the untouched holdout it misses half of what patients say ("brought up my lunch", "cheeks are all blistered", "saliva is like glue"). That gap is the reason for the LLM, and the reason the rule parser is kept only as a cross-check and offline fallback.

---

## AI usage

- **Where it's used in the product:** extracting structured rows from free text (local LLM).
- **How it's checked:** schema-constrained decoding, Pydantic validation, an evidence-grounding check, a rule-based cross-check, a review queue, and scoring against gold labels including a holdout set.
- **Where it's deliberately not used:** no diagnosis, no clinical interpretation, no AI-written summary on the overview, no AI deciding what to flag. The since-last-visit callouts and conflict flags are plain rules that can be read and audited.
- **Building it:** the code and the hand-labeled edge cases were drafted with an AI coding assistant, then reviewed, run and tested. Labels for the generated messages come from code with a fixed seed (the text is rendered from the labels), so they are exact by construction rather than judged by a model.

---

## Trade-offs and next steps

- **SQLite** keeps the demo zero-setup; production would be Postgres with row-level access control and audit logging.
- **Rules for rollup and alerts** are transparent but blunt (e.g., thresholds are fixed per symptom); a practice would configure them per protocol, the same way the demo's alert line is clinic-set.
- **A local LLM** keeps text on the machine but is slower (~seconds per message) than a hosted model; extraction runs once at ingest and is cached.
- **Intensity words are coarse** by design. Mapping the vocabulary to a published patient-reported outcome instrument (for example the NCI's PRO-CTCAE) and to standard codes would make the data interoperable; each observation already maps naturally to a FHIR `Observation` with the report as a `QuestionnaireResponse`.
- **Not covered:** authentication, HIPAA controls (no compliance is claimed), real patient data, voice transcription, photos, multilingual evaluation beyond one Spanish case.
