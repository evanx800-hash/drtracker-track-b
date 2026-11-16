"""Load the synthetic data into SQLite, parse free text, validate, and build the daily rollup.

    python -m pipeline.build_db                    # rules parser (offline, instant)
    python -m pipeline.build_db --backend ollama   # local LLM + rules cross-check
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from . import parse_rules, parse_llm
from .validate import validate
from .vocab import (BOOLEAN, INTENSITY_FRACTION, MEDICATIONS, SCORED, scale_fraction)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "drtracker.db"
INTENSITY_ORDER = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ----------------------------------------------------------------- parsing
def run_parser(text: str, submitted_at: str, backend: str, model: str | None = None):
    """Returns (accepted_obs, rejected_obs, med_mentions, run_record)."""
    rules_ext = parse_rules.parse(text, submitted_at)
    if backend == "rules":
        acc, rej = validate(rules_ext, text, "rules")
        run = dict(method="rules", status="ok", raw_output=rules_ext.model_dump_json(),
                   error=None, latency_s=0.0)
        return acc, rej, [(m, "rules") for m in rules_ext.medications], run
    model = model or parse_llm.DEFAULT_MODEL
    ext, audit = parse_llm.parse(text, submitted_at, model=model)
    method = f"llm:{model}"
    run = dict(method=method, status="ok" if ext else "failed", raw_output=audit.get("raw"),
               error=audit.get("error"), latency_s=audit.get("latency_s"))
    if ext is None:
        # LLM failed twice: fall back to rules, but nothing is auto-accepted.
        acc, rej = validate(rules_ext, text, "rules")
        for o in acc:
            o["review_status"] = "needs_review"
            o["review_reason"] = "LLM extraction failed; rules fallback"
        return acc, rej, [(m, "rules") for m in rules_ext.medications], run
    acc, rej = validate(ext, text, method, cross_check=rules_ext)
    return acc, rej, [(m, method) for m in ext.medications], run


def insert_free_text(conn, report_id, patient_id, submitted_at, text, backend, model=None):
    conn.execute("INSERT INTO reports VALUES (?,?,?,?,?,NULL)",
                 (report_id, patient_id, submitted_at, "free_text", text))
    acc, rej, meds, run = run_parser(text, submitted_at, backend, model)
    conn.execute("INSERT INTO extraction_runs (report_id, method, status, raw_output, error, latency_s) "
                 "VALUES (?,?,?,?,?,?)", (report_id, run["method"], run["status"], run["raw_output"],
                                          run["error"], run["latency_s"]))
    day = datetime.fromisoformat(submitted_at).date()
    for o in acc:
        occurred = day + timedelta(days=o["relative_day"])
        conn.execute(
            "INSERT INTO symptom_observations (report_id, patient_id, symptom, present, score, intensity,"
            " episode_count, occurred_date, time_of_day, time_precision, source, method, evidence,"
            " confidence, review_status, review_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (report_id, patient_id, o["symptom"], int(o["present"]), o["value_num"], o["intensity"],
             o["count"], occurred.isoformat(), o["time_of_day"],
             "part_of_day" if o["time_of_day"] else "day", "free_text", o["method"], o["evidence"],
             o["confidence"], o["review_status"], o["review_reason"]))
    for r in rej:
        conn.execute("INSERT INTO rejected_extractions (report_id, payload, reason) VALUES (?,?,?)",
                     (report_id, json.dumps(r, default=str), r["review_reason"]))
    for m, method in meds:
        conn.execute("INSERT INTO medication_mentions (report_id, patient_id, med_code, action, evidence, method)"
                     " VALUES (?,?,?,?,?,?)", (report_id, patient_id, m.medication, m.action, m.evidence, method))
    return acc, rej, meds, run


def insert_questionnaire(conn, r):
    a, rid, pid, ts = r["answers"], r["report_id"], r["patient_id"], r["submitted_at"]
    conn.execute("INSERT INTO reports VALUES (?,?,?,?,NULL,?)",
                 (rid, pid, ts, "questionnaire", json.dumps(a)))
    day = ts[:10]

    def obs(symptom, present, score=None, count=None):
        conn.execute(
            "INSERT INTO symptom_observations (report_id, patient_id, symptom, present, score, intensity,"
            " episode_count, occurred_date, time_of_day, time_precision, source, method, evidence,"
            " confidence, review_status, review_reason) VALUES (?,?,?,?,?,NULL,?,?,NULL,'day',"
            "'questionnaire','questionnaire',NULL,1.0,'auto_accepted',NULL)",
            (rid, pid, symptom, int(present), score, count, day))

    for s in SCORED:
        obs(s, a[s] > 0, score=a[s])
    obs("vomiting", a["vomiting_24h"] > 0, count=a["vomiting_24h"])
    for b in ["dry_mouth", "taste_change", "thick_saliva", "fever"]:   # the demo's "Also today" list
        obs(b, b in a["also_today"])
    if a.get("weight_lb") is not None:
        conn.execute("INSERT INTO measurements (report_id, patient_id, measured_at, type, value) "
                     "VALUES (?,?,?,?,?)", (rid, pid, ts, "weight_lb", a["weight_lb"]))


# ----------------------------------------------------------------- rollup
def rebuild_daily(conn):
    """One row per patient/day/symptom for charting. Rules are deterministic and documented:
    - scored items: the questionnaire score wins; free text is shown alongside, never averaged in
    - a questionnaire score and the patient's words that differ by >= half the scale -> conflict
    - free text alone: coloured by the patient's intensity word, marked as 'from text'
    - vomiting: max of questionnaire count and summed message counts (windows overlap)
    - items still awaiting review are excluded until a person accepts them
    """
    conn.execute("DELETE FROM daily_symptoms")
    rows = conn.execute(
        "SELECT patient_id, occurred_date, symptom, present, score, intensity, episode_count, source, evidence "
        "FROM symptom_observations WHERE review_status IN ('auto_accepted','confirmed','corrected')").fetchall()
    groups = {}
    for r in rows:
        groups.setdefault((r["patient_id"], r["occurred_date"], r["symptom"]), []).append(r)
    out = []
    for (pid, d, sym), obs in groups.items():
        q = [o for o in obs if o["source"] == "questionnaire"]
        t = [o for o in obs if o["source"] == "free_text"]
        t_int = None
        t_frac = None
        if t:
            if any(o["present"] for o in t):
                ints = [o["intensity"] for o in t if o["present"] and o["intensity"]]
                t_int = max(ints, key=INTENSITY_ORDER.get) if ints else "unspecified"
                nums = [o["score"] for o in t if o["present"] and o["score"] is not None]
                if nums and sym in SCORED:
                    t_frac = scale_fraction(sym, max(nums))
                elif ints:
                    t_frac = INTENSITY_FRACTION[t_int]
            else:
                t_int, t_frac = "none", 0.0
        value, frac, source, conflict, note = None, None, None, 0, None
        if sym in SCORED:
            if q:
                value = max(o["score"] for o in q)
                frac = scale_fraction(sym, value)
                source = "both" if t else "questionnaire"
                if t_frac is not None and abs(t_frac - frac) >= 0.5:
                    conflict = 1
                    ev = "; ".join(f'"{o["evidence"]}"' for o in t if o["evidence"])
                    note = f"Questionnaire {SCORED[sym][0].lower()} = {value:g}, but message says {ev}"
            elif t:
                value = None
                frac = t_frac if t_frac is not None else 0.5
                source = "free_text"
        elif sym == "vomiting":
            qc = max((o["episode_count"] or 0) for o in q) if q else None
            tc = sum((o["episode_count"] if o["episode_count"] is not None else 1)
                     for o in t if o["present"]) if t else None
            vals = [v for v in (qc, tc) if v is not None]
            value = max(vals) if vals else 0
            frac = scale_fraction("vomiting", value)
            source = "both" if (q and t) else ("questionnaire" if q else "free_text")
            if qc is not None and tc is not None and qc != tc:
                note = f"Questionnaire {qc} episode(s) in 24 h; message reports {tc} - showing the larger"
        else:   # yes/no items and general_unwell
            value = 1.0 if any(o["present"] for o in obs) else 0.0
            frac = value
            source = "both" if (q and t) else ("questionnaire" if q else "free_text")
            if q and t and any(o["present"] for o in q) != any(o["present"] for o in t):
                note = "Questionnaire and message disagree"
        out.append((pid, d, sym, value, frac, source, t_int, conflict, note))
    conn.executemany("INSERT INTO daily_symptoms VALUES (?,?,?,?,?,?,?,?,?)", out)
    conn.commit()


# ----------------------------------------------------------------- build
def build(backend="rules", model=None, db_path: Path = DB_PATH, limit=None, verbose=True):
    if db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    conn.executescript((ROOT / "pipeline" / "schema.sql").read_text())
    for p in json.loads((RAW / "patients.json").read_text()):
        conn.execute("INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?,?)", tuple(p.values()))
    for code, info in MEDICATIONS.items():
        conn.execute("INSERT INTO medications VALUES (?,?)", (code, info["class"]))
    with open(RAW / "treatment_events.csv") as f:
        for r in csv.DictReader(f):
            conn.execute("INSERT INTO treatment_events (patient_id, event_date, type, detail) VALUES (?,?,?,?)",
                         (r["patient_id"], r["date"], r["type"], r["detail"]))
    with open(RAW / "medication_events.csv") as f:
        for r in csv.DictReader(f):
            conn.execute("INSERT INTO medication_events (patient_id, med_code, event_date, event_type, dose, note)"
                         " VALUES (?,?,?,?,?,?)", (r["patient_id"], r["medication"], r["date"],
                                                    r["event_type"], r["dose"], r["note"]))
    reports = [json.loads(l) for l in (RAW / "reports.jsonl").read_text().splitlines()]
    free = [r for r in reports if r["source"] == "free_text"][:limit]
    for r in reports:
        if r["source"] == "questionnaire":
            insert_questionnaire(conn, r)
    for i, r in enumerate(free, 1):
        insert_free_text(conn, r["report_id"], r["patient_id"], r["submitted_at"], r["text"], backend, model)
        if verbose and backend != "rules" and (i % 10 == 0 or i == len(free)):
            print(f"  parsed {i}/{len(free)} free-text reports", flush=True)
    conn.commit()
    rebuild_daily(conn)
    if verbose:
        c = lambda q: conn.execute(q).fetchone()[0]
        print(f"Built {db_path.name}: {c('SELECT COUNT(*) FROM reports')} reports, "
              f"{c('SELECT COUNT(*) FROM symptom_observations')} observations, "
              f"{c('SELECT COUNT(*) FROM review_queue')} awaiting review, "
              f"{c('SELECT COUNT(*) FROM rejected_extractions')} rejected, "
              f"{c('SELECT COUNT(*) FROM daily_symptoms WHERE conflict=1')} conflicting days")
    return conn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rules", "ollama"], default="rules")
    ap.add_argument("--model", default=None, help="Ollama model tag (default qwen2.5:7b)")
    args = ap.parse_args()
    if args.backend == "ollama" and not parse_llm.available(args.model or parse_llm.DEFAULT_MODEL):
        raise SystemExit(f"Ollama model not reachable at {parse_llm.OLLAMA_URL}. "
                         f"Run: ollama pull {args.model or parse_llm.DEFAULT_MODEL}")
    build(args.backend, args.model)
