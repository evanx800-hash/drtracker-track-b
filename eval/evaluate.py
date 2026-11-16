"""Score a parser against gold labels.

    python -m eval.evaluate                      # rules parser
    python -m eval.evaluate --backend ollama     # local LLM (+ rules cross-check, as in the pipeline)

Two test sets:
- generated: free-text reports rendered from the synthetic ground truth (data/raw/reports.jsonl)
- hard_dev:  30 hand-written edge cases (data/hard_cases.jsonl) - negation scope, vague counts,
             self-corrections, other people's symptoms, Spanish, hedging, two days in one message.
             The rule lexicon was tuned while looking at these, so rules scores here are optimistic.
- holdout:   20 more hand-written cases (data/holdout_cases.jsonl) written AFTER the rules and the
             prompt were frozen. This is the honest test.

Writes eval/results_<backend>.json and eval/errors_<backend>.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pipeline import parse_llm, parse_rules
from pipeline.validate import validate

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def load_sets():
    gen = [json.loads(l) for l in (ROOT / "data/raw/reports.jsonl").read_text().splitlines()]
    gen = [r for r in gen if r["source"] == "free_text"]
    load = lambda f: [json.loads(l) for l in (ROOT / "data" / f).read_text().splitlines() if l.strip()]
    return {"generated": gen, "hard_dev": load("hard_cases.jsonl"), "holdout": load("holdout_cases.jsonl")}


def predict(r, backend, model):
    rules_ext = parse_rules.parse(r["text"], r["submitted_at"])
    if backend == "rules":
        acc, rej = validate(rules_ext, r["text"], "rules")
        return acc, rej, [m.model_dump() for m in rules_ext.medications], None
    ext, audit = parse_llm.parse(r["text"], r["submitted_at"], model=model)
    if ext is None:
        return [], [], [], audit.get("error")
    acc, rej = validate(ext, r["text"], f"llm:{model}", cross_check=rules_ext)
    # the eval scores what the LLM produced; rules-only "missed?" flags are review aids, not predictions
    acc = [o for o in acc if o["method"] != "rules"]
    return acc, rej, [m.model_dump() for m in ext.medications], audit.get("latency_s")


def match(golds, preds):
    """Greedy one-to-one matching on symptom, preferring same day and polarity."""
    pairs, used = [], set()
    for g in golds:
        cands = [(i, p) for i, p in enumerate(preds) if i not in used and p["symptom"] == g["symptom"]]
        if not cands:
            pairs.append((g, None))
            continue
        cands.sort(key=lambda ip: (ip[1]["relative_day"] != g["relative_day"],
                                   ip[1]["present"] != g["present"],
                                   ip[1].get("count") != g.get("count")))
        i, p = cands[0]
        used.add(i)
        pairs.append((g, p))
    extra = [p for i, p in enumerate(preds) if i not in used]
    return pairs, extra


def score(reports, backend, model):
    c = dict(gold=0, pred=0, tp=0, pol_ok=0, int_n=0, int_ok=0, cnt_n=0, cnt_ok=0, vague_n=0, vague_ok=0,
             val_n=0, val_ok=0, day_n=0, day_ok=0, wrong=0, wrong_flagged=0, right=0, right_flagged=0,
             rejected=0, med_gold=0, med_pred=0, med_tp=0, failures=0)
    errors, latencies = [], []
    for r in reports:
        preds, rej, meds, lat = predict(r, backend, model)
        if isinstance(lat, (int, float)):
            latencies.append(lat)
        elif lat:
            c["failures"] += 1
        c["rejected"] += len(rej)
        golds = r["gold"]
        pairs, extra = match(golds, preds)
        c["gold"] += len(golds)
        c["pred"] += len(preds)
        for g, p in pairs:
            if p is None:
                errors.append((r["report_id"], r["text"], "missed", g["symptom"], json.dumps(g), ""))
                continue
            c["tp"] += 1
            ok = True
            if p["present"] == g["present"]:
                c["pol_ok"] += 1
            else:
                ok = False
                errors.append((r["report_id"], r["text"], "wrong present/absent", g["symptom"], json.dumps(g), json.dumps(p)))
            if g["present"] and g["intensity"]:
                c["int_n"] += 1
                if p["intensity"] == g["intensity"]:
                    c["int_ok"] += 1
                else:
                    errors.append((r["report_id"], r["text"], "intensity", g["symptom"], g["intensity"], str(p["intensity"])))
            if g["symptom"] == "vomiting" and g["present"]:
                if g["count"] is not None:
                    c["cnt_n"] += 1
                    if p["count"] == g["count"]:
                        c["cnt_ok"] += 1
                    else:
                        ok = False
                        errors.append((r["report_id"], r["text"], "count", "vomiting", str(g["count"]), str(p["count"])))
                else:
                    c["vague_n"] += 1
                    if p["count"] is None:
                        c["vague_ok"] += 1
                    else:
                        ok = False
                        errors.append((r["report_id"], r["text"], "invented a count", "vomiting", "null", str(p["count"])))
            if g["value_num"] is not None:
                c["val_n"] += 1
                if p["value_num"] is not None and abs(p["value_num"] - g["value_num"]) < 0.01:
                    c["val_ok"] += 1
                else:
                    errors.append((r["report_id"], r["text"], "number", g["symptom"], str(g["value_num"]), str(p["value_num"])))
            c["day_n"] += 1
            if p["relative_day"] == g["relative_day"]:
                c["day_ok"] += 1
            else:
                errors.append((r["report_id"], r["text"], "day", g["symptom"], str(g["relative_day"]), str(p["relative_day"])))
            flagged = p["review_status"] == "needs_review"
            if ok:
                c["right"] += 1
                c["right_flagged"] += flagged
            else:
                c["wrong"] += 1
                c["wrong_flagged"] += flagged
        for p in extra:
            c["wrong"] += 1
            c["wrong_flagged"] += p["review_status"] == "needs_review"
            errors.append((r["report_id"], r["text"], "extra (not in gold)", p["symptom"], "", json.dumps(
                {k: p[k] for k in ("present", "intensity", "count", "evidence")})))
        gm = {(m["medication"], m["action"]) for m in r.get("gold_meds", [])}
        pm = {(m["medication"], m["action"]) for m in meds}
        c["med_gold"] += len(gm)
        c["med_pred"] += len(pm)
        c["med_tp"] += len(gm & pm)

    div = lambda a, b: round(a / b, 3) if b else None
    prec, rec = div(c["tp"], c["pred"]), div(c["tp"], c["gold"])
    metrics = {
        "reports": len(reports),
        "gold_mentions": c["gold"], "predicted_mentions": c["pred"],
        "precision": prec, "recall": rec,
        "f1": round(2 * prec * rec / (prec + rec), 3) if prec and rec else None,
        "present_vs_absent_accuracy": div(c["pol_ok"], c["tp"]),
        "intensity_accuracy": div(c["int_ok"], c["int_n"]),
        "vomiting_count_exact": div(c["cnt_ok"], c["cnt_n"]),
        "vague_count_left_null": div(c["vague_ok"], c["vague_n"]),
        "patient_numbers_exact": div(c["val_ok"], c["val_n"]),
        "day_accuracy": div(c["day_ok"], c["day_n"]),
        "wrong_items_flagged_for_review": div(c["wrong_flagged"], c["wrong"]),
        "correct_items_flagged_for_review": div(c["right_flagged"], c["right"]),
        "rejected_by_grounding_check": c["rejected"],
        "medication_precision": div(c["med_tp"], c["med_pred"]),
        "medication_recall": div(c["med_tp"], c["med_gold"]),
        "llm_failures": c["failures"],
        "median_latency_s": sorted(latencies)[len(latencies) // 2] if latencies else None,
    }
    return metrics, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rules", "ollama"], default="rules")
    ap.add_argument("--model", default=parse_llm.DEFAULT_MODEL)
    args = ap.parse_args()
    if args.backend == "ollama" and not parse_llm.available(args.model):
        raise SystemExit(f"Ollama model {args.model} not reachable. Run: ollama pull {args.model}")
    name = "rules" if args.backend == "rules" else f"llm_{args.model.replace(':', '_')}"
    results, all_errors = {"backend": name}, []
    for set_name, reports in load_sets().items():
        m, errs = score(reports, args.backend, args.model)
        results[set_name] = m
        all_errors += [(set_name, *e) for e in errs]
    (OUT / f"results_{name}.json").write_text(json.dumps(results, indent=2))
    with open(OUT / f"errors_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "report_id", "text", "error", "symptom", "gold", "predicted"])
        w.writerows(all_errors)
    keys = [k for k in results["generated"]]
    print(f"\n{name}")
    print(f"{'metric':36s} {'generated':>10s} {'hard_dev':>10s} {'holdout':>10s}")
    for k in keys:
        print(f"{k:36s} " + " ".join(f"{str(results[s][k]):>10s}" for s in ("generated", "hard_dev", "holdout")))
    print(f"\nwrote eval/results_{name}.json and eval/errors_{name}.csv ({len(all_errors)} error rows)")


if __name__ == "__main__":
    main()
