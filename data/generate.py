"""Synthetic dataset generator.

Design: generate the TRUTH first, then render patient-voice free text FROM it.
Every free-text report carries gold labels describing exactly what the text says,
so the parser can be scored instead of eyeballed.

Scenario: head & neck (oral cavity) radiation, 33 sessions over ~7 weeks with
weekly cisplatin - modeled on the Dr.Tracker oncology demo. All patients and
values are fictional.

Run:  python -m data.generate
"""
from __future__ import annotations

import csv
import json
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent / "raw"
AS_OF = date(2026, 11, 13)          # "today" for the whole dataset
SEED = 42

# ------------------------------------------------------------------ patients
# day_count = which treatment day "today" is for this patient.
PATIENTS = [
    dict(patient_id="P01", name="James C.", age=58, day_count=25, baseline_weight=172.0,
         target_loss=8.4, alert_line_lb=165.0, sores_sens=0.85, nausea_sens=0.7,
         adherence=0.97, free_text_rate=0.6, style="plain",
         note="Modeled on the demo patient: below the 165 lb alert line by day 25."),
    dict(patient_id="P02", name="Maria R.", age=64, day_count=49, baseline_weight=148.0,
         target_loss=13.0, alert_line_lb=140.0, sores_sens=1.0, nausea_sens=0.6,
         adherence=0.9, free_text_rate=0.6, style="plain",
         note="Completed course; severe mouth sores; oxycodone started week 4."),
    dict(patient_id="P03", name="Robert K.", age=61, day_count=42, baseline_weight=196.0,
         target_loss=9.0, alert_line_lb=186.0, sores_sens=0.7, nausea_sens=1.25,
         adherence=0.92, free_text_rate=0.7, style="plain",
         note="Nausea-heavy; olanzapine added on day 16 alongside ondansetron."),
    dict(patient_id="P04", name="Linda M.", age=55, day_count=35, baseline_weight=135.0,
         target_loss=3.5, alert_line_lb=128.0, sores_sens=0.5, nausea_sens=0.4,
         adherence=0.98, free_text_rate=0.45, style="plain",
         note="Mild course, very consistent check-ins."),
    dict(patient_id="P05", name="David O.", age=67, day_count=38, baseline_weight=181.0,
         target_loss=7.0, alert_line_lb=172.0, sores_sens=0.8, nausea_sens=0.85,
         adherence=0.72, free_text_rate=0.85, style="vague",
         note="Irregular check-ins, vague free text, a 3-day gap, one fever day."),
]

MED_EVENTS = [  # (patient, day, med, type, dose, note) - prescriber-entered
    ("P01", 1, "ondansetron", "start", "8 mg PRN", "with cisplatin"),
    ("P02", 1, "ondansetron", "start", "8 mg PRN", "with cisplatin"),
    ("P02", 10, "lidocaine_rinse", "start", "15 mL before meals", ""),
    ("P02", 24, "oxycodone", "start", "5 mg q6h PRN", "pain >= 6 two days running"),
    ("P03", 1, "ondansetron", "start", "8 mg PRN", "with cisplatin"),
    ("P03", 16, "olanzapine", "start", "5 mg nightly", "added for breakthrough nausea"),
    ("P04", 1, "ondansetron", "start", "8 mg PRN", "with cisplatin"),
    ("P05", 1, "ondansetron", "start", "8 mg PRN", "with cisplatin"),
    ("P05", 20, "lidocaine_rinse", "start", "15 mL before meals", ""),
]
OLANZAPINE_START = {"P03": 16}
OXY_START = {"P02": 24}
DAVID_GAP = (27, 29)     # missed days for P05
DAVID_FEVER_DAY = 31


def clip(x, lo, hi):
    return max(lo, min(hi, x))


# ------------------------------------------------------------------ truth model
def build_course(p, rng):
    start = AS_OF - timedelta(days=p["day_count"] - 1)
    days = [start + timedelta(days=i) for i in range(p["day_count"])]
    # radiation on weekdays, up to 33 sessions
    sessions, n = {}, 0
    for d in days:
        if d.weekday() < 5 and n < 33:
            n += 1
            sessions[d] = n
    # weekly cisplatin on treatment days 1, 8, 15, ...
    cis_days = [days[i] for i in range(0, len(days), 7)][:7]
    return start, days, sessions, cis_days


def latent_states(p, rng, days, sessions, cis_days):
    rows = []
    sess_so_far = 0
    weight = p["baseline_weight"]
    cum_difficulty = []
    for i, d in enumerate(days):
        day_no = i + 1
        sess_so_far = sessions.get(d, sess_so_far) if d in sessions else sess_so_far
        frac = sess_so_far / 33
        # mouth sores grow with cumulative radiation
        sores_cont = p["sores_sens"] * 4 * clip((day_no - 6) / 28, 0, 1) + rng.gauss(0, 0.35)
        sores = int(round(clip(sores_cont, 0, 4)))
        # pain follows sores; opioid lowers it
        pain_c = 1 + 1.6 * sores + rng.gauss(0, 0.8)
        if p["patient_id"] in OXY_START and day_no >= OXY_START[p["patient_id"]]:
            pain_c -= 2.0
        pain = int(round(clip(pain_c, 0, 10)))
        swallow = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}[sores]
        if rng.random() < 0.12:
            swallow = int(clip(swallow + rng.choice([-1, 1]), 0, 3))
        # nausea: spike 1-3 days after each cisplatin dose
        since = min([(d - c).days for c in cis_days if c <= d], default=99)
        curve = [1.0, 3.0, 3.0, 2.0, 1.0, 0.5, 0.3]
        base = curve[since] if since < len(curve) else 0.2
        n_c = base * p["nausea_sens"] + rng.gauss(0, 0.4)
        if p["patient_id"] in OLANZAPINE_START and day_no >= OLANZAPINE_START[p["patient_id"]]:
            n_c *= 0.45
        nausea = int(round(clip(n_c, 0, 4)))
        if nausea >= 4:
            vomit = rng.choice([2, 3, 3, 4])
        elif nausea == 3:
            vomit = rng.choice([0, 1, 1, 2])
        elif nausea == 2:
            vomit = 1 if rng.random() < 0.2 else 0
        else:
            vomit = 0
        dry = rng.random() < 0.1 + 0.8 * frac
        taste = rng.random() < 0.05 + 0.85 * frac
        thick = rng.random() < 0.6 * frac
        fever = p["patient_id"] == "P05" and day_no == DAVID_FEVER_DAY
        fatigue = rng.random() < 0.3 + 0.45 * frac
        cum_difficulty.append(0.4 + swallow + 0.6 * nausea)
        rows.append(dict(patient_id=p["patient_id"], date=d.isoformat(), day=day_no,
                         session=sess_so_far, days_since_cisplatin=since,
                         mouth_sores=sores, pain=pain, swallowing=swallow, nausea=nausea,
                         vomiting=vomit, dry_mouth=dry, taste_change=taste, thick_saliva=thick,
                         fever=fever, fatigue=fatigue))
    # weight: loss proportional to cumulative eating difficulty, scaled to target
    total = sum(cum_difficulty)
    acc = 0
    for r, c in zip(rows, cum_difficulty):
        acc += c
        r["weight_true"] = round(p["baseline_weight"] - p["target_loss"] * acc / total
                                 + rng.gauss(0, 0.35), 1)
    rows[0]["weight_true"] = p["baseline_weight"]
    if p["patient_id"] == "P01":
        rows[-1]["weight_true"] = 163.6       # matches the demo console
    return rows


# ------------------------------------------------------------------ free text
NUM_WORDS = {1: "once", 2: "twice", 3: "three times", 4: "four times"}


def time_phrase(rel, tod, rng):
    if rel == -1 and tod == "night":
        return rng.choice(["last night", "overnight", "in the night"])
    if rel == -1:
        return "yesterday"
    if tod == "morning":
        return rng.choice(["this morning", "this am", "when i woke up"])
    if tod == "afternoon":
        return rng.choice(["after lunch", "this afternoon"])
    if tod == "evening":
        return rng.choice(["after dinner", "tonight", "this evening"])
    return ""


PHRASES = {
    ("nausea", "none"): ["no nausea today", "stomach's been fine", "didn't feel sick at all",
                         "nausea is gone"],
    ("nausea", "mild"): ["a little queasy", "slightly nauseous", "bit of nausea",
                         "stomach a little unsettled"],
    ("nausea", "moderate"): ["pretty nauseous", "felt sick to my stomach most of the day",
                             "nauseous on and off", "queasy all day"],
    ("nausea", "severe"): ["nausea was horrible", "so nauseous i couldn't eat",
                           "worst nausea so far", "really bad nausea"],
    ("vomiting", "none"): ["no vomiting", "didn't throw up", "kept everything down"],
    ("mouth_sores", "none"): ["no mouth sores yet"],
    ("mouth_sores", "mild"): ["mouth is a bit sore", "some redness in my mouth",
                              "a couple small sores starting"],
    ("mouth_sores", "moderate"): ["sores in my mouth are pretty bad",
                                  "mouth sores getting worse", "the sores are pretty bad"],
    ("mouth_sores", "severe"): ["mouth sores are killing me", "sores so bad i can barely drink",
                                "mouth is raw, sores everywhere"],
    ("pain", "none"): ["no pain today"],
    ("pain", "mild"): ["a little achy", "mild pain, nothing major"],
    ("pain", "moderate"): ["throat really hurts", "pain is pretty constant"],
    ("pain", "severe"): ["pain was unbearable", "worst pain yet", "pain is really bad"],
    ("swallowing", "none"): ["swallowing is fine"],
    ("swallowing", "mild"): ["swallowing is getting harder", "sticking to soft foods"],
    ("swallowing", "moderate"): ["can only get liquids down", "only managing shakes and liquids"],
    ("swallowing", "severe"): ["can't swallow anything", "couldn't even get water down"],
    ("dry_mouth", "present"): ["mouth so dry", "dry mouth again", "constantly sipping water, mouth is dry"],
    ("taste_change", "present"): ["everything tastes like metal", "food tastes off",
                                  "can't taste anything"],
    ("thick_saliva", "present"): ["spit is super thick", "thick mucus in my throat"],
    ("fatigue", "present"): ["so tired", "wiped out after radiation", "exhausted today"],
    ("general_unwell", "present"): ["feel kinda off", "not great today", "rough day",
                                    "just feeling blah"],
}
FILLERS = ["radiation went ok", "long day", "wife made soup", "appointment ran late",
           "feeling ok overall", "trying to drink more water", "slept badly",
           "did the salt rinse", "went for a short walk"]


def intensity_from_score(sym, v):
    if sym == "pain":
        return "none" if v == 0 else "mild" if v <= 3 else "moderate" if v <= 6 else "severe"
    if sym == "swallowing":
        return ["none", "mild", "moderate", "severe"][v]
    return ["none", "mild", "moderate", "severe", "severe"][v]


def render_mention(m, rng):
    s, inten = m["symptom"], m["intensity"]
    t = time_phrase(m["relative_day"], m["time_of_day"], rng)
    if s == "vomiting" and m["present"]:
        c = m["count"]
        if c is None:
            core = rng.choice(["threw up a few times", "was throwing up on and off",
                               "puked a couple of times"])
        else:
            core = rng.choice([f"threw up {NUM_WORDS[c]}", f"vomited {c} times" if c > 1 else "vomited once",
                               f"puked {NUM_WORDS[c]}", f"got sick {NUM_WORDS[c]}"])
        return f"{core} {t}".strip()
    if s == "pain" and m["value_num"] is not None:
        n = int(m["value_num"])
        core = rng.choice([f"pain is a {n}/10", f"pain around {n}", f"pain at like {n} out of 10"])
        return f"{core} {t}".strip()
    if s == "fever" and m["present"]:
        temp = m["value_num"]
        core = f"had a fever of {temp}" if temp else "felt feverish with chills"
        return f"{core} {t}".strip()
    key = (s, inten) if (s, inten) in PHRASES else (s, "present")
    core = rng.choice(PHRASES[key])
    if t and (m["relative_day"] != 0 or rng.random() < 0.6):
        return f"{core} {t}"
    m["time_of_day"] = None          # keep gold in sync with what the text actually says
    return core


def typo(word, rng):
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:] if rng.random() < 0.6 else word[:i] + word[i] + word[i:]


def messify(text, style, rng):
    words = text.split(" ")
    p_typo = 0.05 if style == "vague" else 0.03
    words = [typo(w, rng) if rng.random() < p_typo else w for w in words]
    text = " ".join(words)
    if rng.random() < 0.5:
        text = text.lower()
    if rng.random() < 0.4:
        text = text.replace("'", "")
    return text


def compose_free_text(p, st, prev, rng, evening, q_answers):
    """Pick what the patient talks about, render it, and return (text, gold, meta)."""
    pid = p["patient_id"]
    mentions = []

    def add(sym, present, intensity=None, count=None, value_num=None, rel=0, tod=None):
        mentions.append(dict(symptom=sym, present=present, intensity=intensity, count=count,
                             value_num=value_num, relative_day=rel, time_of_day=tod))

    # vomiting
    if st["vomiting"] > 0 and rng.random() < 0.9:
        if evening:
            rel, tod = 0, rng.choice(["afternoon", "evening", None])
        else:
            rel, tod = rng.choice([(-1, "night"), (0, "morning"), (0, None)])
        cnt = None if (p["style"] == "vague" and rng.random() < 0.5) or rng.random() < 0.12 else st["vomiting"]
        add("vomiting", True, None, cnt, None, rel, tod)
    elif st["vomiting"] == 0 and prev and prev["vomiting"] > 0 and rng.random() < 0.5:
        add("vomiting", False, "none")
    # nausea
    nz = st["nausea"]
    if (nz >= 2 and rng.random() < 0.7) or (nz == 1 and rng.random() < 0.3):
        add("nausea", True, intensity_from_score("nausea", nz),
            tod="morning" if (not evening and rng.random() < 0.3) else None)
    elif nz == 0 and prev and prev["nausea"] >= 2 and rng.random() < 0.35:
        add("nausea", False, "none")
    # mouth sores / pain / swallowing
    so = st["mouth_sores"]
    if rng.random() < {0: 0.03, 1: 0.2, 2: 0.35, 3: 0.6, 4: 0.7}[so]:
        add("mouth_sores", so > 0, intensity_from_score("mouth_sores", so))
    pa = st["pain"]
    if rng.random() < (0.5 if pa >= 6 else 0.18):
        if p["style"] != "vague" and rng.random() < 0.4 and pa > 0:
            add("pain", True, None, None, pa)
        else:
            add("pain", pa > 0, intensity_from_score("pain", pa),
                rel=-1 if rng.random() < 0.15 else 0)
            if mentions[-1]["relative_day"] == -1:
                mentions[-1]["time_of_day"] = "night"
    sw = st["swallowing"]
    if rng.random() < {0: 0.03, 1: 0.25, 2: 0.5, 3: 0.6}[sw]:
        add("swallowing", sw > 0, intensity_from_score("swallowing", sw))
    for b in ["dry_mouth", "taste_change", "thick_saliva", "fatigue"]:
        if st[b] and rng.random() < (0.25 if b == "fatigue" else 0.13):
            add(b, True, None)
    if st["fever"]:
        add("fever", True, None, None, 100.9, rel=-1, tod="night")
    if rng.random() < (0.25 if p["style"] == "vague" else 0.06):
        add("general_unwell", True, None)

    # deliberate conflict with the same-morning questionnaire
    conflict = None
    if q_answers is not None and not evening and rng.random() < 0.12:
        sym = rng.choice(["nausea", "mouth_sores"])
        qv = q_answers[sym]
        text_int = "severe" if qv <= 1 else "none"
        mentions = [m for m in mentions if m["symptom"] != sym]
        add(sym, text_int != "none", text_int)
        conflict = dict(symptom=sym, questionnaire=qv, text_intensity=text_int)

    rng.shuffle(mentions)
    mentions = sorted(mentions, key=lambda m: m["symptom"] != "fever")[:4]   # never drop a fever

    # medication mentions (patient-reported adherence/effect)
    meds = []
    day = st["day"]
    if nz >= 2 and rng.random() < 0.3:
        meds.append(dict(medication="ondansetron", action="took", phrase=rng.choice(
            ["took a zofran", "took my zofran", "zofrin helped a bit"])))
    if pid in OLANZAPINE_START and day >= OLANZAPINE_START[pid] and rng.random() < 0.3:
        meds.append(dict(medication="olanzapine", action="helped",
                         phrase="the new nausea pill seems to be helping"))
    if pid in OXY_START and day >= OXY_START[pid] and rng.random() < 0.3:
        meds.append(dict(medication="oxycodone", action="helped", phrase="the oxy takes the edge off"))
    if pid in ("P02", "P05") and so >= 2 and rng.random() < 0.15:
        meds.append(dict(medication="lidocaine_rinse", action="took",
                         phrase="used the magic mouthwash before eating"))

    parts = [render_mention(m, rng) for m in mentions] + [m["phrase"] for m in meds]
    if not parts or rng.random() < 0.35:
        parts.append(rng.choice(FILLERS))
    rng.shuffle(parts)
    joiner = rng.choice([". ", ", ", ". ", " and "]) if len(parts) > 1 else ""
    text = joiner.join(parts)
    text = text[0].upper() + text[1:]
    text = messify(text, p["style"], rng)
    gold_meds = [dict(medication=m["medication"], action=m["action"]) for m in meds]
    return text, mentions, gold_meds, conflict


# ------------------------------------------------------------------ main
def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    patients_out, treat_rows, med_rows, reports, latent_all = [], [], [], [], []

    for p in PATIENTS:
        start, days, sessions, cis_days = build_course(p, rng)
        states = latent_states(p, rng, days, sessions, cis_days)
        latent_all += states
        last_visit = AS_OF - timedelta(days=7) if p["day_count"] > 7 else start
        patients_out.append(dict(
            patient_id=p["patient_id"], display_name=p["name"], age=p["age"],
            diagnosis="Oral cavity cancer", regimen="Radiation (33 sessions) + weekly cisplatin",
            treatment_start=start.isoformat(), baseline_weight_lb=p["baseline_weight"],
            alert_line_lb=p["alert_line_lb"], last_visit_date=last_visit.isoformat(),
            synthetic_note=p["note"]))
        for d, n in sessions.items():
            treat_rows.append(dict(patient_id=p["patient_id"], date=d.isoformat(),
                                   type="radiation", detail=f"session {n} of 33"))
        for k, c in enumerate(cis_days, 1):
            treat_rows.append(dict(patient_id=p["patient_id"], date=c.isoformat(),
                                   type="cisplatin", detail=f"weekly dose {k}"))
        for (pid, day, med, typ, dose, note) in MED_EVENTS:
            if pid == p["patient_id"] and day <= p["day_count"]:
                med_rows.append(dict(patient_id=pid, date=(start + timedelta(days=day - 1)).isoformat(),
                                     medication=med, event_type=typ, dose=dose, note=note))

        prev = None
        for st in states:
            d = date.fromisoformat(st["date"])
            pid = p["patient_id"]
            gap = pid == "P05" and DAVID_GAP[0] <= st["day"] <= DAVID_GAP[1]
            checks_in = (not gap) and (rng.random() < p["adherence"] or st["day"] == p["day_count"])
            if not checks_in:
                prev = st
                continue
            text_only = rng.random() < (0.3 if p["style"] == "vague" else 0.05)
            morning = datetime.combine(d, datetime.min.time()) + timedelta(
                hours=7, minutes=rng.randint(0, 150))
            if pid == "P01" and st["day"] == p["day_count"]:
                morning = datetime.combine(d, datetime.min.time()) + timedelta(hours=8, minutes=5)
                text_only = False
            q = None
            if not text_only:
                weigh = rng.random() < 0.85 or st["day"] in (1, p["day_count"])
                q = dict(mouth_sores=st["mouth_sores"], pain=st["pain"], swallowing=st["swallowing"],
                         nausea=st["nausea"], vomiting_24h=st["vomiting"],
                         also_today=[b for b in ["dry_mouth", "taste_change", "thick_saliva", "fever"] if st[b]],
                         weight_lb=st["weight_true"] if weigh else None)
                reports.append(dict(report_id=f"{pid}-{st['day']:02d}-Q", patient_id=pid,
                                    submitted_at=morning.isoformat(timespec="minutes"),
                                    source="questionnaire", answers=q))
            if text_only or rng.random() < p["free_text_rate"]:
                evening = (not text_only) and rng.random() < 0.35
                ts = (datetime.combine(d, datetime.min.time()) + timedelta(hours=19, minutes=rng.randint(0, 180))
                      if evening else morning + timedelta(minutes=1))
                text, gold, gold_meds, conflict = compose_free_text(p, st, prev, rng, evening, q)
                reports.append(dict(report_id=f"{pid}-{st['day']:02d}-T{'E' if evening else 'M'}",
                                    patient_id=pid, submitted_at=ts.isoformat(timespec="minutes"),
                                    source="free_text", text=text, gold=gold, gold_meds=gold_meds,
                                    injected_conflict=conflict))
            prev = st

    (OUT / "patients.json").write_text(json.dumps(patients_out, indent=2))
    for name, rows in [("treatment_events.csv", treat_rows), ("medication_events.csv", med_rows),
                       ("latent_truth.csv", latent_all)]:
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(OUT / "reports.jsonl", "w") as f:
        for r in sorted(reports, key=lambda r: (r["patient_id"], r["submitted_at"])):
            f.write(json.dumps(r) + "\n")
    n_ft = sum(r["source"] == "free_text" for r in reports)
    print(f"{len(patients_out)} patients, {len(reports)} reports "
          f"({n_ft} free text, {len(reports) - n_ft} questionnaires) -> {OUT}")


if __name__ == "__main__":
    main()
