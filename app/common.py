"""Shared data access, provider callouts and chart builders for the Streamlit app."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.build_db import DB_PATH, build, connect            # noqa: E402
from pipeline.vocab import (BOOLEAN, LABELS, MOUTH_SORES_LABELS, NAUSEA_LABELS,  # noqa: E402
                            SCORED, SWALLOWING_LABELS)

# Reference palette (dataviz skill): sequential blue ramp, neutral ink, status colours.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, MISSING = "#fcfcfb", "#e4e3df"
SERIES_1 = "#2a78d6"
CRITICAL, WARNING_C, SERIOUS = "#d03b3b", "#fab219", "#ec835a"

HEATMAP_ROWS = ["mouth_sores", "pain", "swallowing", "nausea", "vomiting",
                "dry_mouth", "taste_change", "thick_saliva", "fever", "fatigue"]
ROW_LABELS = {
    "mouth_sores": "Mouth sores (0–4)", "pain": "Pain (0–10)", "swallowing": "Swallowing",
    "nausea": "Nausea (0–4)", "vomiting": "Vomiting (episodes)", "dry_mouth": "Dry mouth",
    "taste_change": "Taste change", "thick_saliva": "Thick saliva", "fever": "Fever",
    "fatigue": "Fatigue (messages only)",
}
THRESHOLDS = {"mouth_sores": 1, "pain": 2, "swallowing": 1, "nausea": 2}   # "worse" = up by at least this


def ensure_db():
    if not DB_PATH.exists():
        build("rules", verbose=False)


def conn():
    ensure_db()
    return connect()


def q(sql, params=()):
    with conn() as c:
        return pd.read_sql_query(sql, c, params=params)


def patients():
    return q("SELECT * FROM patients ORDER BY patient_id")


def as_of_date(pid):
    d = q("SELECT MAX(date(submitted_at)) AS d FROM reports WHERE patient_id = ?", (pid,))["d"].iloc[0]
    return date.fromisoformat(d)


def fmt_day(d: date) -> str:
    return d.strftime("%b %-d") if sys.platform != "win32" else d.strftime("%b %#d")


def course_line(p, as_of):
    start = date.fromisoformat(p["treatment_start"])
    day_no = (as_of - start).days + 1
    sess = q("SELECT COUNT(*) n FROM treatment_events WHERE patient_id=? AND type='radiation' AND event_date<=?",
             (p["patient_id"], as_of.isoformat()))["n"].iloc[0]
    week = min(7, (day_no - 1) // 7 + 1)
    return f"Day {day_no} · week {week} of 7 · radiation session {sess} of 33 · weekly cisplatin", day_no


def describe_value(sym, v):
    if v is None or pd.isna(v):
        return ""
    if sym == "mouth_sores":
        return f"{v:g} of 4 ({MOUTH_SORES_LABELS.get(int(v), '')})"
    if sym == "pain":
        return f"{v:g} of 10"
    if sym == "swallowing":
        return SWALLOWING_LABELS.get(int(v), str(v))
    if sym == "nausea":
        return f"{v:g} of 4 ({NAUSEA_LABELS.get(int(v), '')})"
    if sym == "vomiting":
        return f"{int(v)} episode(s)"
    return "Yes" if v else "No"


# ------------------------------------------------------------------ callouts
def callouts(pid: str, last_visit: date):
    """Deterministic, rule-based 'since last visit' items. Returns [(level, label, text)].
    level: critical | warning | info. No model output is used here."""
    p = q("SELECT * FROM patients WHERE patient_id=?", (pid,)).iloc[0]
    as_of = as_of_date(pid)
    lv = last_visit.isoformat()
    window = max(1, (as_of - last_visit).days + 1)
    prev_start = (last_visit - timedelta(days=window)).isoformat()
    items = []

    # weight
    w = q("SELECT date(measured_at) d, value FROM measurements WHERE patient_id=? ORDER BY measured_at", (pid,))
    if len(w):
        latest = w.iloc[-1]
        drop = latest["value"] - p["baseline_weight_lb"]
        pct = 100 * drop / p["baseline_weight_lb"]
        at_visit = w[w["d"] <= lv]
        since = latest["value"] - at_visit.iloc[-1]["value"] if len(at_visit) else None
        txt = (f"{latest['value']:.1f} lb on {fmt_day(date.fromisoformat(latest['d']))}. "
               f"{drop:+.1f} lb ({pct:+.1f}%) since day 1"
               + (f", {since:+.1f} lb since last visit." if since is not None else "."))
        if p["alert_line_lb"] and latest["value"] < p["alert_line_lb"]:
            items.append(("critical", "Weight below alert line", f"{txt} Alert line is {p['alert_line_lb']:g} lb."))
        elif since is not None and since <= -0.03 * p["baseline_weight_lb"]:
            items.append(("warning", "Weight dropping", txt))
        else:
            items.append(("info", "Weight", txt))

    d = q("SELECT * FROM daily_symptoms WHERE patient_id=? AND day_date>=?", (pid, prev_start))
    since_df, before_df = d[d["day_date"] >= lv], d[d["day_date"] < lv]

    # fever (accepted or awaiting review - never hide it)
    fev = q("SELECT o.occurred_date, o.evidence, o.source, o.review_status FROM symptom_observations o "
            "WHERE o.patient_id=? AND o.symptom='fever' AND o.present=1 AND o.occurred_date>=? "
            "AND o.review_status!='rejected' ORDER BY o.occurred_date", (pid, lv))
    for _, r in fev.drop_duplicates("occurred_date").iterrows():
        src = f'message: "{r["evidence"]}"' if r["source"] == "free_text" else "check-in form"
        pending = " (awaiting review)" if r["review_status"] == "needs_review" else ""
        items.append(("critical", "Fever reported",
                      f"{fmt_day(date.fromisoformat(r['occurred_date']))}, from {src}{pending}."))

    # scored symptoms: worse / better
    for sym, thr in THRESHOLDS.items():
        a = since_df[(since_df["symptom"] == sym) & since_df["display_value"].notna()]["display_value"]
        b = before_df[(before_df["symptom"] == sym) & before_df["display_value"].notna()]["display_value"]
        if a.empty:
            continue
        worst_a = a.max()
        worst_b = b.max() if not b.empty else None
        name = SCORED[sym][0]
        if worst_b is not None and worst_a - worst_b >= thr:
            items.append(("warning", f"{name} worse",
                          f"Worst since last visit: {describe_value(sym, worst_a)}. Before: {describe_value(sym, worst_b)}."))
        elif worst_b is not None and worst_b - worst_a >= thr:
            items.append(("info", f"{name} better",
                          f"Worst since last visit: {describe_value(sym, worst_a)}. Before: {describe_value(sym, worst_b)}."))

    # vomiting
    va = since_df[since_df["symptom"] == "vomiting"]
    vb = before_df[before_df["symptom"] == "vomiting"]
    ea, eb = va["display_value"].sum(), vb["display_value"].sum()
    if ea > 0:
        level = "warning" if ea >= 3 or ea > eb else "info"
        items.append((level, "Vomiting", f"{int(ea)} episode(s) on {int((va['display_value'] > 0).sum())} day(s) "
                                         f"since last visit (previous {window} days: {int(eb)})."))

    # new yes/no symptoms
    new = []
    for sym in ["dry_mouth", "taste_change", "thick_saliva", "fatigue"]:
        if (since_df[(since_df["symptom"] == sym)]["display_value"] == 1).any() and \
                not (before_df[(before_df["symptom"] == sym)]["display_value"] == 1).any():
            new.append(LABELS[sym].lower())
    if new:
        items.append(("info", "New since last visit", ", ".join(new).capitalize() + "."))

    # conflicts
    for _, r in since_df[since_df["conflict"] == 1].iterrows():
        items.append(("warning", "Reports disagree",
                      f"{fmt_day(date.fromisoformat(r['day_date']))}: {r['conflict_note']}."))

    # medication changes
    meds = q("SELECT med_code, event_type, event_date, dose FROM medication_events "
             "WHERE patient_id=? AND event_date>=?", (pid, lv))
    for _, r in meds.iterrows():
        items.append(("info", "Prescription change",
                      f"{r['med_code'].replace('_', ' ')} {r['event_type']} {fmt_day(date.fromisoformat(r['event_date']))} ({r['dose']})."))

    # check-ins
    days = set(q("SELECT DISTINCT date(submitted_at) d FROM reports WHERE patient_id=? AND date(submitted_at)>=?",
                 (pid, lv))["d"])
    missed = [last_visit + timedelta(days=i) for i in range(window)
              if (last_visit + timedelta(days=i)).isoformat() not in days]
    if missed:
        items.append(("warning" if len(missed) >= 3 else "info", "Missed check-ins",
                      f"{len(missed)} of {window} days: " + ", ".join(fmt_day(m) for m in missed) +
                      ". Missing days are shown as gaps, not as 'no symptoms'."))
    rv = q("SELECT COUNT(*) n FROM review_queue WHERE patient_id=?", (pid,))["n"].iloc[0]
    if rv:
        items.append(("info", "Awaiting review", f"{rv} extracted item(s) from messages are not charted until reviewed."))

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(items, key=lambda x: order[x[0]])


# ------------------------------------------------------------------ main chart
def overview_figure(pid: str, last_visit: date | None = None, height=620):
    p = q("SELECT * FROM patients WHERE patient_id=?", (pid,)).iloc[0]
    start, as_of = date.fromisoformat(p["treatment_start"]), as_of_date(pid)
    days = [start + timedelta(days=i) for i in range((as_of - start).days + 1)]
    x = [d.isoformat() for d in days]
    daily = q("SELECT * FROM daily_symptoms WHERE patient_id=?", (pid,))
    report_days = set(q("SELECT DISTINCT date(submitted_at) d FROM reports WHERE patient_id=?", (pid,))["d"])
    texts = q("SELECT o.occurred_date, o.symptom, o.evidence FROM symptom_observations o WHERE o.patient_id=? "
              "AND o.source='free_text' AND o.review_status IN ('auto_accepted','confirmed','corrected')", (pid,))

    rows = [s for s in HEATMAP_ROWS]
    z, hover = [], []
    text_x, text_y, text_h, conf_x, conf_y, conf_h = [], [], [], [], [], []
    idx = {(r.day_date, r.symptom): r for r in daily.itertuples()}
    for sym in rows:
        zr, hr = [], []
        for d in days:
            k = (d.isoformat(), sym)
            day_label = f"{fmt_day(d)} · day {(d - start).days + 1}"
            if d.isoformat() not in report_days:
                zr.append(None)
                hr.append(f"{day_label}<br>No check-in")
                continue
            r = idx.get(k)
            if r is None:
                zr.append(None)
                hr.append(f"{day_label}<br>{LABELS[sym]}: not asked / not mentioned")
                continue
            # yes/no rows use a fixed mid tone so they don't out-shout graded severity
            zr.append(0.45 if (sym in BOOLEAN and r.display_frac) else r.display_frac)
            quotes = texts[(texts["occurred_date"] == d.isoformat()) & (texts["symptom"] == sym)]["evidence"].tolist()
            qtxt = "".join(f'<br>Message: "{t}"' for t in quotes[:2])
            if r.value_source == "free_text":
                val = f"from message only ({r.text_intensity or 'mentioned'})"
                text_x.append(d.isoformat()); text_y.append(ROW_LABELS[sym])
                text_h.append(f"{day_label}<br>{LABELS[sym]}: {val}{qtxt}")
            else:
                val = describe_value(sym, r.display_value)
            h = f"{day_label}<br>{LABELS[sym]}: {val}{qtxt}"
            if r.conflict:
                conf_x.append(d.isoformat()); conf_y.append(ROW_LABELS[sym])
                conf_h.append(f"{day_label}<br><b>Reports disagree</b><br>{r.conflict_note}")
            elif isinstance(r.conflict_note, str) and r.conflict_note:
                h += f"<br>{r.conflict_note}"
            hr.append(h)
        z.append(zr)
        hover.append(hr)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.66, 0.34])
    scale = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]
    fig.add_trace(go.Heatmap(
        x=x, y=[ROW_LABELS[s] for s in rows], z=z, customdata=hover,
        hovertemplate="%{customdata}<extra></extra>", colorscale=scale, zmin=0, zmax=1,
        xgap=2, ygap=2, showscale=True,
        colorbar=dict(tickvals=[0, 0.45, 1], ticktext=["none reported", "yes (yes/no rows)", "worst on<br>row's scale"],
                      len=0.55, y=0.72,
                      thickness=10, outlinewidth=0, tickfont=dict(size=10, color=INK_2))), row=1, col=1)
    if text_x:
        fig.add_trace(go.Scatter(x=text_x, y=text_y, mode="markers", name="Only in patient's message",
                                 marker=dict(symbol="circle", size=7, color="#ffffff", line=dict(color=INK, width=1.5)),
                                 customdata=text_h, hovertemplate="%{customdata}<extra></extra>"), row=1, col=1)
    fev = daily[(daily["symptom"] == "fever") & (daily["display_value"] == 1)]
    if len(fev):
        fig.add_trace(go.Scatter(x=fev["day_date"], y=[ROW_LABELS["fever"]] * len(fev), mode="markers",
                                 name="Fever reported", marker=dict(symbol="triangle-up", size=12, color=CRITICAL,
                                 line=dict(color="#ffffff", width=1.5)),
                                 hovertemplate="%{x|%b %-d}: fever reported<extra></extra>"), row=1, col=1)
    if conf_x:
        fig.add_trace(go.Scatter(x=conf_x, y=conf_y, mode="markers", name="Form and message disagree",
                                 marker=dict(symbol="diamond", size=11, color=CRITICAL, line=dict(color="#ffffff", width=1.5)),
                                 customdata=conf_h, hovertemplate="%{customdata}<extra></extra>"), row=1, col=1)

    # no-check-in columns
    missing = [d for d in days if d.isoformat() not in report_days]
    for d in missing:
        dt = datetime.combine(d, datetime.min.time())
        fig.add_shape(type="rect", xref="x", yref="y domain", x0=(dt - timedelta(hours=11)).isoformat(),
                      x1=(dt + timedelta(hours=11)).isoformat(), y0=0, y1=1, fillcolor=MISSING,
                      line_width=0, layer="below")
    if missing:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="No check-in that day",
                                 marker=dict(symbol="square", size=12, color=MISSING)), row=1, col=1)

    # weight
    w = q("SELECT measured_at, value FROM measurements WHERE patient_id=? ORDER BY measured_at", (pid,))
    w["d"] = w["measured_at"].str[:10]
    fig.add_trace(go.Scatter(x=w["d"], y=w["value"], mode="lines+markers", name="Weight (lb)",
                             line=dict(color=SERIES_1, width=2), marker=dict(size=8, color=SERIES_1,
                             line=dict(color=SURFACE, width=2)), connectgaps=False,
                             hovertemplate="%{x|%b %-d}: %{y:.1f} lb<extra></extra>"), row=2, col=1)
    if p["alert_line_lb"]:
        fig.add_hline(y=p["alert_line_lb"], line=dict(color=CRITICAL, width=1.5, dash="dash"), row=2, col=1,
                      annotation_text=f"Alert line {p['alert_line_lb']:g} lb", annotation_position="bottom left",
                      annotation_font=dict(color=CRITICAL, size=11))
    lo = min(w["value"].min(), p["alert_line_lb"] or 1e9) - 2
    hi = max(w["value"].max(), p["baseline_weight_lb"]) + 2
    fig.update_yaxes(range=[lo, hi], row=2, col=1, title=dict(text="Weight (lb)", font=dict(size=11, color=INK_2)))

    # treatment and prescription markers (neutral ink; text carries meaning, not colour)
    cis = q("SELECT event_date FROM treatment_events WHERE patient_id=? AND type='cisplatin'", (pid,))["event_date"]
    for c in cis:
        fig.add_vline(x=c, line=dict(color=INK_2, width=1, dash="dot"), row="all", col=1)
        fig.add_annotation(x=c, y=1.0, xref="x", yref="paper", text="Cis", showarrow=False, yanchor="bottom",
                           font=dict(size=10, color=INK_2))
    meds = q("SELECT event_date, med_code FROM medication_events WHERE patient_id=? AND event_date>?",
             (pid, p["treatment_start"]))
    for _, m in meds.iterrows():
        fig.add_vline(x=m["event_date"], line=dict(color=INK, width=1.5), row="all", col=1)
        fig.add_annotation(x=m["event_date"], y=1.06, xref="x", yref="paper", showarrow=False, yanchor="bottom",
                           text=f"+ {m['med_code'].replace('_', ' ')}", font=dict(size=11, color=INK))
    if last_visit and last_visit >= start:
        lv_dt = (datetime.combine(last_visit, datetime.min.time()) - timedelta(hours=12)).isoformat()
        fig.add_vline(x=lv_dt, line=dict(color=INK, width=2, dash="dash"), row="all", col=1)
        fig.add_annotation(x=lv_dt, y=1.0, xref="x", yref="paper", showarrow=False, xanchor="left",
                           text=" ◀ since last visit ▶", font=dict(size=11, color=INK), yanchor="bottom", yshift=14)

    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=60), plot_bgcolor=SURFACE,
                      paper_bgcolor=SURFACE, font=dict(family="Inter, Helvetica, Arial", color=INK, size=12),
                      legend=dict(orientation="h", y=-0.14, x=0, font=dict(size=11, color=INK_2)),
                      hoverlabel=dict(bgcolor="#ffffff", font_size=12, bordercolor=MUTED))
    fig.update_yaxes(autorange="reversed", row=1, col=1, tickfont=dict(size=12, color=INK))
    fig.update_xaxes(showgrid=False, tickformat="%b %-d", dtick=7 * 24 * 3600 * 1000,
                     tick0=start.isoformat(), tickfont=dict(color=INK_2), range=[
                         (datetime.combine(start, datetime.min.time()) - timedelta(hours=12)).isoformat(),
                         (datetime.combine(as_of, datetime.min.time()) + timedelta(hours=12)).isoformat()])
    fig.update_yaxes(showgrid=True, gridcolor="#ecebe7", row=2, col=1, tickfont=dict(color=INK_2))
    return fig


def recent_messages(pid: str, since: date):
    r = q("SELECT report_id, submitted_at, raw_text FROM reports WHERE patient_id=? AND source='free_text' "
          "AND date(submitted_at)>=? ORDER BY submitted_at DESC", (pid, since.isoformat()))
    o = q("SELECT report_id, symptom, present, intensity, score, episode_count, review_status "
          "FROM symptom_observations WHERE patient_id=? AND source='free_text'", (pid,))
    rows = []
    for _, rep in r.iterrows():
        obs = o[o["report_id"] == rep["report_id"]]
        chips = []
        for _, x in obs.iterrows():
            s = LABELS[x["symptom"]].lower()
            if not x["present"]:
                s = "no " + s
            elif x["episode_count"] is not None and not pd.isna(x["episode_count"]):
                s += f" ×{int(x['episode_count'])}"
            elif x["score"] is not None and not pd.isna(x["score"]):
                s += f" {x['score']:g}"
            elif x["intensity"]:
                s += f" ({x['intensity']})"
            if x["review_status"] == "needs_review":
                s += " ⚑"
            chips.append(s)
        rows.append({"When": datetime.fromisoformat(rep["submitted_at"]).strftime("%a %b %-d, %-I:%M %p"),
                     "Patient wrote": rep["raw_text"], "Captured as": " · ".join(chips) or "—"})
    return pd.DataFrame(rows)


def eval_results():
    out = {}
    for f in sorted((ROOT / "eval").glob("results_*.json")):
        out[f.stem.replace("results_", "")] = json.loads(f.read_text())
    return out
