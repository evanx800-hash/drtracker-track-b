import html
import json
import re
from datetime import datetime, time

import pandas as pd
import streamlit as st

from common import as_of_date, conn
from pipeline import parse_llm
from pipeline.build_db import insert_free_text, rebuild_daily, run_parser
from pipeline.vocab import LABELS
from sidebar import patient_picker

pid, p = patient_picker()

st.markdown("## Parse a patient message")
st.caption("Type what a patient might send. The parser turns it into structured rows; validation decides what "
           "is charted automatically and what waits for a person.")

EXAMPLES = [
    "didnt puke today but super queasy after lunch, mouth still killing me",
    "threw up twice last night and once this morning. took my zofran",
    "feel kinda off, not sure if the new nausea pill is doing anything",
    "temp 100.9 overnight, chills. pain like 6/10",
    "my husband threw up after dinner, I'm ok",
]
ex = st.selectbox("Start from an example (or type your own below)", EXAMPLES)
text = st.text_area("Patient message", value=ex, height=90)

c1, c2, c3 = st.columns([1, 1, 2])
d = c1.date_input("Sent on", value=as_of_date(pid))
t = c2.time_input("at", value=time(20, 15))
llm_ok = parse_llm.available()
options = ["Ollama (local LLM) + rules cross-check", "Rules only"]
backend_label = c3.radio("Parser", options, index=0 if llm_ok else 1, horizontal=True)
backend = "ollama" if backend_label.startswith("Ollama") else "rules"
if backend == "ollama" and not llm_ok:
    st.warning(f"Ollama model `{parse_llm.DEFAULT_MODEL}` isn't reachable at {parse_llm.OLLAMA_URL}. "
               f"Start Ollama and run `ollama pull {parse_llm.DEFAULT_MODEL}`, or use Rules only.")

submitted_at = datetime.combine(d, t).isoformat(timespec="minutes")
if st.button("Extract", type="primary", disabled=(backend == "ollama" and not llm_ok)):
    with st.spinner("Extracting…"):
        acc, rej, meds, run = run_parser(text, submitted_at, backend)
    st.session_state.parsed = dict(text=text, submitted_at=submitted_at, backend=backend,
                                   acc=acc, rej=rej, meds=[(m.model_dump(), meth) for m, meth in meds], run=run)

res = st.session_state.get("parsed")
if res and res["text"] == text:
    # highlight evidence spans in the original text
    marked = html.escape(res["text"])
    for o in sorted(res["acc"], key=lambda o: -len(o["evidence"] or "")):
        ev = html.escape(o["evidence"] or "")
        if ev and ev in marked and f">{ev}<" not in marked:
            marked = marked.replace(ev, f"<mark title='{o['symptom']}'>{ev}</mark>", 1)
    st.markdown(f"<div style='font-size:1.15rem;padding:.6rem .8rem;border:1px solid #e4e3df;border-radius:6px'>"
                f"{marked}</div>", unsafe_allow_html=True)
    run = res["run"]
    st.caption(f"Parser: {run['method']} · status {run['status']}"
               + (f" · {run['latency_s']} s" if run.get("latency_s") else "")
               + (f" · error: {run['error']}" if run.get("error") else ""))

    rows = []
    for o in res["acc"]:
        rows.append({
            "Status": "✅ auto-accepted" if o["review_status"] == "auto_accepted" else "⚑ needs review",
            "Symptom": LABELS[o["symptom"]], "Present": "yes" if o["present"] else "NO (stated absent)",
            "Intensity (patient's words)": o["intensity"] or "—",
            "Episodes": "—" if o["count"] is None else o["count"],
            "Number given": "—" if o["value_num"] is None else o["value_num"],
            "Day": {0: "same day", -1: "day before", -2: "2 days before"}.get(o["relative_day"], o["relative_day"]),
            "Time": o["time_of_day"] or "—", "Evidence": o["evidence"],
            "Why flagged": o["review_reason"] or "", "Found by": o["method"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("Nothing extracted. (Correct for messages with no symptoms of the patient's own.)")
    if res["meds"]:
        st.markdown("**Medication mentions** (patient-reported, separate from prescriptions)")
        st.dataframe(pd.DataFrame([{"Medication": m["medication"], "Action": m["action"],
                                    "Evidence": m["evidence"], "Found by": meth} for m, meth in res["meds"]]),
                     hide_index=True, width="stretch")
    if res["rej"]:
        st.error("Rejected by the grounding check (evidence not found in the message):")
        st.json(res["rej"])
    with st.expander("Raw parser output (stored in extraction_runs for audit)"):
        raw = run.get("raw_output")
        try:
            st.json(json.loads(raw) if raw else {})
        except Exception:
            st.code(raw or "")

    if st.button(f"Save to {p['display_name']}'s record"):
        rid = f"LIVE-{pid}-{datetime.now().strftime('%H%M%S')}"
        with conn() as c:
            insert_free_text(c, rid, pid, res["submitted_at"], res["text"], res["backend"])
            c.commit()
            rebuild_daily(c)
        st.success(f"Saved as {rid}. Open Patient overview to see it on the chart"
                   + (" (flagged items appear after review)." if any(o['review_status'] == 'needs_review' for o in res['acc']) else "."))

with st.sidebar:
    st.divider()
    if st.button("Remove live-entered messages"):
        with conn() as c:
            ids = [r[0] for r in c.execute("SELECT report_id FROM reports WHERE report_id LIKE 'LIVE-%'")]
            for t_ in ["symptom_observations", "medication_mentions", "extraction_runs", "rejected_extractions"]:
                c.executemany(f"DELETE FROM {t_} WHERE report_id=?", [(i,) for i in ids])
            c.executemany("DELETE FROM reports WHERE report_id=?", [(i,) for i in ids])
            c.commit()
            rebuild_daily(c)
        st.success(f"Removed {len(ids)} live message(s).")
