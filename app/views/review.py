import pandas as pd
import streamlit as st

from common import conn, q
from pipeline.build_db import rebuild_daily
from pipeline.vocab import LABELS

st.markdown("## Review queue")
st.caption("Extractions that validation would not auto-accept. Nothing here is charted until a person confirms "
           "it. Rejecting keeps the row (status = rejected) so the audit trail survives.")

items = q("SELECT * FROM review_queue ORDER BY submitted_at DESC")
if items.empty:
    st.success("Nothing awaiting review.")
for r in items.itertuples():
    with st.container(border=True):
        a, b = st.columns([4, 1])
        what = LABELS[r.symptom] + (" (stated absent)" if not r.present else "")
        detail = ", ".join(x for x in [
            r.intensity if isinstance(r.intensity, str) else None,
            f"{int(r.episode_count)} episode(s)" if pd.notna(r.episode_count) else None,
            f"value {r.score:g}" if pd.notna(r.score) else None] if x)
        a.markdown(f"**{r.patient_id} · {r.submitted_at.replace('T', ' ')} · {what}** {('— ' + detail) if detail else ''}")
        a.markdown(f"> {r.raw_text}")
        a.caption(f"Evidence: “{r.evidence}” · flagged because: {r.review_reason} · by {r.method}")
        if b.button("Confirm", key=f"c{r.obs_id}", type="primary"):
            with conn() as c:
                c.execute("UPDATE symptom_observations SET review_status='confirmed' WHERE obs_id=?", (r.obs_id,))
                c.commit()
                rebuild_daily(c)
            st.rerun()
        if b.button("Reject", key=f"r{r.obs_id}"):
            with conn() as c:
                c.execute("UPDATE symptom_observations SET review_status='rejected' WHERE obs_id=?", (r.obs_id,))
                c.commit()
                rebuild_daily(c)
            st.rerun()

st.markdown("#### Same-day conflicts between the check-in form and a message")
conf = q("SELECT patient_id, day_date, symptom, conflict_note FROM daily_symptoms WHERE conflict=1 "
         "ORDER BY day_date DESC")
st.dataframe(conf, hide_index=True, width="stretch")
st.caption("Conflicts are not resolved automatically: the chart keeps the form score and marks the cell ◆ "
           "so the provider sees both.")

st.markdown("#### Rejected by the grounding check")
rej = q("SELECT report_id, reason, payload FROM rejected_extractions")
if rej.empty:
    st.write("None. (Rows land here when a model quotes evidence that isn't in the patient's message.)")
else:
    st.dataframe(rej, hide_index=True, width="stretch")
