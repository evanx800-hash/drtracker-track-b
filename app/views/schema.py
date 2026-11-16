import streamlit as st

from common import ROOT, q

st.markdown("## Schema & design")
st.markdown("""
**Event model.** Every submission is stored untouched in `reports`. Each symptom statement becomes one row in
`symptom_observations`, linked back to its report and carrying the exact words that support it. The chart reads a
`daily_symptoms` rollup that is rebuilt from observations by documented rules – it is never edited by hand.

**Vague reports.** Unknown stays unknown: `score`, `intensity` and `episode_count` are NULL unless the patient
stated them ("threw up a few times" → present, count NULL). Vague complaints ("feel kinda off") are stored as
`general_unwell` and are never mapped to a specific symptom. `time_precision` records whether the time is a part of
day or just the day.

**Absent vs not mentioned.** "No vomiting" is a row with `present = 0`. Not mentioning vomiting is no row. The chart
shows these differently, and a missed check-in is a gray gap, not a zero.

**Conflicts.** Nothing is overwritten or averaged. If the form says nausea 1 and a message the same morning says
"worst nausea so far", both rows stay; the rollup keeps the form score and marks the day as a conflict (◆) with
both sources quoted.

**Scores vs words.** Questionnaire answers are scores. Free text keeps the patient's intensity *word*; a number is
stored only if the patient typed one. Text-only days are drawn with a ○ marker so nobody mistakes them for a form score.

**Review.** Model output is grounded (evidence must appear in the message), cross-checked against the rule parser,
and anything hedged, low-confidence or safety-relevant (fever) waits for a person in the review queue.
""")
c1, c2 = st.columns(2)
c1.metric("Reports", int(q("SELECT COUNT(*) n FROM reports")["n"].iloc[0]))
c2.metric("Symptom observations", int(q("SELECT COUNT(*) n FROM symptom_observations")["n"].iloc[0]))
st.markdown("#### Example rows")
st.dataframe(q("SELECT o.report_id, o.symptom, o.present, o.score, o.intensity, o.episode_count, o.occurred_date, "
               "o.time_of_day, o.source, o.method, o.evidence, o.review_status FROM symptom_observations o "
               "WHERE o.source='free_text' LIMIT 25"), hide_index=True, width="stretch")
with st.expander("schema.sql"):
    st.code((ROOT / "pipeline" / "schema.sql").read_text(), language="sql")
