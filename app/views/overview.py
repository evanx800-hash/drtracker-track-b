from datetime import date, timedelta

import streamlit as st

from common import (callouts, course_line, overview_figure, patients, recent_messages, as_of_date, fmt_day)
from sidebar import patient_picker

pid, p = patient_picker()
as_of = as_of_date(pid)
default_lv = date.fromisoformat(p["last_visit_date"])
last_visit = st.sidebar.date_input("Last visit", value=default_lv,
                                   min_value=date.fromisoformat(p["treatment_start"]), max_value=as_of)

line, day_no = course_line(p, as_of)
st.markdown(f"## {p['display_name']}")
st.caption(f"{p['diagnosis']} · {line} · as of {fmt_day(as_of)} · synthetic patient")

items = callouts(pid, last_visit)
icon = {"critical": "⛔", "warning": "⚠️", "info": "ℹ️"}
with st.container(border=True):
    st.markdown(f"**Since last visit ({fmt_day(last_visit)})** · rule-based, from what the patient reported")
    left, right = st.columns(2)
    urgent = [i for i in items if i[0] != "info"]
    other = [i for i in items if i[0] == "info"]
    with left:
        for lvl, label, text in urgent or [("info", "Nothing flagged", "No worsening, conflicts or alerts.")]:
            st.markdown(f"{icon[lvl]} **{label}** — {text}")
    with right:
        for lvl, label, text in other:
            st.markdown(f"{icon[lvl]} **{label}** — {text}")

st.plotly_chart(overview_figure(pid, last_visit), width="stretch",
                config={"displayModeBar": False})
st.caption("Each row is on its own scale; darker = worse. Hover any cell for the patient's actual answer and "
           "words. Gray column = no check-in (a gap, not 'no symptoms'). White = not asked / not mentioned. "
           "○ = only in a message (shown as the patient's words, not converted to a score). "
           "◆ = the form and a message the same day disagree. Dotted lines = cisplatin; solid = prescription change.")

st.markdown("#### What the patient wrote since last visit")
msgs = recent_messages(pid, last_visit)
if msgs.empty:
    st.write("No free-text messages in this window.")
else:
    st.dataframe(msgs, hide_index=True, width="stretch",
                 column_config={"Patient wrote": st.column_config.TextColumn(width="large")})
    st.caption("⚑ = extracted but awaiting review; not yet charted.")
