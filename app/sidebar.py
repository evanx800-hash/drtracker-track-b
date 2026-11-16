import streamlit as st

from common import patients


def patient_picker():
    ps = patients()
    labels = {r.patient_id: f"{r.display_name} ({r.patient_id})" for r in ps.itertuples()}
    if "pid" not in st.session_state:
        st.session_state.pid = ps.iloc[0]["patient_id"]
    pid = st.sidebar.selectbox("Patient", list(labels), format_func=labels.get,
                               index=list(labels).index(st.session_state.pid))
    st.session_state.pid = pid
    st.sidebar.caption("All patients and reports are synthetic, modeled on the Dr.Tracker oncology demo "
                       "(head & neck radiation + weekly cisplatin).")
    return pid, ps[ps["patient_id"] == pid].iloc[0]
