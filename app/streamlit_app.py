"""Dr.Tracker · Track B prototype.  Run from the project root:

    streamlit run app/streamlit_app.py
"""
import streamlit as st

from common import ensure_db

st.set_page_config(page_title="Dr.Tracker · patient-reported data", page_icon="📈", layout="wide")
ensure_db()

pages = [
    st.Page("views/overview.py", title="Patient overview", icon=":material/monitoring:", default=True),
    st.Page("views/parse.py", title="Parse a report (live)", icon=":material/edit_note:"),
    st.Page("views/review.py", title="Review queue", icon=":material/fact_check:"),
    st.Page("views/queries.py", title="Queries", icon=":material/database:"),
    st.Page("views/evaluation.py", title="Parser evaluation", icon=":material/science:"),
    st.Page("views/schema.py", title="Schema & design", icon=":material/schema:"),
]
st.navigation(pages).run()
