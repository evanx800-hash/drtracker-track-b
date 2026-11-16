import plotly.graph_objects as go
import streamlit as st

from common import INK_2, SERIES_1, SURFACE, conn
from pipeline import queries
from sidebar import patient_picker

pid, p = patient_picker()
st.markdown("## Queries")
st.caption("Plain SQL over the stored observations (pipeline/queries.sql).")

qs = queries.load()
name = st.selectbox("Query", list(qs), format_func=lambda n: n.replace("_", " "))
st.markdown(qs[name]["description"].replace("\n", "  \n"))
params = {}
if ":patient_id" in qs[name]["sql"]:
    params = {"patient_id": pid, "last_visit": p["last_visit_date"]}
    st.caption(f"Parameters: patient {pid}, last visit {p['last_visit_date']}")
with conn() as c:
    df = queries.run(c, name, **params)
st.dataframe(df, hide_index=True, width="stretch")

if name == "nausea_by_days_since_cisplatin" and len(df):
    fig = go.Figure(go.Bar(x=df["days_since_cisplatin"], y=df["avg_nausea_0_4"], marker_color=SERIES_1,
                           hovertemplate="Day %{x} after dose: %{y:.2f}<extra></extra>"))
    fig.update_layout(height=300, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text="Average nausea (0–4) by days since cisplatin, all patients",
                                 font=dict(size=13)), bargap=0.35)
    fig.update_xaxes(title="Days since cisplatin dose", dtick=1, tickfont=dict(color=INK_2))
    fig.update_yaxes(range=[0, 4], gridcolor="#ecebe7", tickfont=dict(color=INK_2))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
if name == "vomiting_per_week" and len(df):
    sub = df[df["patient_id"] == pid]
    fig = go.Figure(go.Bar(x=sub["treatment_week"], y=sub["vomiting_episodes"], marker_color=SERIES_1,
                           hovertemplate="Week %{x}: %{y} episodes<extra></extra>"))
    fig.update_layout(height=280, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text=f"{p['display_name']}: vomiting episodes per treatment week", font=dict(size=13)),
                      bargap=0.35)
    fig.update_xaxes(title="Treatment week", dtick=1)
    fig.update_yaxes(gridcolor="#ecebe7")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with st.expander("SQL"):
    st.code(qs[name]["sql"], language="sql")
