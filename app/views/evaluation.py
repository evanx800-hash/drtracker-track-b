import pandas as pd
import streamlit as st

from common import ROOT, eval_results

st.markdown("## Parser evaluation")
st.caption("Scored against gold labels. Regenerate with `python -m eval.evaluate` and "
           "`python -m eval.evaluate --backend ollama`.")
st.markdown("""
- **generated** – free-text reports rendered from the synthetic ground truth. The rule parser shares
  vocabulary with the generator, so rules look better here than they really are.
- **hard_dev** – 30 hand-written edge cases. The rule lexicon was tuned while looking at these.
- **holdout** – 20 hand-written cases written *after* the rules and the LLM prompt were frozen. The honest test.
""")
res = eval_results()
if not res:
    st.warning("No results yet. Run `python -m eval.evaluate`.")
    st.stop()

metric_names = {
    "precision": "Precision (extracted items that are real)",
    "recall": "Recall (real items that were extracted)",
    "f1": "F1",
    "present_vs_absent_accuracy": "Present vs absent correct",
    "intensity_accuracy": "Intensity word correct",
    "vomiting_count_exact": "Vomiting count exact",
    "vague_count_left_null": "Vague count left blank (not guessed)",
    "patient_numbers_exact": "Patient-given number exact",
    "day_accuracy": "Correct day",
    "wrong_items_flagged_for_review": "Wrong items caught by review flags",
    "correct_items_flagged_for_review": "Correct items flagged (review burden)",
    "medication_recall": "Medication mention recall",
    "median_latency_s": "Median latency (s)",
}
for set_name in ["holdout", "hard_dev", "generated"]:
    st.markdown(f"#### {set_name}")
    tbl = {b: {metric_names[k]: v for k, v in r[set_name].items() if k in metric_names} for b, r in res.items()}
    df = pd.DataFrame(tbl)
    st.dataframe(df, width="stretch", height=(len(df) + 1) * 35 + 3)

errs = sorted((ROOT / "eval").glob("errors_*.csv"))
if errs:
    st.markdown("#### Error log")
    pick = st.selectbox("Backend", [e.stem.replace("errors_", "") for e in errs])
    df = pd.read_csv(ROOT / "eval" / f"errors_{pick}.csv")
    which = st.radio("Set", ["holdout", "hard_dev", "generated"], horizontal=True)
    st.dataframe(df[df["set"] == which].drop(columns=["set"]), hide_index=True, width="stretch")
