"""
ContractGuard - Streamlit demo.   Run:  streamlit run app.py
Needs models/contractguard.joblib  (created by  python src/train_audit.py)
"""
import os, sys
import numpy as np, pandas as pd, joblib, streamlit as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from risk_rules import (RISK_MAP, PLAIN_ENGLISH, EXPECTED_PROTECTIONS,
                        extract_obligations, segment_contract)

st.set_page_config(page_title="ContractGuard", page_icon="🛡️", layout="wide")
BADGE = {"High": "🔴 HIGH", "Medium": "🟠 MEDIUM", "Low (protective)": "🟢 PROTECTIVE", "None": "⚪ none"}
ORDER = {"High": 0, "Medium": 1, "Low (protective)": 2, "None": 3}


@st.cache_resource
def load():
    return joblib.load("models/contractguard.joblib")


pipe = load()
vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
vocab = np.array(vec.get_feature_names_out())


def predict(texts):
    proba = pipe.predict_proba(texts)
    idx = proba.argmax(axis=1)
    return clf.classes_[idx], proba.max(axis=1)


def why(text, label, k=6):
    """Top words in THIS clause pushing the model towards its predicted label."""
    X = vec.transform([text]); ci = list(clf.classes_).index(label)
    contrib = X.toarray()[0] * clf.coef_[ci]
    top = np.argsort(contrib)[::-1][:k]
    return [(vocab[i], round(float(contrib[i]), 3)) for i in top if contrib[i] > 0]


st.title("🛡️ ContractGuard")
st.caption("Legal contract clause risk & obligation detection · NLP CA3 · Group 44 · "
           "Trained on CUAD v1 (CC BY 4.0). Educational prototype - not legal advice.")

with st.sidebar:
    st.header("Settings")
    thr = st.slider("Flag a clause only if model confidence ≥", 0.30, 0.95, 0.50, 0.05)
    st.markdown("**Risk tiers** are our own assumption (viewpoint: a small company *signing* the contract).")
    st.dataframe(pd.DataFrame({"Clause type": list(RISK_MAP), "Tier": list(RISK_MAP.values())}),
                 hide_index=True, height=420)

tab1, tab2 = st.tabs(["🔎 Single clause", "📄 Full contract scan"])

EXAMPLES = {
    "Uncapped liability": "Notwithstanding anything to the contrary, Supplier's liability for breach of confidentiality or infringement shall be unlimited.",
    "Non-compete": "During the Term and for two (2) years thereafter, Distributor shall not, directly or indirectly, market or sell any product that competes with the Products.",
    "Termination for convenience": "Either party may terminate this Agreement at any time without cause upon thirty (30) days prior written notice to the other party.",
    "Liability cap": "In no event shall either party's aggregate liability under this Agreement exceed the fees paid in the twelve (12) months preceding the claim.",
}

with tab1:
    ex = st.selectbox("Try an example (or paste your own below)", ["-"] + list(EXAMPLES))
    text = st.text_area("Clause text", value=EXAMPLES.get(ex, ""), height=150)
    if st.button("Analyse clause", type="primary") and text.strip():
        lab, conf = predict([text]); lab, conf = lab[0], float(conf[0])
        tier = RISK_MAP.get(lab, "None")
        c1, c2, c3 = st.columns(3)
        c1.metric("Clause type", lab); c2.metric("Risk tier", BADGE[tier]); c3.metric("Confidence", f"{conf:.0%}")
        if conf < thr:
            st.warning("Confidence is below your threshold - treat this prediction as uncertain.")
        st.info(PLAIN_ENGLISH.get(lab, ""))
        w = why(text, lab)
        if w:
            st.markdown("**Why? Words pushing the model to this label:** " +
                        ", ".join(f"`{t}` ({v})" for t, v in w))
        ob = extract_obligations(text)
        st.subheader("Obligations, prohibitions & rights")
        st.dataframe(pd.DataFrame(ob), hide_index=True, width="stretch") if ob else st.write("None detected.")

with tab2:
    up = st.file_uploader("Upload a contract (.txt)", type=["txt"])
    pasted = st.text_area("...or paste the full contract", height=180, key="full")
    raw = up.read().decode("utf-8", errors="ignore") if up else pasted
    if st.button("Scan contract", type="primary") and raw.strip():
        chunks = segment_contract(raw)
        if not chunks:
            st.error("Could not split the text into clauses. Use blank lines between paragraphs.")
            st.stop()
        labs, confs = predict(chunks)
        df = pd.DataFrame({"clause_type": labs, "confidence": confs.round(2), "text": chunks})
        df["risk"] = df.clause_type.map(RISK_MAP)
        flagged = df[(df.risk != "None") & (df.confidence >= thr)].copy()
        flagged["order"] = flagged.risk.map(ORDER)
        flagged = flagged.sort_values(["order", "confidence"], ascending=[True, False]).drop(columns="order")
        a, b, c, d = st.columns(4)
        a.metric("Clauses scanned", len(df)); b.metric("🔴 High", int((flagged.risk == "High").sum()))
        c.metric("🟠 Medium", int((flagged.risk == "Medium").sum()))
        d.metric("🟢 Protective", int((flagged.risk == "Low (protective)").sum()))
        found = set(flagged.clause_type)
        for p in EXPECTED_PROTECTIONS:
            if p not in found:
                st.error(f"⚠️ Missing protection: no **{p}** clause detected. Check with a lawyer before signing.")
        st.subheader("Flagged clauses (highest risk first)")
        for _, r in flagged.iterrows():
            with st.expander(f"{BADGE[r.risk]} · {r.clause_type} · {r.confidence:.0%}"):
                st.write(r.text); st.caption(PLAIN_ENGLISH.get(r.clause_type, ""))
        st.subheader("Obligation register")
        ob = extract_obligations(raw)
        obdf = pd.DataFrame(ob)
        if len(obdf):
            kinds = st.multiselect("Show", ["OBLIGATION", "PROHIBITION", "RIGHT"], default=["OBLIGATION", "PROHIBITION"])
            st.dataframe(obdf[obdf.type.isin(kinds)], hide_index=True, width="stretch")
            st.download_button("Download register (CSV)", obdf.to_csv(index=False), "obligations.csv")
