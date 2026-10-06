"""
train_audit.py - Steps 2-4: BUILD the model, AUDIT it, EXPLAIN it.

Run:  python src/train_audit.py
Out:  models/contractguard.joblib           (TF-IDF + Logistic Regression pipeline)
      outputs/metrics.json                  (all headline numbers for your report)
      outputs/model_comparison.csv
      outputs/per_class_report.csv
      outputs/audit_by_length.csv, audit_by_doctype.csv
      outputs/confusion_matrix.png, audit_charts.png, top_features.csv, top_features.png
"""
import json, os, sys
import numpy as np, pandas as pd
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, precision_recall_fscore_support)
sys.path.insert(0, os.path.dirname(__file__))
from risk_rules import RISK_MAP

SEED = 42
os.makedirs("outputs", exist_ok=True); os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/clauses.csv")

# ---- 1. Split BY CONTRACT (no clause from the same contract in both train and test) ----
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, df["label"], groups=df["doc"]))
train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
print(f"train={len(train)}  test={len(test)}  (contracts: {train.doc.nunique()} / {test.doc.nunique()})")

def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                           sublinear_tf=True, lowercase=True, stop_words=None)

# ---- 2. Compare models ----
models = {
    "TF-IDF + Logistic Regression": LogisticRegression(max_iter=2000, C=10, class_weight="balanced"),
    "TF-IDF + Linear SVM":          LinearSVC(C=1.0, class_weight="balanced"),
    "TF-IDF + Complement NB":       ComplementNB(alpha=0.3),
}
comp, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)]).fit(train.text, train.label)
    pred = pipe.predict(test.text); fitted[name] = pipe
    comp.append({"model": name,
                 "accuracy": accuracy_score(test.label, pred),
                 "macro_f1": f1_score(test.label, pred, average="macro"),
                 "weighted_f1": f1_score(test.label, pred, average="weighted")})
comp = pd.DataFrame(comp).round(4); comp.to_csv("outputs/model_comparison.csv", index=False)
print(comp.to_string(index=False))

best_name = "TF-IDF + Logistic Regression"   # chosen because it is explainable (coefficients)
pipe = fitted[best_name]
test["pred"] = pipe.predict(test.text)
test["risk_true"] = test.label.map(RISK_MAP); test["risk_pred"] = test.pred.map(RISK_MAP)

# ---- 3. Overall + per-class ----
rep = classification_report(test.label, test.pred, output_dict=True, zero_division=0)
pd.DataFrame(rep).T.round(3).to_csv("outputs/per_class_report.csv")
labels = sorted(df.label.unique())
cm = confusion_matrix(test.label, test.pred, labels=labels)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
plt.xticks(rotation=60, ha="right"); plt.ylabel("True"); plt.xlabel("Predicted")
plt.title("ContractGuard - confusion matrix (test contracts)"); plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()

# ---- 4. Risk-tier view: the business-relevant metric ----
hi_true = test.risk_true == "High"; hi_pred = test.risk_pred == "High"
p, r, f, _ = precision_recall_fscore_support(hi_true, hi_pred, average="binary", zero_division=0)
risk_metrics = {"high_risk_precision": p, "high_risk_recall": r, "high_risk_f1": f}
flag_true = test.risk_true.isin(["High", "Medium"]); flag_pred = test.risk_pred.isin(["High", "Medium"])
p2, r2, f2, _ = precision_recall_fscore_support(flag_true, flag_pred, average="binary", zero_division=0)
risk_metrics.update({"flag_high_or_medium_precision": p2, "flag_high_or_medium_recall": r2})

# ---- 5. AUDIT across subgroups ----
def group_scores(frame, col):
    rows = []
    for g, sub in frame.groupby(col, observed=True):
        if len(sub) < 30:
            continue
        hi_t = sub.risk_true == "High"; hi_p = sub.risk_pred == "High"
        rows.append({col: g, "n": len(sub),
                     "accuracy": accuracy_score(sub.label, sub.pred),
                     "macro_f1": f1_score(sub.label, sub.pred, average="macro", zero_division=0),
                     "high_risk_recall": (hi_t & hi_p).sum() / max(hi_t.sum(), 1),
                     "n_high_risk": int(hi_t.sum())})
    return pd.DataFrame(rows).round(3)

test["length_group"] = pd.cut(test.n_words, bins=[0, 40, 100, 10_000],
                              labels=["Short (<=40 words)", "Medium (41-100)", "Long (>100)"])
by_len = group_scores(test, "length_group"); by_len.to_csv("outputs/audit_by_length.csv", index=False)
by_doc = group_scores(test, "doc_type");     by_doc.to_csv("outputs/audit_by_doctype.csv", index=False)
print("\nAUDIT by clause length\n", by_len.to_string(index=False))
print("\nAUDIT by contract type\n", by_doc.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, (tbl, col, title) in zip(ax, [(by_len, "length_group", "By clause length"),
                                     (by_doc, "doc_type", "By contract type")]):
    m = tbl.melt(id_vars=[col, "n"], value_vars=["accuracy", "macro_f1", "high_risk_recall"])
    sns.barplot(data=m, x=col, y="value", hue="variable", ax=a)
    a.set_ylim(0, 1.05); a.set_title(title); a.set_xlabel(""); a.legend(title="")
    a.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig("outputs/audit_charts.png", dpi=150); plt.close()

# ---- 6. Explainability: top-weighted n-grams per clause type ----
vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
vocab = np.array(vec.get_feature_names_out())
top = []
for i, cls in enumerate(clf.classes_):
    for j in np.argsort(clf.coef_[i])[::-1][:10]:
        top.append({"clause_type": cls, "term": vocab[j], "weight": round(float(clf.coef_[i][j]), 3)})
pd.DataFrame(top).to_csv("outputs/top_features.csv", index=False)
show = ["Uncapped Liability", "Cap On Liability", "Non-Compete", "Termination For Convenience",
        "Liquidated Damages", "Change Of Control"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for a, cls in zip(axes.ravel(), show):
    d = pd.DataFrame(top); d = d[d.clause_type == cls].head(8)[::-1]
    a.barh(d.term, d.weight, color="#2b6cb0"); a.set_title(cls, fontsize=10)
plt.suptitle("Top words/phrases driving each prediction (LR coefficients)"); plt.tight_layout()
plt.savefig("outputs/top_features.png", dpi=150); plt.close()

# ---- 7. Error analysis: most confident mistakes ----
proba = pipe.predict_proba(test.text)
test["confidence"] = proba.max(axis=1)
errs = test[test.label != test.pred].sort_values("confidence", ascending=False)
errs[["label", "pred", "confidence", "n_words", "doc_type", "text"]].head(40).to_csv(
    "outputs/error_analysis_top40.csv", index=False)

# ---- 8. Save everything ----
joblib.dump(pipe, "models/contractguard.joblib")
metrics = {"n_train": len(train), "n_test": len(test), "best_model": best_name,
           "accuracy": accuracy_score(test.label, test.pred),
           "macro_f1": f1_score(test.label, test.pred, average="macro"),
           "weighted_f1": f1_score(test.label, test.pred, average="weighted"), **risk_metrics}
json.dump({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
          open("outputs/metrics.json", "w"), indent=2)
print("\nHEADLINE:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=2))
