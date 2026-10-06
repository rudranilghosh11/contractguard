# 🛡️ ContractGuard — Legal Contract Clause Risk & Obligation Detection
NLP CA3 · Build–Audit–Pitch · Group 44 (Aarav Seth, Aryan Bakshi, Rudranil Ghosh)

ContractGuard reads contract text, flags risk-relevant clauses (uncapped liability, non-compete,
exclusivity, liquidated damages, change of control, …), assigns a High / Medium / Protective risk tier,
warns about missing protections, and extracts obligations, prohibitions, rights, deadlines and amounts.

**Data:** CUAD v1 — Contract Understanding Atticus Dataset (The Atticus Project, CC BY 4.0).
Hendrycks et al., 2021, https://arxiv.org/abs/2103.06268 · https://github.com/TheAtticusProject/cuad
> Educational prototype. Not legal advice.

## Quick start
```bash
pip install -r requirements.txt
python src/prepare_data.py     # downloads CUAD (~18 MB) -> data/clauses.csv
python src/train_audit.py      # trains, audits, explains -> models/ + outputs/
streamlit run app.py           # demo (try data/sample_contract.txt in the "Full contract scan" tab)
```
Or open `notebooks/ContractGuard.ipynb` (Colab / Jupyter) and run all cells.

## Layout
| Path | What |
|---|---|
| `src/prepare_data.py` | CUAD → clause-level CSV (14 risk types + Neutral) |
| `src/train_audit.py` | Build (3 models) · Audit (length, contract type) · Explain (coefficients) |
| `src/risk_rules.py` | Risk-tier map (our assumption) + obligation/prohibition/right extractor |
| `app.py` | Streamlit demo |
| `notebooks/ContractGuard.ipynb` | Narrated notebook for submission |
| `outputs/` | Metrics, audit tables, confusion matrix, charts, error analysis |
| `docs/` | Step-by-step guide, report & pitch templates |

## Results (seed 42, split by contract)
See `outputs/metrics.json` — overall accuracy 0.845, macro-F1 0.818, high-risk recall 0.795.
