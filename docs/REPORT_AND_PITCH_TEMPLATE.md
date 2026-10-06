# Report & Pitch Template — ContractGuard

## A. Report (6–8 pages)
1. **Title, team, PRNs, GitHub link**
2. **Problem & users** — founders/SMEs sign contracts without lawyers; manual review is slow and costly. (3–4 lines)
3. **Data** — CUAD v1 (510 contracts, 41 clause types, CC BY 4.0). What we kept: 14 types + Neutral; counts table; class-balance chart.
4. **Method (Build)** — preprocessing, TF-IDF 1–2 grams, 3 models, split by contract (and why), risk-tier map, obligation rules.
5. **Results** — table: accuracy / macro-F1 / weighted-F1 for 3 models; per-class F1; confusion matrix.
6. **Audit** — length table+chart, contract-type table+chart, high-risk recall; what gap, what cause, what you tried.
7. **Explainability** — top features figure, 2 LIME examples, 3 error cases.
8. **Limitations & ethics** — US-only SEC contracts; tiers are our assumption; neutral-class assumption; no negation handling; not legal advice; privacy (never upload confidential contracts to a public demo).
9. **Future work** — Legal-BERT, span-level extraction (CUAD QA setup), Indian-contract data, party-aware risk (who benefits).
10. **References** — Hendrycks et al. 2021 (CUAD, NeurIPS Datasets & Benchmarks); scikit-learn; LIME (Ribeiro et al. 2016).

## B. Pitch deck (8 slides, ~5 min)
| # | Slide | Say |
|---|---|---|
| 1 | **Hook** | "A founder signs a vendor contract. Buried on page 14: unlimited liability. Nobody noticed." |
| 2 | **Problem** | Contracts are long, lawyers are expensive, risk hides in specific clause types. |
| 3 | **Solution: ContractGuard** | Paste a contract → risky clauses flagged by tier + obligations register. |
| 4 | **How it works** | CUAD → TF-IDF + LR → risk tier + rule engine. One diagram. |
| 5 | **Live demo** | Sample contract: 🔴 non-compete, 🟠 termination, ⚠️ missing liability cap. |
| 6 | **Audit — our honesty slide** | Overall 0.845 acc… but long clauses −7 pts, Distribution contracts macro-F1 0.66, Uncapped Liability F1 0.53. We measured it, not hid it. |
| 7 | **Explainability** | Top words per clause type + one LIME chart. "No black box." |
| 8 | **Limits & next steps** | Not legal advice; US data; Legal-BERT & Indian contracts next. |

**Pitch tips:** open with the story not the model; show the weakest number yourselves — the "Audit" part of the
rubric rewards that; keep the demo to 60–90 seconds on a pre-loaded sample; one speaker per section.

## C. Likely viva questions (prepare 2-line answers)
1. Why split by contract instead of randomly? (leakage via boilerplate)
2. Why TF-IDF + LR rather than BERT? (explainability, small data, CPU; BERT is future work)
3. How did you create the Neutral class and what could go wrong? (un-annotated ≠ risk-free)
4. Where do risk tiers come from? Would a lawyer agree? (our assumption; perspective-dependent)
5. Which metric matters most, and why? (high-risk recall)
6. Why is Uncapped Liability the weakest class? (vocabulary overlap with Cap On Liability; negation/scope)
7. What's the biggest limitation? (US SEC data, bag-of-words, small subgroup sizes)
8. How do you know your obligation rules work? (60-sentence hand validation)
