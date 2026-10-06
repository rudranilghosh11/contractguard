# ContractGuard — Step-by-Step Guide (Group 44)

## 0. Why this topic (one-paragraph justification for faculty)
Contract review is a real, expensive, high-stakes NLP task. CUAD gives expert-labelled clauses from 510 real
contracts, so we are not inventing labels. The topic is *outside the suggested list* but meets every CA3 rule:
a documented dataset, ≥2 text subgroups (clause length, contract type), one explainability check
(coefficients + LIME), and a pitchable product. It sits closest to #10 "Complaint Classification" and #16
"Resume/JD Classification" (text → category) but is a different domain, so it needs faculty approval — email/ask first.
No other team on the registration sheet has a contract or legal topic.

## 1. Division of work (swap as you like)
| Member | Owns | Deliverables |
|---|---|---|
| **Aarav (team leader)** | Data + model (Steps 2–4) | `prepare_data.py`, `train_audit.py`, GitHub repo, registration row |
| **Aryan** | Audit + explainability + error analysis (Steps 5–7) | Audit tables/charts, LIME, 60-sentence rule validation, error notes |
| **Rudranil** | App + pitch + report polish (Steps 8–10) | Streamlit demo, slides, demo video, final report |
Everyone must be able to explain *every* step in the viva.

## 2. Get approval & register (Day 1)
1. Ask faculty to approve the off-list topic (use the justification above).
2. Aarav pastes the filled row from `NLP_Project_Team_Registration_Group44_filled.xlsx` into the team sheet
   (row NLP-44; **do not change headings, formulas or status values**).
3. After Step 3 add the GitHub link in column N *before the first review*.

## 3. Set up (Day 1)
```bash
git init contractguard && cd contractguard      # or create repo on GitHub, then clone
# copy the project files in
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Create a public GitHub repo, push, add `.gitignore` for `data/raw/` and `.venv/`.

## 4. BUILD — data (Day 1–2)
`python src/prepare_data.py`
* Downloads CUADv1.json from the official Atticus GitHub.
* For each of 14 clause types, takes the expert-highlighted spans (5–300 words).
* Adds ~1,800 **Neutral / Other** chunks from the same contracts that overlap none of the 14 types.
* Expect ≈6,970 rows from 473 contracts.
**Look at the data** (notebook §1.1): class balance, length distribution, 5 random clauses. Put 2 plots in the report.

## 5. BUILD — model (Day 2–3)
`python src/train_audit.py`
* **Split by contract** (GroupShuffleSplit): clauses from the same contract never sit in both train and test.
  Random clause splitting would inflate scores because contracts reuse their own boilerplate. Say this in the report.
* Features: TF-IDF word 1–2-grams. Models compared: Logistic Regression, Linear SVM, Complement NB.
* We choose Logistic Regression because its coefficients are directly interpretable, not because it is far ahead.
Reference numbers from our run: accuracy 0.845, macro-F1 0.818 (LR) vs 0.815 (SVM) vs 0.743 (NB).

## 6. Risk tiers + obligations (Day 3)
`src/risk_rules.py`
* **Risk tier** = our mapping from clause type (High: uncapped liability, liquidated damages, non-compete,
  exclusivity, change of control, IP assignment; Medium: termination for convenience, anti-assignment, minimum
  commitment, ROFR, covenant not to sue, audit rights; Protective: liability cap, insurance).
  This is an **assumption** (viewpoint = small company signing). Defend it; discuss whether a lawyer would agree.
* **Obligation engine**: regex for *shall/must/will* (OBLIGATION), *shall not/may not* (PROHIBITION),
  *may/has the right* (RIGHT); also pulls deadlines ("within 30 days"), amounts ($, ₹) and the acting party.
* Validate it: label the 60-sentence template the notebook creates (`outputs/obligation_validation_template.csv`)
  and report rule-vs-human agreement. Honest numbers (even 75%) beat no numbers.

## 7. AUDIT (Day 4) — the heart of "Build–Audit–Pitch"
Questions to answer with tables/plots from `outputs/`:
1. **By clause length** (short ≤40 / medium / long >100 words). Our run: accuracy 0.862 → 0.792 (≈7-pt drop for long clauses).
2. **By contract type** (distribution, license, services, supply, marketing/alliance). Our run: macro-F1 from 0.829
   (Marketing/Alliance) down to 0.664 (Distribution/Reseller).
3. **By clause type** (`per_class_report.csv`): *Uncapped Liability* is weakest (F1 ≈ 0.53) because it shares
   vocabulary with *Cap On Liability*.
4. **Business metric:** high-risk precision 0.85 / recall 0.80 — "we miss about 1 in 5 high-risk clauses".
   Missing a risky clause costs more than a false alarm, so recall is what a lawyer would care about.
5. **Mitigation idea (pick one, test it, report before/after):** lower decision threshold for High-risk classes;
   class weights; add char n-grams; try Legal-BERT (`nlpaueb/legal-bert-base-uncased`) as an *optional* stretch.
Note: n per subgroup is small (e.g. 83 License clauses) — mention confidence limits, don't over-claim.
(Your numbers may differ by a point or two with other library versions.)

## 8. EXPLAIN (Day 4–5)
* Global: `outputs/top_features.png` — e.g. Non-Compete ← *competitive, competitor, competing*; Termination
  for Convenience ← *terminate, written notice, any time*.
* Local: LIME bar chart in the notebook for 2–3 example clauses, plus the app's "Why?" words.
* **Error analysis:** `outputs/error_analysis_top40.csv` (most confident wrong predictions). Pick 3, explain each.
  Typical story: *Uncapped liability ↔ Cap on liability* (one word "except" flips meaning — bag-of-words can't see scope).

## 9. PITCH app (Day 5)
`streamlit run app.py`
* Tab 1: paste a clause → type, tier, confidence, plain-English meaning, "Why?" words, obligations.
* Tab 2: upload/paste a full contract (use `data/sample_contract.txt`) → flagged clauses, "Missing protection" warning,
  downloadable obligation register.
Record a 2-minute screen-recording as backup in case live demo fails.

## 10. Report, slides, viva (Day 6–7)
Use `docs/REPORT_AND_PITCH_TEMPLATE.md`. Rehearse the viva questions at the bottom of that file.

## 11. Submission checklist
- [ ] Faculty approved topic · registration row filled · GitHub link added
- [ ] Repo runs from scratch with the 3 commands in README
- [ ] Split by contract explained · 3 models compared · metrics table
- [ ] ≥2 audit subgroups (length + contract type) with table **and** chart
- [ ] Explainability (coefficients + LIME) + 3 error cases explained
- [ ] Limitations + "not legal advice" stated · CUAD cited (CC BY 4.0)
- [ ] Slides + demo (live + recorded) · every member can answer questions
