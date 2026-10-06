"""
prepare_data.py - Step 1. Download CUAD v1 and turn it into a clean clause-level CSV.

Run:  python src/prepare_data.py
Out:  data/clauses.csv   (text, label, doc, doc_type, n_words)

CUAD = Contract Understanding Atticus Dataset (The Atticus Project, CC BY 4.0).
510 real commercial contracts (SEC filings), 41 expert-labelled clause types.
We keep 14 risk-relevant clause types + a 'Neutral / Other' class built from
contract text that does NOT overlap any of those labelled clauses.
"""
import json, os, random, re, zipfile, io, sys
import requests
import pandas as pd

URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
RAW_DIR = "data/raw"
TARGET = [
    "Cap On Liability", "Uncapped Liability", "Liquidated Damages",
    "Termination For Convenience", "Non-Compete", "Exclusivity", "Anti-Assignment",
    "Change Of Control", "Audit Rights", "Insurance", "Ip Ownership Assignment",
    "Minimum Commitment", "Rofr/Rofo/Rofn", "Covenant Not To Sue",
]
NEUTRAL = "Neutral / Other"
SEED = 42
MIN_WORDS, MAX_WORDS = 5, 300          # drop tiny fragments / huge blobs
NEUTRAL_N = 1800                       # number of neutral chunks to sample


def download():
    path = os.path.join(RAW_DIR, "CUADv1.json")
    if os.path.exists(path):
        return path
    os.makedirs(RAW_DIR, exist_ok=True)
    print("Downloading CUAD v1 (~18 MB)...")
    r = requests.get(URL, timeout=180); r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extract("CUADv1.json", RAW_DIR)
    return path


def doc_type(title: str) -> str:
    t = title.upper()
    rules = [
        ("Distribution / Reseller", ["DISTRIBUT", "RESELL", "RESELLER", "AGENCY", "REFERRAL"]),
        ("License / IP",            ["LICENS", "LICENCE", "INTELLECTUAL", "TRADEMARK", "PATENT", "ROYALTY"]),
        ("Services / Maintenance",  ["SERVICE", "MAINTENANCE", "CONSULT", "HOSTING", "OUTSOURC", "SUPPORT", "MANAGEMENT"]),
        ("Supply / Manufacturing",  ["SUPPLY", "MANUFACTUR", "PURCHASE", "OEM"]),
        ("Marketing / Alliance",    ["MARKETING", "SPONSOR", "AFFILIATE", "ALLIANCE", "PROMOTION", "CO-BRAND", "COBRAND", "ENDORSE", "JOINT", "STRATEGIC", "COOPERATION", "COLLABORATION", "DEVELOPMENT", "FRANCHISE"]),
    ]
    for name, kws in rules:
        if any(k in t for k in kws):
            return name
    return "Other"


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def neutral_chunks(context, spans, rng):
    """Paragraph-ish chunks of the contract that don't overlap any target clause span."""
    out, pos = [], 0
    for para in re.split(r"\n\s*\n|(?<=[\.;])\s{2,}", context):
        start = context.find(para, pos)
        if start < 0:
            continue
        end = start + len(para); pos = end
        txt = clean(para); n = len(txt.split())
        if n < 12 or n > MAX_WORDS:
            continue
        if any(not (end <= s or start >= e) for s, e in spans):
            continue
        out.append(txt)
    return out


def main():
    rng = random.Random(SEED)
    data = json.load(open(download()))["data"]
    rows, neutral_pool = [], []
    for doc in data:
        title = doc["title"]; para = doc["paragraphs"][0]; ctx = para["context"]
        spans = []
        for qa in para["qas"]:
            cat = qa["id"].split("__")[-1]
            if cat not in TARGET:
                continue
            seen = set()
            for a in qa["answers"]:
                txt = clean(a["text"]); n = len(txt.split())
                if txt in seen or n < MIN_WORDS or n > MAX_WORDS:
                    continue
                seen.add(txt)
                rows.append({"text": txt, "label": cat, "doc": title,
                             "doc_type": doc_type(title), "n_words": n})
                spans.append((a["answer_start"], a["answer_start"] + len(a["text"])))
        for c in neutral_chunks(ctx, spans, rng):
            neutral_pool.append({"text": c, "label": NEUTRAL, "doc": title,
                                 "doc_type": doc_type(title), "n_words": len(c.split())})
    rng.shuffle(neutral_pool)
    rows += neutral_pool[:NEUTRAL_N]
    df = pd.DataFrame(rows).drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/clauses.csv", index=False)
    print(f"Saved data/clauses.csv  rows={len(df)}  contracts={df['doc'].nunique()}")
    print(df["label"].value_counts().to_string())
    print(df["doc_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
