"""
risk_rules.py  -  the "legal brain" of ContractGuard.

Two things live here:
  1. RISK_MAP / PLAIN_ENGLISH : our *own* design choice mapping each CUAD clause type to a
     risk tier (High / Medium / Low-Protective) from the viewpoint of a small company or
     startup that is SIGNING the contract. Risk tiers are NOT part of CUAD - say this clearly
     in your report. Change the map if your team disagrees; it is an assumption, not a fact.
  2. extract_obligations() : transparent rule-based detector that tags each sentence as
     OBLIGATION / PROHIBITION / RIGHT / NONE and pulls out deadlines and money amounts.
"""
import re

# ---------------------------------------------------------------------------
# 1. Clause-type -> risk tier (perspective: small business signing the contract)
# ---------------------------------------------------------------------------
RISK_MAP = {
    "Uncapped Liability":          "High",
    "Liquidated Damages":          "High",
    "Non-Compete":                 "High",
    "Exclusivity":                 "High",
    "Change Of Control":           "High",
    "Ip Ownership Assignment":     "High",
    "Termination For Convenience": "Medium",
    "Anti-Assignment":             "Medium",
    "Minimum Commitment":          "Medium",
    "Rofr/Rofo/Rofn":              "Medium",
    "Covenant Not To Sue":         "Medium",
    "Audit Rights":                "Medium",
    "Cap On Liability":            "Low (protective)",
    "Insurance":                   "Low (protective)",
    "Neutral / Other":             "None",
}

PLAIN_ENGLISH = {
    "Uncapped Liability":          "One side can be liable for unlimited damages. Could exceed the contract value many times over.",
    "Liquidated Damages":          "A pre-agreed penalty must be paid on breach, whether or not real loss occurred.",
    "Non-Compete":                 "Restricts you from competing or working with rivals during/after the contract.",
    "Exclusivity":                 "You may only buy from / sell to / work with one party. Limits flexibility.",
    "Change Of Control":           "A merger, acquisition or ownership change can trigger termination or consent rights.",
    "Ip Ownership Assignment":     "Intellectual property you create may be owned by the other party.",
    "Termination For Convenience": "A party can exit without cause. Check who holds this right and the notice period.",
    "Anti-Assignment":             "You cannot transfer the contract (e.g. in a sale or restructuring) without consent.",
    "Minimum Commitment":          "You must buy/pay/deliver a minimum amount regardless of actual demand.",
    "Rofr/Rofo/Rofn":              "The other party gets first right / first offer / first negotiation on future deals.",
    "Covenant Not To Sue":         "A party gives up the right to bring certain legal claims.",
    "Audit Rights":                "The other party may inspect your books/records. Check frequency, notice and cost.",
    "Cap On Liability":            "Limits the maximum damages payable. Protective, but check whose liability is capped.",
    "Insurance":                   "Requires specific insurance cover. Check cost and minimum coverage amounts.",
    "Neutral / Other":             "No risk-relevant clause type detected.",
}

# Protections a founder would normally want to SEE in a contract (missing = flag)
EXPECTED_PROTECTIONS = ["Cap On Liability"]

# ---------------------------------------------------------------------------
# 2. Obligation / prohibition / right detection (rule based, fully explainable)
# ---------------------------------------------------------------------------
_PROHIB = re.compile(
    r"\b(shall|must|will|may|agrees?\s+to|is\s+to|are\s+to)\s+not\b|"
    r"\bmay\s+not\b|\bis\s+prohibited\b|\bare\s+prohibited\b|\bnot\s+be\s+permitted\b|"
    r"\bin\s+no\s+event\b|\bunder\s+no\s+circumstances\b", re.I)
_RIGHT = re.compile(
    r"\bshall\s+have\s+the\s+(?:right|option)\b|\bhas\s+the\s+(?:right|option)\b|"
    r"\bhave\s+the\s+(?:right|option)\b|\bis\s+entitled\s+to\b|\bare\s+entitled\s+to\b|"
    r"\bat\s+its\s+(?:sole\s+)?(?:option|discretion)\b|\bmay\b", re.I)
_OBLIG = re.compile(
    r"\b(shall|must|will|agrees?\s+to|is\s+required\s+to|are\s+required\s+to|"
    r"is\s+obligated\s+to|are\s+obligated\s+to|undertakes?\s+to|"
    r"is\s+responsible\s+for|are\s+responsible\s+for)\b", re.I)

_DEADLINE = re.compile(
    r"\b(?:within|no\s+later\s+than|not\s+less\s+than|at\s+least|prior\s+to|not\s+more\s+than)\s+"
    r"(?:\w+\s+)?\(?\d+\)?\s*(?:\(\d+\)\s*)?(?:business\s+|calendar\s+)?"
    r"(?:days?|weeks?|months?|years?)\b", re.I)
_MONEY = re.compile(r"(?:US\s?)?\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand))?|"
                    r"\b(?:Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?|\u20b9\s?\d[\d,]*(?:\.\d+)?", re.I)
_ACTOR = re.compile(
    r"^\W*(?:\d+(?:\.\d+)*\W+)?(?:Each\s+(?:of\s+)?(?:the\s+)?)?(?:The\s+)?"
    r"((?:[A-Z][\w\.&]*)(?:\s+[A-Z][\w\.&]*){0,2})\s+"
    r"(?:shall|must|will|may|agrees?|is|are|undertakes?)\b")


def split_sentences(text: str):
    """Light-weight sentence splitter that is good enough for contracts."""
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[\.;:])\s+(?=[A-Z0-9\(\"])", text)
    return [p.strip() for p in parts if len(p.split()) >= 3]


def classify_sentence(s: str) -> str:
    if _PROHIB.search(s):
        return "PROHIBITION"
    if _RIGHT.search(s):
        return "RIGHT"
    if _OBLIG.search(s):
        return "OBLIGATION"
    return "NONE"


def extract_obligations(text: str):
    """Return a list of dicts: type, actor, deadlines, amounts, sentence."""
    rows = []
    for s in split_sentences(text):
        t = classify_sentence(s)
        if t == "NONE":
            continue
        m = _ACTOR.search(s)
        rows.append({
            "type": t,
            "actor": m.group(1) if m else "",
            "deadlines": "; ".join(d.group(0) for d in _DEADLINE.finditer(s)),
            "amounts": "; ".join(d.group(0) for d in _MONEY.finditer(s)),
            "sentence": s,
        })
    return rows


def segment_contract(text: str, max_words: int = 160, min_words: int = 8):
    """Split a full contract into clause-sized chunks (paragraphs; long ones -> sentence groups)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) < 3:  # file that only uses single newlines
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for p in paras:
        p = re.sub(r"\s+", " ", p)
        words = p.split()
        if len(words) < min_words:
            continue
        if len(words) <= max_words:
            chunks.append(p)
        else:
            buf = []
            for s in split_sentences(p):
                buf.append(s)
                if len(" ".join(buf).split()) >= max_words // 2:
                    chunks.append(" ".join(buf)); buf = []
            if buf and len(" ".join(buf).split()) >= min_words:
                chunks.append(" ".join(buf))
    return chunks
