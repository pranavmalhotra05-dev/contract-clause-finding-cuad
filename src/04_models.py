"""Clause finding under four evaluation splits, three scorers.
Task: given a contract split into segments, rank segments by how likely they hold a clause of one category.
Scorers: keyword rules (10 categories), TF-IDF + logistic regression, MiniLM embeddings + logistic regression.
Splits: random segments (leaky), random contracts, same-company-held-out, same-agreement-type-held-out."""
import pathlib, re, sys, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, KFold
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
seg = pd.read_parquet(ROOT / "data/segments.parquet")
import duckdb
meta = duckdb.connect(str(ROOT / "data/cuad.duckdb"), read_only=True).execute("select * from contracts").df().set_index("contract_id")
META = ["contract_id", "seg_id", "start", "end", "text", "rel_pos"]
CATS = [c for c in seg.columns if c not in META and c not in ("Document Name", "Parties")]
EMB = np.load(ROOT / "data/emb_minilm.npy")
K = 5; SEED = 0

RULES = {  # hand-written regexes, one list per category (a transparent, auditable baseline)
    "Governing Law": r"governed by|governing law|construed (in accordance )?(with|under)|laws of the state",
    "Anti-Assignment": r"\bassign(ed|ment)?\b.{0,80}(consent|without)|may not (assign|transfer)",
    "Cap On Liability": r"in no event.{0,200}(exceed|aggregate)|aggregate liability|liability.{0,60}shall not exceed|limit(ed|ation) of liability",
    "Audit Rights": r"\baudit|inspect(ion)? .{0,40}records|books and records",
    "Non-Compete": r"non-?compet|shall not.{0,60}compet|not (directly or indirectly )?(engage|enter).{0,80}(business|compet)",
    "Exclusivity": r"exclusive(ly)?|sole and exclusive",
    "Termination For Convenience": r"terminate.{0,80}(for convenience|without cause|at any time|upon .{0,20} (days|notice))|without cause",
    "Insurance": r"\binsur(ance|ed|er)\b|policy of insurance",
    "License Grant": r"(grants?|hereby grants?).{0,80}licen[sc]e|license is hereby granted|non-exclusive.{0,30}license",
    "Change Of Control": r"change (of|in) control|merger|acquires? .{0,40}(all|substantially all)",
    "Liquidated Damages": r"liquidated damages",
    "Third Party Beneficiary": r"third.party benefici",
    "Most Favored Nation": r"most favou?red|more favou?rable (pricing|terms)",
    "Source Code Escrow": r"escrow",
    "Non-Disparagement": r"disparag",
    "Renewal Term": r"(auto(matic(ally)?)?|successive).{0,40}renew|renew(ed|al).{0,60}(year|term)",
    "Revenue/Profit Sharing": r"revenue shar|profit shar|royalt(y|ies)|percent(age)? of (net |gross )?(revenue|sales|profit)",
    "Covenant Not To Sue": r"covenant not to sue|shall not (bring|commence|assert).{0,60}(claim|action|suit)",
}

def top_hit(scores, y, contract, k=3):
    df = pd.DataFrame(dict(c=contract, s=scores, y=y))
    out = []
    for _, g in df.groupby("c"):
        if g.y.sum() == 0: continue
        out.append(int(g.nlargest(k, "s").y.sum() > 0))
    return np.mean(out) if out else np.nan

def metrics(scores, y, contract):
    d = pd.DataFrame(dict(c=contract, s=scores, y=y))
    cm = d.groupby("c").agg(s=("s", "max"), y=("y", "max"))
    return dict(seg_aupr=average_precision_score(y, scores) if y.sum() else np.nan,
                contract_aupr=average_precision_score(cm.y, cm.s) if cm.y.sum() else np.nan,
                hit_at_3=top_hit(scores, y, contract),
                base_rate=y.mean(), contract_base=cm.y.mean(), n_pos=int(y.sum()))

def folds(scheme):
    cid = seg.contract_id.values
    if scheme == "random_segments":
        return list(KFold(K, shuffle=True, random_state=SEED).split(seg))
    group = {"random_contracts": cid,
             "company_held_out": meta.loc[cid, "company"].values,
             "type_held_out": meta.loc[cid, "agreement_type"].values}[scheme]
    if scheme == "random_contracts":
        rng = np.random.RandomState(SEED); perm = rng.permutation(cid.max() + 1); g = perm[cid] % K
        return [(np.where(g != f)[0], np.where(g == f)[0]) for f in range(K)]
    return list(GroupKFold(K).split(seg, groups=group))

def fit_pred(kind, tr, te, y, cat):
    ytr = y[tr]
    if ytr.sum() < 3: return np.zeros(len(te))
    rng = np.random.RandomState(SEED)
    neg = np.where(ytr == 0)[0]; pos = np.where(ytr == 1)[0]
    keep = np.concatenate([pos, rng.choice(neg, min(len(neg), 12 * len(pos)), replace=False)])
    idx = tr[keep]
    if kind == "tfidf":
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True, stop_words="english")
        Xtr = v.fit_transform(seg.text.values[idx]); Xte = v.transform(seg.text.values[te])
    else:
        Xtr, Xte = EMB[idx], EMB[te]
    m = LogisticRegression(C=4.0, max_iter=300, class_weight="balanced")
    m.fit(Xtr, y[idx]); return m.decision_function(Xte)

def run_cat(cat, scheme, fl):
    y = seg[cat].values; rows = []
    for kind in ["tfidf", "minilm"]:
        oof = np.zeros(len(seg))
        for tr, te in fl: oof[te] = fit_pred(kind, tr, te, y, cat)
        rows.append(dict(category=cat, scheme=scheme, model=kind, **metrics(oof, y, seg.contract_id.values)))
    if cat in RULES:
        rx = re.compile(RULES[cat], re.I)
        s = np.array([1.0 if rx.search(t) else 0.0 for t in seg.text.values])
        rng = np.random.RandomState(1); s = s + rng.rand(len(s)) * 1e-3  # random tie-break
        rows.append(dict(category=cat, scheme=scheme, model="rules", **metrics(s, y, seg.contract_id.values)))
    return rows

if __name__ == "__main__":
    out = []
    for scheme in ["random_segments", "random_contracts", "company_held_out", "type_held_out"]:
        fl = folds(scheme)
        res = Parallel(n_jobs=12)(delayed(run_cat)(c, scheme, fl) for c in CATS)
        out += [r for rs in res for r in rs]
        print(scheme, "done", flush=True)
    pd.DataFrame(out).to_csv(ROOT / "outputs/results_by_category.csv", index=False)
