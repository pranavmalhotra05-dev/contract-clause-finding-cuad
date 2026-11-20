"""Load CUAD v1, split every contract into sentence-like segments, and label each
segment against the human-marked spans. Writes data/segments.parquet and a DuckDB file."""
import json, re, pathlib, duckdb, pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw/cuad/CUAD_v1/CUAD_v1.json"
OUT = ROOT / "data"; OUT.mkdir(exist_ok=True)
SPLIT = re.compile(r'(?<=[.;:])\s+(?=[A-Z0-9(\"])')

def norm_type(title):
    t = title.split('-')[-1] if '-' in title else title
    t = re.sub(r'^\d+(\.\d+)*_', '', t.strip())
    return re.sub(r'\s+\d+$', '', t.lower().strip())

def segment(text, min_len=40, max_len=900):
    """Split on sentence-like boundaries, merge tiny pieces, cut very long ones."""
    pieces, pos = [], 0
    for m in SPLIT.finditer(text):
        pieces.append((pos, m.start())); pos = m.end()
    pieces.append((pos, len(text)))
    merged = []
    for s, e in pieces:
        if merged and (e - s < min_len or merged[-1][1] - merged[-1][0] < min_len):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        while e - s > max_len:
            out.append((s, s + max_len)); s += max_len
        out.append((s, e))
    return out

def main():
    data = json.load(open(RAW, encoding="utf-8"))["data"]
    contracts, spans, segs = [], [], []
    for ci, a in enumerate(data):
        title = a["title"]; ctx = a["paragraphs"][0]["context"]
        contracts.append(dict(contract_id=ci, title=title, company=re.split(r'_', title)[0].lower(),
                              agreement_type=norm_type(title), n_chars=len(ctx)))
        cat_spans = {}
        for q in a["paragraphs"][0]["qas"]:
            cat = q["id"].split("__")[1]
            for s in q["answers"]:
                spans.append(dict(contract_id=ci, category=cat, start=s["answer_start"],
                                  end=s["answer_start"] + len(s["text"]), text=s["text"]))
                cat_spans.setdefault(cat, []).append((s["answer_start"], s["answer_start"] + len(s["text"])))
        for si, (s, e) in enumerate(segment(ctx)):
            row = dict(contract_id=ci, seg_id=si, start=s, end=e, text=ctx[s:e], rel_pos=s / max(len(ctx), 1))
            for cat, sp in cat_spans.items():
                # positive if the segment is mostly inside a marked span, or it holds most of a short span
                row[cat] = int(any(max(0, min(e, b) - max(s, a_)) >= 0.5 * min(e - s, b - a_) for a_, b in sp))
            segs.append(row)
    cats = sorted({s["category"] for s in spans})
    seg = pd.DataFrame(segs)
    for c in cats:
        seg[c] = seg[c].fillna(0).astype(int) if c in seg else 0
    seg.to_parquet(OUT / "segments.parquet")
    con = duckdb.connect(str(OUT / "cuad.duckdb"))
    con.execute("create or replace table contracts as select * from contracts_df", {}) if False else None
    cdf = pd.DataFrame(contracts); sdf = pd.DataFrame(spans)
    con.register("cdf", cdf); con.register("sdf", sdf)
    con.execute("create or replace table contracts as select * from cdf")
    con.execute("create or replace table spans as select * from sdf")
    print(f"contracts {len(cdf)}, spans {len(sdf)}, segments {len(seg)}, categories {len(cats)}")
    print("segment length median", int(seg.text.str.len().median()), "p95", int(seg.text.str.len().quantile(.95)))

if __name__ == "__main__":
    main()
