"""Embed every segment with a small open sentence model (all-MiniLM-L6-v2, CPU)."""
import pathlib, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
ROOT = pathlib.Path(__file__).resolve().parents[1]
seg = pd.read_parquet(ROOT / "data/segments.parquet", columns=["text"])
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
m.max_seq_length = 256
X = m.encode(seg.text.tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
np.save(ROOT / "data/emb_minilm.npy", X.astype("float32"))
print("embedded", X.shape)
