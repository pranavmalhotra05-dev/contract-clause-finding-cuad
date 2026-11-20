"""Download CUAD v1 (CC BY 4.0) from Zenodo and unpack the JSON."""
import pathlib, urllib.request, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
raw = ROOT / "data_raw"; raw.mkdir(exist_ok=True)
zp = raw / "CUAD_v1.zip"
if not zp.exists():
    urllib.request.urlretrieve("https://zenodo.org/api/records/4595826/files/CUAD_v1.zip/content", zp)
with zipfile.ZipFile(zp) as z:
    for n in ("CUAD_v1/CUAD_v1.json", "CUAD_v1/CUAD_v1_README.txt"):
        z.extract(n, raw / "cuad")
print("ready")
