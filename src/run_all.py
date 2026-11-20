import subprocess, sys, pathlib
here = pathlib.Path(__file__).parent
for s in ["00_download", "01_load", "02_embed", "03_audit", "04_models", "05_report"]:
    print("running", s, flush=True)
    subprocess.run([sys.executable, "-I", str(here / f"{s}.py")], check=True)
