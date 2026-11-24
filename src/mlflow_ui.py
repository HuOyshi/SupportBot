"""
Open the MLflow UI on the runs stored in ./mlruns (works on Windows, macOS and Linux).

    python src/mlflow_ui.py            ->  http://localhost:5000
    python src/mlflow_ui.py 5001       ->  other port

MLflow stores ABSOLUTE artifact paths in mlruns/**/meta.yaml. After the repository is cloned
to another folder or computer those paths are wrong and the plots would not show. This script
first rewrites them to the current location of mlruns/, then starts the UI.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLRUNS = ROOT / "mlruns"
PATH_RE = re.compile(r"^(artifact_(?:location|uri):\s*)file:\S*?/mlruns/", re.MULTILINE)


def fix_artifact_paths():
    prefix = MLRUNS.as_uri() + "/"
    changed = 0
    for meta in MLRUNS.rglob("meta.yaml"):
        text = meta.read_text(encoding="utf-8")
        new = PATH_RE.sub(lambda m: m.group(1) + prefix, text)
        if new != text:
            meta.write_text(new, encoding="utf-8")
            changed += 1
    if changed:
        print(f"Updated artifact paths in {changed} MLflow metadata files.")


if __name__ == "__main__":
    fix_artifact_paths()
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"   # newer MLflow needs this for the ./mlruns folder store
    port = sys.argv[1] if len(sys.argv) > 1 else "5000"
    print(f"MLflow UI: http://localhost:{port}   (Ctrl+C to stop)")
    subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", MLRUNS.as_uri(),
                    "--host", "127.0.0.1", "--port", port], cwd=ROOT)
