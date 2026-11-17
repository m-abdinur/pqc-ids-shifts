"""Check the PQC-OAV-calibration folder is ready before running scripts/01_audit.py.

Run from the project folder:  python scripts/00_check_setup.py
Uses only the Python standard library.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
NEEDED = ["raw_sessions.jsonl", "ml_features_and_labels.csv", "scenario_manifest.csv"]

ok = True
print(f"Project folder: {ROOT}\n")

for name in NEEDED:
    f = RAW / name
    if not f.exists():
        print(f"[MISSING] data/raw/{name}")
        ok = False
        continue
    size_mb = f.stat().st_size / 1e6
    if name.endswith(".csv"):
        with f.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        print(f"[OK] {name}: {size_mb:.1f} MB, {rows:,} rows, {len(header)} columns")
        print(f"     first columns: {header[:8]}")
    else:
        rows, keys = 0, None
        with f.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if keys is None:
                    try:
                        keys = list(json.loads(line).keys())
                    except json.JSONDecodeError:
                        keys = ["<first line is not valid JSON>"]
                rows += 1
        print(f"[OK] {name}: {size_mb:.1f} MB, {rows:,} sessions")
        print(f"     top-level keys: {(keys or [])[:10]}")

code = [p for p in (ROOT / "my_c2_code").iterdir()
        if p.suffix in {".ipynb", ".py"}]
if code:
    print(f"\n[OK] C2 code found: {[p.name for p in code]}")
else:
    print("\n[MISSING] no .ipynb or .py file in my_c2_code/")
    ok = False

print("\nReady to run scripts/01_audit.py." if ok else "\nNot ready yet: fix the items marked MISSING.")
sys.exit(0 if ok else 1)
