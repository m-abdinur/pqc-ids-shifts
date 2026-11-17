"""Rerun split 0 of the main design and compare it with run A.

This is a short check (42 fits, about 6 minutes on 4 CPUs). It does not change
any locked script. It:

1. checks the SHA-256 of PQC_IDS_Reproduction_Working_Copy.zip,
2. unpacks the ZIP into a new scratch folder and checks every file against
   the ZIP's package_manifest.json (as the September notebooks do),
3. runs the unchanged scripts/run_experiment.py from the unpacked copy with
   the arguments the main notebook uses. Only the split seeds of the locked
   pipeline are set to [0] in memory, so split 0 alone is fitted,
4. compares split 0 with the stored run A results
   (reference/main_results/revised_pipeline_raw.json) using compare_tree from
   scripts/compare_main_reproduction.py and the tolerances in
   environment/main_known_versions.json.

Run it with the Python of the environment built from requirements.txt:

    .venv/bin/python reproduction/check_split0.py --data-dir cicids2018

The exit code is 0 when every field is within its tolerance, 1 otherwise.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
WORKING_COPY = HERE / "PQC_IDS_Reproduction_Working_Copy.zip"
WORKING_COPY_SHA256 = "ca03febf6f4cc5c76131ff57fb25feba2745f58445a49c26b935ca05e6d607a7"

# Runs inside the fitting interpreter: set the locked pipeline to split seed 0,
# then call the unchanged run_experiment.main() with the remaining arguments.
RUNNER = """
import sys
package = sys.argv.pop(1)
sys.path.insert(0, package + "/scripts")
sys.path.insert(0, package + "/reference/locked_v5")
import revised_pipeline_main_only as pipeline
pipeline.SPLIT_SEEDS = [0]
pipeline.N_SPLITS = 1
import run_experiment
sys.exit(run_experiment.main())
"""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack(workdir):
    print("Checking", WORKING_COPY.name, flush=True)
    if sha256(WORKING_COPY) != WORKING_COPY_SHA256:
        raise SystemExit("The working copy ZIP does not have its recorded SHA-256. Do not repack it.")
    target = workdir / "package"
    target.mkdir()
    with zipfile.ZipFile(WORKING_COPY) as archive:
        for name in archive.namelist():
            member = (target / name).resolve()
            if not member.is_relative_to(target.resolve()):
                raise SystemExit(f"Unsafe path in the ZIP: {name}")
        archive.extractall(target)
    manifests = list(target.glob("*/package_manifest.json"))
    if len(manifests) != 1:
        raise SystemExit("Expected one package_manifest.json in the working copy.")
    package = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    bad = [item["path"] for item in manifest["files"]
           if sha256(package / item["path"]) != item["sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} unpacked files differ from the manifest, e.g. {bad[0]}")
    print(f"Unpacked and checked {len(manifest['files'])} files in {package}", flush=True)
    return package


def run_split0(package, data_dir, workdir, python):
    seed0 = package / "reference/seed0_reproduction_20260920/results"
    output = workdir / "split0"
    command = [python, "-u", "-c", RUNNER, str(package),
               "--stage", "train", "--variant", "main",
               "--data-dir", str(data_dir), "--output", str(output),
               "--expected-environment", str(package / "environment/main_known_versions.json"),
               "--expected-inputs", str(seed0 / "input_files.json"),
               "--expected-prepared", str(seed0 / "seed0_status.json")]
    started = time.time()
    with (workdir / "split0_run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=workdir, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            if line.startswith(("Fit ", "Input verified", "  split")):
                print(line, end="", flush=True)
        returncode = process.wait()
    print(f"Run finished with exit code {returncode} after {time.time() - started:.0f} s", flush=True)
    if returncode != 0:
        raise SystemExit(f"The run failed. See {workdir / 'split0_run.log'}")
    return output / "revised_pipeline_raw.json"


def compare(package, raw_path, workdir):
    sys.path.insert(0, str(package / "scripts"))
    from compare_main_reproduction import compare_tree
    reference = json.loads((package / "reference/main_results/revised_pipeline_raw.json").read_text())
    rules = json.loads((package / "environment/main_known_versions.json").read_text())["acceptance"]
    actual = json.loads(raw_path.read_text())
    if set(actual) != {"0"}:
        raise SystemExit(f"Expected split 0 only, found {sorted(actual)}")
    rows = sorted(compare_tree(actual["0"], reference["0"], rules, ("0",)), key=lambda r: r["field"])
    with (workdir / "split0_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failures = [r for r in rows if not r["passed"]]
    print(f"\nSplit 0 against run A: {len(rows)} fields compared, "
          f"{len(rows) - len(failures)} within tolerance, {len(failures)} outside.")
    print("Tolerances:", json.dumps({k: v for k, v in rules.items() if k.endswith("tolerance")}))
    for row in failures:
        print(f"  {row['field']}: new {row['actual']}, stored {row['reference']}, "
              f"difference {row['difference']}")
    print("Full comparison:", workdir / "split0_comparison.csv")
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with the four CSE-CIC-IDS2018 CSV files")
    parser.add_argument("--workdir", type=Path,
                        help="New scratch folder (default: a new temporary folder)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python used for fitting (default: this Python)")
    args = parser.parse_args()
    if args.workdir:
        workdir = args.workdir.expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=False)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="pqc_split0_"))
    print("Scratch folder:", workdir, flush=True)
    package = unpack(workdir)
    raw_path = run_split0(package, args.data_dir.expanduser().resolve(), workdir, args.python)
    return compare(package, raw_path, workdir)


if __name__ == "__main__":
    sys.exit(main())
