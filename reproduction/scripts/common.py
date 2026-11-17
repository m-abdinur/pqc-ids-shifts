"""Small file and statistical helpers for the saved-results checks."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("rf", "xgb", "mlp")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError("There are no rows to write.")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def new_output(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def holm(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Holm correction requires finite p-values.")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def paired_summary(differences):
    """Replay the recorded procedure; this does not correct split dependence."""
    import numpy as np
    from scipy import stats
    d = np.asarray(differences, dtype=float)
    if len(d) < 3 or not np.isfinite(d).all():
        raise ValueError("At least three finite paired differences are required.")
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0:
        raise ValueError("Constant differences need an explicit degenerate-case review.")
    se = sd / math.sqrt(n)
    shapiro = stats.shapiro(d)
    t_test = stats.ttest_1samp(d, 0)
    wilcoxon = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
    selected = wilcoxon if shapiro.pvalue < .05 else t_test
    return {
        "n_splits": n, "mean_difference": mean, "difference_sd": sd,
        "difference_se": se, "median_difference": float(np.median(d)),
        "minimum_difference": float(d.min()), "maximum_difference": float(d.max()),
        "ci95_lower": float(mean - stats.t.ppf(.975, n - 1) * se),
        "ci95_upper": float(mean + stats.t.ppf(.975, n - 1) * se),
        "ci90_lower": float(mean - stats.t.ppf(.95, n - 1) * se),
        "ci90_upper": float(mean + stats.t.ppf(.95, n - 1) * se),
        "shapiro_statistic": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "t_statistic": float(t_test.statistic), "t_p": float(t_test.pvalue),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "selected_test": "Wilcoxon" if shapiro.pvalue < .05 else "paired t-test",
        "selected_statistic": float(selected.statistic),
        "selected_p": float(selected.pvalue), "cohen_dz": mean / sd,
    }


def tost(summary, margin):
    from scipy import stats
    mean, se = summary["mean_difference"], summary["difference_se"]
    df = summary["n_splits"] - 1
    lower_t, upper_t = (mean + margin) / se, (mean - margin) / se
    lower_p = float(stats.t.sf(lower_t, df))
    upper_p = float(stats.t.cdf(upper_t, df))
    return {"margin": margin, "lower_t": lower_t, "upper_t": upper_t,
            "lower_p": lower_p, "upper_p": upper_p, "raw_p": max(lower_p, upper_p)}


def compare_number(actual, expected, label, differences, atol=1e-10, rtol=1e-8):
    actual, expected = float(actual), float(expected)
    if not math.isfinite(actual) or not math.isfinite(expected):
        differences.append({"field": label, "problem": "non-finite value"})
    elif not math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol):
        differences.append({"field": label, "computed": actual, "stored": expected})


def verify_reference_code():
    manifest = json.loads((ROOT / "evidence/locked_v5_code_manifest.json").read_text())
    for row in manifest["files"]:
        path = ROOT / "reference/locked_v5" / Path(row["archive_member"]).name
        if sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Reference source differs from its recorded hash: {path.name}")
    return len(manifest["files"])
