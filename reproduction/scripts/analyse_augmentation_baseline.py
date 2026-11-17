"""Pre-specified analysis for environment/augmentation_baseline_plan.json.

Usage: analyse_augmentation_baseline.py <run_dir> <out_dir>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from common import holm, paired_summary, tost

run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(run / "metrics.csv")
K, RATIO = 10, 35872 / 83699
MODELS = ["rf", "xgb", "mlp"]
MIS = ["hybrid512", "pure768", "hybrid768_p2", "packet_1200", "packet_1330"]

# 1. replay every weighted F1 from the saved predictions
maxdiff = 0.0
for _, r in m.iterrows():
    yt = np.load(run / "predictions" / f"seed{r.seed}_y_true.npz")["y_true"]
    yp = np.load(run / "predictions" / f"seed{r.seed}_{r.model}_{r.fit}_{r.test_set}.npz")["y_pred"]
    maxdiff = max(maxdiff, abs(f1_score(yt, yp, average="weighted") - r.weighted_f1))
locked = json.loads((run / "locked_checks.json").read_text())
replay = {"rows_replayed": len(m), "max_abs_weighted_f1_difference": maxdiff,
          "locked_checks": len(locked), "locked_checks_max_difference": max(x["difference"] for x in locked)}
(out / "replay.json").write_text(json.dumps(replay, indent=2))


def s(fit, test_set, model):
    x = m[(m.fit == fit) & (m.test_set == test_set) & (m.model == model)].sort_values("seed")
    assert len(x) == K, (fit, test_set, model, len(x))
    return x.weighted_f1.to_numpy()


def corrected(d, margin):
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd * math.sqrt(1 / K + RATIO)
    q = stats.t.ppf(.95, K - 1)
    return (mean - q * se, mean + q * se,
            max(stats.t.sf((mean + margin) / se, K - 1), stats.t.cdf((mean - margin) / se, K - 1)),
            2 * stats.t.sf(abs(mean) / se, K - 1))


rows = []
tests = list(dict.fromkeys(m.test_set))
for ts in tests:
    for contrast in ("A-C1", "C3-C1", "C3-A"):
        block = []
        for mo in MODELS:
            c1 = s("M_full", "original", mo)
            d = {"A-C1": s("A", ts, mo) - c1, "C3-C1": s("C3", ts, mo) - c1, "C3-A": s("C3", ts, mo) - s("A", ts, mo)}[contrast]
            r = {"test_set": ts, "contrast": contrast, "model": mo, "mean_difference": float(d.mean()),
                 "A": float(s("A", ts, mo).mean()), "C3": float(s("C3", ts, mo).mean()), "C1": float(c1.mean()),
                 "M_full": float(s("M_full", ts, mo).mean())}
            if np.std(d) == 0:
                r["degenerate"] = True
                block.append(r)
                continue
            p = paired_summary(d)
            r.update({k: p[k] for k in ("ci90_lower", "ci90_upper", "ci95_lower", "ci95_upper", "selected_test", "selected_p",
                                         "minimum_difference", "maximum_difference")})
            r["splits_positive"] = int((d > 0).sum())
            for margin in (.01, .005):
                r[f"tost_raw_{margin}"] = tost(p, margin)["raw_p"]
                lo, hi, pc, p2 = corrected(d, margin)
                r["corr_ci90_lower"], r["corr_ci90_upper"], r[f"corr_tost_raw_{margin}"], r["corr_p_two"] = lo, hi, pc, p2
            block.append(r)
        live = [b for b in block if not b.get("degenerate")]
        if len(live) == 3:
            for key, name in [("selected_p", "diff_holm"), ("tost_raw_0.01", "tost_0.01_holm"), ("tost_raw_0.005", "tost_0.005_holm"),
                              ("corr_tost_raw_0.01", "corr_tost_0.01_holm"), ("corr_p_two", "corr_diff_holm")]:
                for b, a in zip(live, holm([b[key] for b in live])):
                    b[name] = float(a)
            for b in live:
                b["equiv_0.01"] = bool(b["tost_0.01_holm"] < .05 and b["ci90_lower"] > -.01 and b["ci90_upper"] < .01)
                b["equiv_0.005"] = bool(b["tost_0.005_holm"] < .05 and b["ci90_lower"] > -.005 and b["ci90_upper"] < .005)
        rows += block
res = pd.DataFrame(rows)
res.to_csv(out / "augmentation_statistics.csv", index=False)


def get(ts, contrast, mo):
    return res[(res.test_set == ts) & (res.contrast == contrast) & (res.model == mo)].iloc[0]


decision = {"C3_advantage_cases": [], "A_advantage_cases": []}
for ts in tests:
    for mo in MODELS:
        c3a, a1, c31 = get(ts, "C3-A", mo), get(ts, "A-C1", mo), get(ts, "C3-C1", mo)
        sig = not c3a.get("degenerate", False) and c3a.get("diff_holm", 1) < .05
        if ts in MIS and sig and c3a.mean_difference > 0 and bool(c31.get("equiv_0.01")) and not bool(a1.get("equiv_0.01")):
            decision["C3_advantage_cases"].append({"test_set": ts, "model": mo, "C3_minus_A": c3a.mean_difference})
        if sig and c3a.mean_difference < 0:
            decision["A_advantage_cases"].append({"test_set": ts, "model": mo, "A_minus_C3": -c3a.mean_difference})
decision["C3_advantage_claimed"] = bool(decision["C3_advantage_cases"])
(out / "decision.json").write_text(json.dumps(decision, indent=2))
print(json.dumps(replay, indent=2))
print(json.dumps(decision, indent=2))
