"""Pre-specified analysis of the timing-shift run (environment/timing_shift_plan.json).

Usage: analyse_timing_shift.py <run_dir> <out_dir>
Also replays every weighted-F1 score from the saved predictions.
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
CLASSES = {0: "Benign", 1: "DoS", 2: "Bot", 3: "Infiltration"}

# ---- 1. replay every weighted F1 from saved predictions ----
maxdiff, n = 0.0, 0
for _, r in m.iterrows():
    yt = np.load(run / "predictions" / f"seed{r.seed}_y_true.npz")["y_true"]
    yp = np.load(run / "predictions" / f"seed{r.seed}_{r.model}_{r.condition}_{r.variant}_{r.delay_us}.npz")["y_pred"]
    maxdiff = max(maxdiff, abs(f1_score(yt, yp, average="weighted") - r.weighted_f1))
    n += 1
replay = {"rows_replayed": n, "max_abs_weighted_f1_difference": maxdiff}

# ---- 2. zero-delay agreement with the locked record ----
raw = json.loads((Path(__file__).resolve().parents[1] / "reference/main_results/revised_pipeline_raw.json").read_text())
lock = []
for _, r in m[(m.delay_us == 0)].iterrows():
    ref = raw[str(r.seed)]["results"][r.model]
    target = {"C1": ref["C1"]["weighted_f1"], "C2": ref["per_variant"]["hybrid768"]["C2"]["weighted_f1"],
              "C3u": ref["per_variant"]["hybrid768"]["C3"]["weighted_f1"],
              "C3a": ref["per_variant"]["hybrid768"]["C3"]["weighted_f1"]}[r.condition]
    lock.append(abs(r.weighted_f1 - target))
replay["zero_delay_rows"] = len(lock)
replay["zero_delay_max_abs_difference_vs_locked"] = float(max(lock))
(out / "replay.json").write_text(json.dumps(replay, indent=2))


def series(cond, var, d, model, col="weighted_f1"):
    s = m[(m.condition == cond) & (m.variant == var) & (m.delay_us == d) & (m.model == model)].sort_values("seed")
    assert len(s) == K, (cond, var, d, model, len(s))
    return s[col].to_numpy()


def corrected(dvals, margin):
    d = np.asarray(dvals, float)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd * math.sqrt(1 / K + RATIO)
    q = stats.t.ppf(.95, K - 1)
    lo_p = stats.t.sf((mean + margin) / se, K - 1)
    hi_p = stats.t.cdf((mean - margin) / se, K - 1)
    p_two = 2 * stats.t.sf(abs(mean / se), K - 1) if se > 0 else 0.0
    return mean - q * se, mean + q * se, max(lo_p, hi_p), p_two


rows = []
c1 = {mo: series("C1", "none", 0, mo) for mo in MODELS}
for cond, variants in [("C3u", ["additive_max", "floor_max"]), ("C3a", ["additive_max"]), ("C2", ["additive_max", "floor_max"])]:
    for var in variants:
        for d in sorted(m.delay_us.unique()):
            block = []
            for mo in MODELS:
                diff = series(cond, var, d, mo) - c1[mo]
                r = {"condition": cond, "variant": var, "delay_us": int(d), "delay_ms": d / 1000, "model": mo,
                     "C1": c1[mo].mean(), "score": series(cond, var, d, mo).mean(),
                     "n_features": int(series(cond, var, d, mo, "n_features")[0]),
                     "features_same_all_splits": bool(len(set(series(cond, var, d, mo, "n_features"))) == 1)}
                if np.std(diff) == 0:
                    r.update({"mean_difference": float(diff.mean()), "degenerate": True})
                    block.append(r)
                    continue
                s = paired_summary(diff)
                r.update({k: s[k] for k in ("mean_difference", "ci90_lower", "ci90_upper", "ci95_lower", "ci95_upper",
                                             "minimum_difference", "maximum_difference", "selected_test", "selected_p")})
                r["splits_below_C1"] = int((diff < 0).sum())
                for margin in (.01, .005):
                    r[f"tost_raw_{margin}"] = tost(s, margin)["raw_p"]
                    lo, hi, pc, p2 = corrected(diff, margin)
                    r[f"corr_ci90_lower"], r[f"corr_ci90_upper"] = lo, hi
                    r[f"corr_tost_raw_{margin}"] = pc
                    r["corr_p_two_sided"] = p2
                block.append(r)
            live = [b for b in block if not b.get("degenerate")]
            if len(live) == 3:
                for key in ["selected_p", "tost_raw_0.01", "tost_raw_0.005", "corr_tost_raw_0.01", "corr_tost_raw_0.005", "corr_p_two_sided"]:
                    adj = holm([b[key] for b in live])
                    for b, a in zip(live, adj):
                        b[key.replace("raw_", "") + "_holm" if "raw" in key else key + "_holm"] = float(a)
                for b in live:
                    b["equiv_0.01"] = (b["tost_0.01_holm"] < .05) and b["ci90_lower"] > -.01 and b["ci90_upper"] < .01
                    b["equiv_0.005"] = (b["tost_0.005_holm"] < .05) and b["ci90_lower"] > -.005 and b["ci90_upper"] < .005
                    b["corr_equiv_0.01"] = b["corr_tost_0.01_holm"] < .05
                    b["corr_equiv_0.005"] = b["corr_tost_0.005_holm"] < .05
            rows += block
res = pd.DataFrame(rows)
res.to_csv(out / "timing_shift_statistics.csv", index=False)

# ---- 3. tolerance summary (primary: C3u, additive_max, recorded procedure, +/-0.01) ----
tol = {}
for cond, var in [("C3u", "additive_max"), ("C3u", "floor_max"), ("C3a", "additive_max")]:
    ok = []
    for d in sorted(res.delay_us.unique()):
        sub = res[(res.condition == cond) & (res.variant == var) & (res.delay_us == d)]
        ok.append((int(d), bool(sub["equiv_0.01"].fillna(False).all())))
    passing = [d for d, good in ok if good]
    first_fail = next((d for d, good in ok if not good), None)
    tol[f"{cond}_{var}"] = {"all_three_equivalent_by_delay": ok,
                            "largest_delay_all_equivalent_below_first_failure":
                                max([d for d in passing if first_fail is None or d < first_fail], default=None),
                            "first_delay_failing": first_fail}
(out / "tolerance_summary.json").write_text(json.dumps(tol, indent=2))

# ---- 4. per-class F1 (descriptive) ----
pc = []
for cond, var in [("C1", "none"), ("C2", "additive_max"), ("C3u", "additive_max"), ("C3a", "additive_max")]:
    for d in (sorted(m.delay_us.unique()) if cond != "C1" else [0]):
        for mo in MODELS:
            r = {"condition": cond, "variant": var, "delay_us": int(d), "model": mo}
            for code, name in CLASSES.items():
                r[name] = series(cond, var, d, mo, f"class{code}_f1").mean()
            pc.append(r)
pd.DataFrame(pc).to_csv(out / "per_class_f1.csv", index=False)

# ---- 5. anticipated selections ----
sel = json.loads((run / "anticipated_selection.json").read_text())
main = {s["seed"]: s for s in sel if s["delay_us"] == 0}
srows = []
for s in sel:
    srows.append({"seed": s["seed"], "delay_us": s["delay_us"], "n_stable": len(s["stable"]), "n_shifted": len(s["shifted"]),
                  "n_selected": len(s["selected"]),
                  "newly_shifted": ";".join(sorted(set(s["shifted"]) - set(main[s["seed"]]["shifted"]))),
                  "main_selected_dropped": ";".join(f for f in main[s["seed"]]["selected"] if f not in s["selected"]),
                  "newly_selected": ";".join(f for f in s["selected"] if f not in main[s["seed"]]["selected"])})
pd.DataFrame(srows).to_csv(out / "anticipated_selection_summary.csv", index=False)
print(json.dumps(replay, indent=2))
print(json.dumps(tol, indent=2))
