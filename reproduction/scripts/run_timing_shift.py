"""Unanticipated timing shift (review item 4.2), following environment/timing_shift_plan.json.

The base transformation is the locked historical operator. A declared timing
operator T_d then adds an assumed handshake delay d to eligible flows. C3u uses
the paper's selection (screen against the base operator only); C3a screens
against the base operator plus T_d. Zero delay must reproduce the locked main
record exactly before any other result is used.
"""
import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, sha256, verify_reference_code, write_csv, write_json
from seed0_reproduction import array_hash, check_environment, now, text_hash
from run_operator_sensitivity import check_historical_selection, load_verified_data

PLAN = ROOT / "environment/timing_shift_plan.json"
X_SHA = "b852fe0225df0c7b3a3407c3d2c98bccfa2a6252b144f3e2e6d6ba5549df5e6e"
Y_SHA = "e1f328443429dcdfa8c521b1296c72c0c14ad3c17d17543065b4f81c6ab9d15a"
TIMING = ["Flow Duration", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
          "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Max", "Fwd IAT Min",
          "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Max", "Bwd IAT Min"]
RATES = {"Flow Byts/s": ("TotLen Fwd Pkts", "TotLen Bwd Pkts"), "Flow Pkts/s": ("Tot Fwd Pkts", "Tot Bwd Pkts"),
         "Fwd Pkts/s": ("Tot Fwd Pkts",), "Bwd Pkts/s": ("Tot Bwd Pkts",)}
VARIANTS = ("additive_max", "floor_max")


def timing_operator(X, B, d, variant):
    """Apply T_d to the base-operator output B; X is the original matrix."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    if d == 0:
        return B
    out = B.copy()
    if not np.array_equal(B[TIMING].to_numpy(), X[TIMING].to_numpy()):
        raise ValueError("Base operator unexpectedly changed a timing column.")
    v = lambda frame, c: frame[c].to_numpy(dtype=np.float64)
    elig = ((X["Tot Fwd Pkts"] > 0) & (X["Tot Bwd Pkts"] > 0)).to_numpy()
    nf, nb = v(X, "Tot Fwd Pkts"), v(X, "Tot Bwd Pkts")
    N = nf + nb
    FD = v(X, "Flow Duration")
    new = {}

    def grow_max(old):
        return old + d if variant == "additive_max" else np.maximum(old, d)

    new["Flow Duration"] = np.where(elig, FD + d, FD)
    gaps = N - 1
    new["Flow IAT Mean"] = np.where(elig, v(X, "Flow IAT Mean") + d / np.where(gaps > 0, gaps, 1), v(X, "Flow IAT Mean"))
    new["Flow IAT Max"] = np.where(elig, grow_max(v(X, "Flow IAT Max")), v(X, "Flow IAT Max"))
    new["Flow IAT Min"] = np.where(elig & (N == 2), v(X, "Flow IAT Min") + d, v(X, "Flow IAT Min"))
    for prefix, n in (("Fwd", nf), ("Bwd", nb)):
        cond = elig & (n >= 2)
        g = np.where(n - 1 > 0, n - 1, 1)
        new[f"{prefix} IAT Tot"] = np.where(cond, v(X, f"{prefix} IAT Tot") + d, v(X, f"{prefix} IAT Tot"))
        new[f"{prefix} IAT Mean"] = np.where(cond, v(X, f"{prefix} IAT Mean") + d / g, v(X, f"{prefix} IAT Mean"))
        new[f"{prefix} IAT Max"] = np.where(cond, grow_max(v(X, f"{prefix} IAT Max")), v(X, f"{prefix} IAT Max"))
        new[f"{prefix} IAT Min"] = np.where(elig & (n == 2), v(X, f"{prefix} IAT Min") + d, v(X, f"{prefix} IAT Min"))
    for rate, counts in RATES.items():
        base_rate = v(B, rate)
        count = sum(v(B, c) for c in counts)
        scaled = base_rate * FD / np.where(FD > 0, FD + d, 1)
        recomputed = count * 1e6 / d
        new[rate] = np.where(elig & (FD > 0), scaled, np.where(elig, recomputed, base_rate))
    for c, values in new.items():
        out[c] = values.astype(B[c].dtype)
    declared = set(new)
    untouched = [c for c in X.columns if c not in declared]
    if not np.array_equal(out[untouched].to_numpy(), B[untouched].to_numpy()):
        raise ValueError("Timing operator changed an undeclared column.")
    if not np.array_equal(out.to_numpy()[~elig], B.to_numpy()[~elig]):
        raise ValueError("Timing operator changed an ineligible row.")
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Nonfinite value after the timing operator.")
    return out


def run(args):
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "predictions").mkdir()
    plan = json.loads(PLAN.read_text())
    status = {"status": "checking", "started_utc": now(), "plan_sha256": sha256(PLAN), "fits": 0,
              "amendment_1_sha256": sha256(ROOT / "environment/timing_shift_plan_amendment_1.md"),
              "OPENBLAS_CORETYPE": __import__("os").environ.get("OPENBLAS_CORETYPE")}
    save = lambda: write_json(out_dir / "status.json", status)
    save()
    try:
        env = check_environment(json.loads((ROOT / "environment/main_known_versions.json").read_text()))
        write_json(out_dir / "runtime_environment.json", env)
        if env["problems"]:
            raise ValueError("Environment mismatch: " + "; ".join(env["problems"]))
        status["reference_sources_verified"] = verify_reference_code()
        sys.path.insert(0, str(ROOT / "reference/locked_v5"))
        import revised_pipeline_main_only as pipeline
        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        if args.matrix is not None:
            # Workspace route: the verified provenance-tagged matrix.
            frame = pd.read_pickle(args.matrix)
            features = json.loads((ROOT / "reference/grouped_evaluation_20260923/results/prepared_data.json").read_text())["features"]
            X = frame[features].copy()
            y = frame["label_id"].to_numpy()
            status["data_route"] = "provenance matrix"
        else:
            # Colab route: rebuild the matrix from the four original CSV files (hash-checked).
            X, y, features, _ = load_verified_data(args, pipeline, out_dir)
            status["data_route"] = "original CSV files"
        if array_hash(X.values) != X_SHA or array_hash(y) != Y_SHA:
            raise ValueError("Prepared matrix identity differs from the verified record.")
        cfg = pipeline.VARIANTS["hybrid768"]
        base = pipeline.ob.make_approx_mlkem(X, cfg)
        delays = plan["operator"]["delay_grid_microseconds"]
        mats = {(var, d): timing_operator(X, base, d, var) for var in VARIANTS for d in delays}
        write_json(out_dir / "operator_checks.json", [
            {"variant": var, "delay_us": d, "sha256": array_hash(m.values),
             "changed_rows_vs_base": int(np.any(m.to_numpy() != base.to_numpy(), axis=1).sum())}
            for (var, d), m in mats.items()])
        reference = json.loads((ROOT / "reference/main_results/revised_pipeline_raw.json").read_text())
        candidates = [f for f in features if f not in pipeline.UNMODELLABLE]
        classes_expected = sorted(set(y.tolist()))
        metrics, selections, checks, fits, locked_diffs = [], [], [], [], []
        seeds = args.seeds if args.seeds else list(range(10))

        def fit(kind, cols, train, ytrain, seed, role):
            scaler = StandardScaler().fit(train[cols]) if kind == "mlp" else None
            values = scaler.transform(train[cols]) if scaler is not None else train[cols].values
            est = pipeline.full.make_model(kind)
            est.fit(values, ytrain)
            if kind == "rf":
                est.set_params(n_jobs=1)
            fits.append({"seed": seed, "model": kind, "role": role, "n_features": len(cols),
                         "feature_order_sha256": text_hash(cols)})
            status["fits"] = len(fits)
            write_json(out_dir / "fit_records.json", fits)
            save()
            return est, scaler, cols

        def evaluate(fitted, test_frame, ytest, seed, kind, condition, variant, d):
            est, scaler, cols = fitted
            values = scaler.transform(test_frame[cols]) if scaler is not None else test_frame[cols].values
            prob = est.predict_proba(values)
            cls = est.classes_
            if not np.isfinite(prob).all() or sorted(cls.tolist()) != classes_expected:
                raise ValueError("Invalid probabilities or classes.")
            pred = cls[np.argmax(prob, axis=1)]
            row = {"seed": seed, "model": kind, "condition": condition, "variant": variant, "delay_us": d,
                   "n_features": len(cols), "weighted_f1": float(f1_score(ytest, pred, average="weighted")),
                   "macro_f1": float(f1_score(ytest, pred, average="macro")),
                   "roc_auc_ovr_weighted": float(roc_auc_score(ytest, prob, labels=cls, multi_class="ovr", average="weighted")),
                   "prediction_sha256": array_hash(pred), "probability_sha256": array_hash(prob)}
            per = f1_score(ytest, pred, labels=cls, average=None)
            for code, f in zip(cls.tolist(), per):
                row[f"class{code}_f1"] = float(f)
            name = f"seed{seed}_{kind}_{condition}_{variant}_{d}.npz"
            np.savez_compressed(out_dir / "predictions" / name, y_pred=pred.astype(np.int8))
            metrics.append(row)
            return row

        for seed in seeds:
            status["current_seed"] = seed
            save()
            tr, te = train_test_split(np.arange(len(X)), test_size=.3, random_state=seed, stratify=y)
            train, test, ytrain, ytest = X.iloc[tr], X.iloc[te], y[tr], y[te]
            np.savez_compressed(out_dir / "predictions" / f"seed{seed}_y_true.npz", y_true=ytest, test_row_positions=te)
            mi_cache = {}

            def select(shifted_train):
                stable, shifted, D = pipeline.ks_holm_stable(train[candidates], shifted_train[candidates], candidates, .05)
                key = tuple(stable)
                if key not in mi_cache:
                    mi_cache[key] = pipeline.mi_rank(train, stable, ytrain, seed)
                mi = mi_cache[key]
                selected = [f for f in stable if mi.get(f, 0) >= .05] or stable[:1]
                return stable, shifted, D, mi, selected

            main = select(base.iloc[tr])
            result = check_historical_selection(seed, candidates, main[0], main[1], main[2], main[3], main[4], reference[str(seed)])
            checks.append(result)
            write_json(out_dir / "main_selection_checks.json", checks)
            if not result["passed"]:
                raise ValueError(f"Main selection differs from the locked record in split {seed}.")
            anticipated = {}
            for d in delays:
                s = select(mats[("additive_max", d)].iloc[tr]) if d else main
                anticipated[d] = s[4]
                selections.append({"seed": seed, "delay_us": d, "stable": s[0], "shifted": s[1], "KS_D": s[2],
                                   "MI": s[3], "selected": s[4]})
            write_json(out_dir / "anticipated_selection.json", selections)
            if anticipated[0] != main[4]:
                raise ValueError("Zero-delay anticipated selection differs from the main selection.")
            for kind in ("rf", "xgb", "mlp"):
                baseline = fit(kind, features, train, ytrain, seed, "C1_and_C2")
                c1 = evaluate(baseline, test, ytest, seed, kind, "C1", "none", 0)
                c3_main = fit(kind, main[4], train, ytrain, seed, "C3_main_selection")
                ref = reference[str(seed)]["results"][kind]
                for var in VARIANTS:
                    for d in delays:
                        m = mats[(var, d)].iloc[te]
                        c2 = evaluate(baseline, m, ytest, seed, kind, "C2", var, d)
                        c3u = evaluate(c3_main, m, ytest, seed, kind, "C3u", var, d)
                        if d == 0:
                            diffs = [abs(c1["weighted_f1"] - ref["C1"]["weighted_f1"]),
                                     abs(c2["weighted_f1"] - ref["per_variant"]["hybrid768"]["C2"]["weighted_f1"]),
                                     abs(c3u["weighted_f1"] - ref["per_variant"]["hybrid768"]["C3"]["weighted_f1"])]
                            locked_diffs.append({"seed": seed, "model": kind, "variant": var,
                                                 "C1": diffs[0], "C2": diffs[1], "C3": diffs[2]})
                            write_json(out_dir / "zero_delay_vs_locked.json", locked_diffs)
                            limit = 5e-3 if kind == "xgb" else 1e-12  # amendment 1
                            if max(diffs) > limit:
                                raise ValueError(f"Zero-delay scores differ from the locked record: seed {seed} {kind} {diffs}")
                refits = {tuple(main[4]): c3_main}
                for d in delays:
                    key = tuple(anticipated[d])
                    if key not in refits:
                        refits[key] = fit(kind, list(key), train, ytrain, seed, f"C3a_delay_{d}")
                    evaluate(refits[key], mats[("additive_max", d)].iloc[te], ytest, seed, kind, "C3a", "additive_max", d)
                write_csv(out_dir / "metrics.csv", metrics)
                print(f"{now()} seed {seed} {kind} done; fits {len(fits)}", flush=True)
            status.setdefault("completed_seeds", []).append(seed)
            save()
        if array_hash(X.values) != X_SHA:
            raise ValueError("Matrix changed during the run.")
        status["status"] = "completed"
    except Exception as error:
        status["status"] = "failed"
        status["error"] = str(error)
        (out_dir / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        status["finished_utc"] = now()
        save()
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, help="verified provenance matrix (.pkl.gz)")
    p.add_argument("--data-dir", type=Path, help="folder with the four original CSV files")
    p.add_argument("--cache-dir", type=Path, help="new local folder for copies of the CSV files")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seeds", type=int, nargs="*")
    a = p.parse_args()
    if (a.matrix is None) == (a.data_dir is None) or (a.data_dir is not None and a.cache_dir is None):
        p.error("give either --matrix, or --data-dir with --cache-dir")
    sys.exit(run(a))
