"""Simulation-augmented baseline versus the stable-feature pipeline (environment/augmentation_baseline_plan.json).

Per split and model: M_full (the paper's C1/C2 model), C3 (the paper's selection) and A (70 features, trained on
the original training partition stacked with its hybrid768-transformed copy). All three are evaluated on the
original, well-specified, footprint, packet-size, partial-adoption and timing-delay test sets.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, sha256, verify_reference_code, write_csv, write_json
from seed0_reproduction import array_hash, check_environment, now, text_hash
from run_operator_sensitivity import check_historical_selection, load_verified_data
from run_timing_shift import X_SHA, Y_SHA, timing_operator

PLAN = ROOT / "environment/augmentation_baseline_plan.json"
FOOTPRINTS = ("hybrid512", "pure768", "hybrid768_p2")
PACKET_SIZES = (1200, 1330)
ADOPTION = (0.25, 0.5, 0.75)
DELAYS = (100, 1000, 10000, 100000)


def partial_adoption(Xte, Xte_full, yte, level, seed):
    """Same row choice as reference/locked_v5/revised_partial_adoption_a1b2.make_partial_adoption_test."""
    mixed = Xte.copy()
    rng = np.random.RandomState(5000 + seed + int(level * 1000))
    chosen = []
    for cls in np.unique(yte):
        positions = np.where(yte == cls)[0]
        n_pick = int(round(len(positions) * level))
        if n_pick > 0:
            chosen.extend(rng.choice(positions, size=n_pick, replace=False).tolist())
    chosen = sorted(chosen)
    mixed.iloc[chosen, :] = Xte_full.iloc[chosen, :].values
    return mixed


def run(args):
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "predictions").mkdir()
    status = {"status": "checking", "started_utc": now(), "plan_sha256": sha256(PLAN), "fits": 0,
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
            frame = pd.read_pickle(args.matrix)
            features = json.loads((ROOT / "reference/grouped_evaluation_20260923/results/prepared_data.json").read_text())["features"]
            X, y = frame[features].copy(), frame["label_id"].to_numpy()
            status["data_route"] = "provenance matrix"
        else:
            X, y, features, _ = load_verified_data(args, pipeline, out_dir)
            status["data_route"] = "original CSV files"
        if array_hash(X.values) != X_SHA or array_hash(y) != Y_SHA:
            raise ValueError("Prepared matrix identity differs from the verified record.")

        main_cfg = pipeline.VARIANTS["hybrid768"]
        if main_cfg != pipeline.ob.MLKEM:
            raise ValueError("Main footprint differs from the partial-adoption default configuration.")
        shifted = {"well_specified": pipeline.ob.make_approx_mlkem(X, main_cfg)}
        for v in FOOTPRINTS:
            shifted[v] = pipeline.ob.make_approx_mlkem(X, pipeline.VARIANTS[v])
        for size in PACKET_SIZES:
            shifted[f"packet_{size}"] = pipeline.ob.make_approx_mlkem(X, dict(main_cfg, added_pkt_size=float(size)))
        for d in DELAYS:
            shifted[f"delay_{d}us"] = timing_operator(X, shifted["well_specified"], d, "additive_max")
        write_json(out_dir / "operator_hashes.json", {k: array_hash(v.values) for k, v in shifted.items()})

        raw = json.loads((ROOT / "reference/main_results/revised_pipeline_raw.json").read_text())
        packet_ref = pd.read_csv(ROOT / "reference/packet_retrain_sources_20260923/packet_size_sensitivity/full_10_splits/packet_size_metrics_split_values.csv")
        partial_ref = json.loads((ROOT / "reference/exposure_sources_20260923/revised_partial_adoption_a1b2_raw.json").read_text())["raw"]
        candidates = [f for f in features if f not in pipeline.UNMODELLABLE]
        classes_expected = sorted(set(y.tolist()))
        metrics, fits, checks, locked = [], [], [], []
        seeds = args.seeds if args.seeds else list(range(10))

        def fit(kind, cols, train, ytrain, seed, role):
            scaler = StandardScaler().fit(train[cols]) if kind == "mlp" else None
            values = scaler.transform(train[cols]) if scaler is not None else train[cols].values
            est = pipeline.full.make_model(kind)
            est.fit(values, ytrain)
            if kind == "rf":
                est.set_params(n_jobs=1)
            fits.append({"seed": seed, "model": kind, "role": role, "n_train": len(train), "n_features": len(cols),
                         "feature_order_sha256": text_hash(cols)})
            status["fits"] = len(fits)
            write_json(out_dir / "fit_records.json", fits)
            save()
            return est, scaler, cols

        def evaluate(fitted, frame, ytest, seed, kind, role, test_set):
            est, scaler, cols = fitted
            values = scaler.transform(frame[cols]) if scaler is not None else frame[cols].values
            prob = est.predict_proba(values)
            cls = est.classes_
            if not np.isfinite(prob).all() or sorted(cls.tolist()) != classes_expected:
                raise ValueError("Invalid probabilities or classes.")
            pred = cls[np.argmax(prob, axis=1)]
            row = {"seed": seed, "model": kind, "fit": role, "test_set": test_set, "n_features": len(cols),
                   "weighted_f1": float(f1_score(ytest, pred, average="weighted")),
                   "macro_f1": float(f1_score(ytest, pred, average="macro")),
                   "roc_auc_ovr_weighted": float(roc_auc_score(ytest, prob, labels=cls, multi_class="ovr", average="weighted")),
                   "prediction_sha256": array_hash(pred)}
            for code, f in zip(cls.tolist(), f1_score(ytest, pred, labels=cls, average=None)):
                row[f"class{code}_f1"] = float(f)
            np.savez_compressed(out_dir / "predictions" / f"seed{seed}_{kind}_{role}_{test_set}.npz", y_pred=pred.astype(np.int8))
            metrics.append(row)
            return row["weighted_f1"]

        def check(seed, kind, what, got, expected, tol):
            diff = abs(got - expected)
            locked.append({"seed": seed, "model": kind, "check": what, "difference": diff})
            write_json(out_dir / "locked_checks.json", locked)
            limit = 5e-3 if kind == "xgb" else tol
            if diff > limit:
                raise ValueError(f"Integrity check failed: seed {seed} {kind} {what} difference {diff}")

        for seed in seeds:
            status["current_seed"] = seed
            save()
            tr, te = train_test_split(np.arange(len(X)), test_size=.3, random_state=seed, stratify=y)
            train, ytrain, ytest = X.iloc[tr], y[tr], y[te]
            np.savez_compressed(out_dir / "predictions" / f"seed{seed}_y_true.npz", y_true=ytest, test_row_positions=te)
            stable, shifted_list, D = pipeline.ks_holm_stable(train[candidates], shifted["well_specified"].iloc[tr][candidates], candidates, .05)
            mi = pipeline.mi_rank(train, stable, ytrain, seed)
            selected = [f for f in stable if mi.get(f, 0) >= .05] or stable[:1]
            result = check_historical_selection(seed, candidates, stable, shifted_list, D, mi, selected, raw[str(seed)])
            checks.append(result)
            write_json(out_dir / "main_selection_checks.json", checks)
            if not result["passed"]:
                raise ValueError(f"Main selection differs from the locked record in split {seed}.")
            aug_train = pd.concat([train, shifted["well_specified"].iloc[tr]], axis=0)
            aug_y = np.concatenate([ytrain, ytrain])
            tests = {"original": X.iloc[te]}
            for name, matrix in shifted.items():
                tests[name] = matrix.iloc[te]
            for level in ADOPTION:
                tests[f"adoption_{int(level * 100)}"] = partial_adoption(X.iloc[te], shifted["well_specified"].iloc[te], ytest, level, seed)
            for kind in ("rf", "xgb", "mlp"):
                ref = raw[str(seed)]["results"][kind]
                models = {"M_full": fit(kind, features, train, ytrain, seed, "M_full"),
                          "C3": fit(kind, selected, train, ytrain, seed, "C3"),
                          "A": fit(kind, features, aug_train, aug_y, seed, "A_augmented")}
                for role, fitted in models.items():
                    for name, frame in tests.items():
                        score = evaluate(fitted, frame, ytest, seed, kind, role, name)
                        if role == "M_full" and name == "original":
                            check(seed, kind, "C1", score, ref["C1"]["weighted_f1"], 1e-12)
                        if role in ("M_full", "C3") and name in ("well_specified",) + FOOTPRINTS:
                            v = "hybrid768" if name == "well_specified" else name
                            cond = "C2" if role == "M_full" else "C3"
                            check(seed, kind, f"{role}_{name}", score, ref["per_variant"][v][cond]["weighted_f1"], 1e-12)
                        if role == "M_full" and name.startswith("packet_"):
                            size = int(name.split("_")[1])
                            r = packet_ref[(packet_ref.seed == seed) & (packet_ref.packet_size == size)
                                           & (packet_ref.model_key == kind) & (packet_ref.condition == "C2")]
                            check(seed, kind, f"M_full_{name}", score, float(r.weighted_f1.iloc[0]), 1e-10)
                        if role == "M_full" and name.startswith("adoption_"):
                            level = str(int(name.split("_")[1]) / 100)
                            exp = partial_ref[str(seed)][kind]["partial"][level]["full_feature_model"]["weighted_f1"]
                            check(seed, kind, f"M_full_{name}", score, exp, 1e-12)
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
    p.add_argument("--matrix", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seeds", type=int, nargs="*")
    a = p.parse_args()
    if (a.matrix is None) == (a.data_dir is None) or (a.data_dir is not None and a.cache_dir is None):
        p.error("give either --matrix, or --data-dir with --cache-dir")
    sys.exit(run(a))
