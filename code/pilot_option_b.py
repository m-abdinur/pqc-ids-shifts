# ================================================================
# OPTION B — model ML-KEM-768's FULL flow footprint, not just bytes
#
# Why: the byte-only pilot showed ~0 effect because detection on
# CSE-CIC-IDS2018 runs on port/window/timing features, which a byte
# shift never touches. ML-KEM-768 key material is large enough to
# exceed the packet size limit (MTU), so it realistically also adds
# PACKETS, HEADERS, and changes PACKET-LENGTH statistics. This script
# perturbs that full footprint, with a one-line physical reason for
# every feature changed (the examiner-defence). It then runs the same
# C1/C2/C3 design for Random Forest and (scaled) MLP, and optionally
# drops the identifier features the model over-relies on.
#
# Requires pilot_pqc_stable_pipeline.py in the same folder (reuses its
# loader, prepare_xy, evaluate, KS+Holm pipeline).
# ================================================================

import json, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import pilot_pqc_stable_pipeline as base   # reuse the validated base functions

warnings.filterwarnings("ignore")

# ============================ CONFIG ============================
SEED            = 42
PER_CLASS       = 30000
TEST_SIZE       = 0.30
HOLM_ALPHA      = 0.05
RF_TREES        = 200

# >>> the one decision that needs supervisor sign-off <<<
DROP_IDENTIFIERS    = False                       # set True to run the B3 variant
IDENTIFIER_FEATURES = ["Dst Port", "Init Fwd Win Byts"]

# ML-KEM-768 footprint (FIPS 203 sizes; hybrid deployment = PQC added on top
# of classical X25519). Every value below has a physical justification.
MLKEM = {
    "d_fwd_bytes":   1184.0,   # client adds ML-KEM-768 encapsulation key  (FIPS 203: 1184 B)
    "d_bwd_bytes":   1088.0,   # server adds ML-KEM-768 ciphertext         (FIPS 203: 1088 B)
    "d_fwd_pkts":       1.0,   # +1184 B over the ~1460 B MSS -> +1 forward packet
    "d_bwd_pkts":       1.0,   # +1088 B over the ~1460 B MSS -> +1 backward packet
    "hdr_bytes":       40.0,   # IPv4(20)+TCP(20) header bytes per extra packet
    "added_pkt_size": 1460.0,  # the extra handshake packets are ~full MSS
}
# Pure-ML-KEM (replacing X25519) alternative: d_fwd_bytes=1152, d_bwd_bytes=1056.

FILES = {
    "/content/drive/MyDrive/cicids2018/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv":    "DoS",
    "/content/drive/MyDrive/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv":    "Bot",
    "/content/drive/MyDrive/cicids2018/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv": "Infiltration",
    "/content/drive/MyDrive/cicids2018/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv":  "Infiltration",
}
# ================================================================


def make_approx_mlkem(X, cfg=MLKEM):
    """
    Apply the ML-KEM-768 handshake footprint to each flow. Applied only to
    BIDIRECTIONAL flows (a key exchange needs a client request AND a server
    response); one-directional flows are left unchanged. Every change below
    is recomputed consistently from the new byte/packet totals.
    """
    Xa = X.copy()
    cols = set(Xa.columns)

    def col(name):
        return Xa[name].astype("float64") if name in cols else None

    # bidirectional mask
    if {"Tot Fwd Pkts", "Tot Bwd Pkts"} <= cols:
        mask = (Xa["Tot Fwd Pkts"] > 0) & (Xa["Tot Bwd Pkts"] > 0)
    else:
        mask = pd.Series(True, index=Xa.index)
    m = mask.values

    # originals needed for recompute
    o_tfp = col("Tot Fwd Pkts");  o_tbp = col("Tot Bwd Pkts")
    o_tlf = col("TotLen Fwd Pkts"); o_tlb = col("TotLen Bwd Pkts")

    def blend(name, new_series):
        """write new_series only where mask is True, keep original elsewhere"""
        if name in cols:
            Xa[name] = np.where(m, new_series.values, Xa[name].astype("float64").values)

    # --- 1. byte totals: + key material (FIPS 203) ---
    n_tlf = (o_tlf + cfg["d_fwd_bytes"]) if o_tlf is not None else None
    n_tlb = (o_tlb + cfg["d_bwd_bytes"]) if o_tlb is not None else None
    if n_tlf is not None: blend("TotLen Fwd Pkts", n_tlf)
    if n_tlb is not None: blend("TotLen Bwd Pkts", n_tlb)
    if "Subflow Fwd Byts" in cols: blend("Subflow Fwd Byts", col("Subflow Fwd Byts") + cfg["d_fwd_bytes"])
    if "Subflow Bwd Byts" in cols: blend("Subflow Bwd Byts", col("Subflow Bwd Byts") + cfg["d_bwd_bytes"])

    # --- 2. packet counts: + extra packets (MTU overflow) ---
    n_tfp = (o_tfp + cfg["d_fwd_pkts"]) if o_tfp is not None else None
    n_tbp = (o_tbp + cfg["d_bwd_pkts"]) if o_tbp is not None else None
    if n_tfp is not None: blend("Tot Fwd Pkts", n_tfp)
    if n_tbp is not None: blend("Tot Bwd Pkts", n_tbp)
    if "Subflow Fwd Pkts" in cols: blend("Subflow Fwd Pkts", col("Subflow Fwd Pkts") + cfg["d_fwd_pkts"])
    if "Subflow Bwd Pkts" in cols: blend("Subflow Bwd Pkts", col("Subflow Bwd Pkts") + cfg["d_bwd_pkts"])

    # --- 3. header lengths: + header bytes per extra packet ---
    if "Fwd Header Len" in cols: blend("Fwd Header Len", col("Fwd Header Len") + cfg["d_fwd_pkts"]*cfg["hdr_bytes"])
    if "Bwd Header Len" in cols: blend("Bwd Header Len", col("Bwd Header Len") + cfg["d_bwd_pkts"]*cfg["hdr_bytes"])

    # --- 4. packet-length means / segment sizes: recompute = total / count ---
    if n_tlf is not None and n_tfp is not None:
        fwd_mean = n_tlf / n_tfp.replace(0, np.nan)
        for c in ["Fwd Pkt Len Mean", "Fwd Seg Size Avg"]:
            if c in cols: blend(c, fwd_mean.fillna(col(c)))
    if n_tlb is not None and n_tbp is not None:
        bwd_mean = n_tlb / n_tbp.replace(0, np.nan)
        for c in ["Bwd Pkt Len Mean", "Bwd Seg Size Avg"]:
            if c in cols: blend(c, bwd_mean.fillna(col(c)))
    if all(v is not None for v in (n_tlf, n_tlb, n_tfp, n_tbp)):
        tot_pk = (n_tfp + n_tbp).replace(0, np.nan)
        overall_mean = (n_tlf + n_tlb) / tot_pk
        for c in ["Pkt Len Mean", "Pkt Size Avg"]:
            if c in cols: blend(c, overall_mean.fillna(col(c)))

    # --- 5. maxima: extra packets are ~full MSS ---
    for c in ["Fwd Pkt Len Max", "Bwd Pkt Len Max", "Pkt Len Max"]:
        if c in cols:
            Xa[c] = np.where(m, np.maximum(Xa[c].astype("float64").values, cfg["added_pkt_size"]),
                             Xa[c].astype("float64").values)

    # --- 6. rates: scale by ratio of new/old (duration unchanged) ---
    def scale_rate(name, old_num, new_num):
        if name in cols and old_num is not None and new_num is not None:
            on = old_num.values; nn = new_num.values
            ratio = np.where(on > 0, nn / np.clip(on, 1, None), 1.0)
            Xa[name] = np.where(m, Xa[name].astype("float64").values * ratio,
                                Xa[name].astype("float64").values)
    if all(v is not None for v in (o_tlf, o_tlb, n_tlf, n_tlb)):
        scale_rate("Flow Byts/s", o_tlf + o_tlb, n_tlf + n_tlb)
    if all(v is not None for v in (o_tfp, o_tbp, n_tfp, n_tbp)):
        scale_rate("Flow Pkts/s", o_tfp + o_tbp, n_tfp + n_tbp)
    scale_rate("Fwd Pkts/s", o_tfp, n_tfp)
    scale_rate("Bwd Pkts/s", o_tbp, n_tbp)

    # NOTE: Std/Var features (Pkt Len Std/Var, Fwd/Bwd Pkt Len Std) are left
    # unchanged — they cannot be recomputed from totals without the individual
    # packet sizes. This is a stated limitation, not an oversight.
    return Xa.astype({c: "float32" for c in Xa.columns if Xa[c].dtype == "float64"})


def _model(kind):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=RF_TREES, random_state=SEED, n_jobs=-1)
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                         early_stopping=True, n_iter_no_change=10, random_state=SEED)


def _run_clf(kind, scale, Xtr, Xte, ytr, yte, features, int_to_class, alpha):
    # C1: baseline (scaler frozen on ORIGINAL train if MLP)
    if scale:
        sc1 = StandardScaler().fit(Xtr[features])
        tr1, te1 = sc1.transform(Xtr[features]), sc1.transform(Xte[features])
    else:
        sc1, tr1, te1 = None, Xtr[features].values, Xte[features].values
    m1 = _model(kind).fit(tr1, ytr)
    C1 = base.evaluate(m1, te1, yte, int_to_class)

    # C2: same model, ML-KEM-perturbed test
    Xte_a = make_approx_mlkem(Xte)
    te2 = sc1.transform(Xte_a[features]) if scale else Xte_a[features].values
    C2 = base.evaluate(m1, te2, yte, int_to_class)

    # pipeline on TRAIN only
    Xtr_a = make_approx_mlkem(Xtr)
    stable, shifted, _ = base.stable_features_via_ks(Xtr[features], Xtr_a[features], features, alpha)

    # C3: new model on stable features, perturbed test
    if scale:
        sc3 = StandardScaler().fit(Xtr[stable])
        tr3, te3 = sc3.transform(Xtr[stable]), sc3.transform(Xte_a[stable])
    else:
        tr3, te3 = Xtr[stable].values, Xte_a[stable].values
    m3 = _model(kind).fit(tr3, ytr)
    C3 = base.evaluate(m3, te3, yte, int_to_class)
    return C1, C2, C3, stable, shifted


def run_option_b(files=FILES, drop_identifiers=DROP_IDENTIFIERS,
                 per_class=PER_CLASS, seed=SEED, alpha=HOLM_ALPHA):
    t0 = time.time()
    print("=" * 64, "\nOPTION B — full ML-KEM-768 footprint")
    print(f"drop_identifiers={drop_identifiers}  footprint={MLKEM}\n" + "=" * 64)

    data = base.load_and_clean(files, base.LABEL_COL, base.LABEL_NORMALISE,
                               base.CLASSES, per_class, seed)
    X, y, features, class_to_int = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
    int_to_class = {i: c for c, i in class_to_int.items()}

    if drop_identifiers:
        removed = [c for c in IDENTIFIER_FEATURES if c in features]
        features = [c for c in features if c not in IDENTIFIER_FEATURES]
        print(f"  B3: dropped identifier features {removed} -> {len(features)} model features")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                           random_state=seed, stratify=y)
    print(f"  train={len(Xtr):,}  test={len(Xte):,}  features={len(features)}")

    # how many features does the footprint actually shift? (diagnostic)
    Xtr_a = make_approx_mlkem(Xtr)
    stable, shifted, _ = base.stable_features_via_ks(Xtr[features], Xtr_a[features], features, alpha)
    print(f"  footprint shifts {len(shifted)} features; {len(stable)} remain stable")
    print(f"  shifted: {shifted}")

    out = {"config": {"drop_identifiers": drop_identifiers, "footprint": MLKEM,
                      "per_class": per_class, "n_features": len(features),
                      "shifted_features": shifted, "stable_features": stable}}

    for kind, scale, label in [("rf", False, "Random Forest"), ("mlp", True, "MLP (scaled)")]:
        print("\n" + "=" * 64 + f"\n{label}\n" + "-" * 64)
        C1, C2, C3, stbl, shft = _run_clf(kind, scale, Xtr, Xte, ytr, yte,
                                          features, int_to_class, alpha)
        d12 = C2["weighted_f1"] - C1["weighted_f1"]
        d23 = C3["weighted_f1"] - C2["weighted_f1"]
        print(f"  C1 baseline           {C1['weighted_f1']:.4f}")
        print(f"  C2 shift, no pipeline {C2['weighted_f1']:.4f}   (C1->C2 {d12:+.4f})")
        print(f"  C3 shift, pipeline    {C3['weighted_f1']:.4f}   (C2->C3 {d23:+.4f})")
        print("  per-class F1 (C1 / C2 / C3):")
        for c in base.CLASSES:
            print(f"    {c:14s}{C1['per_class_f1'][c]:.4f} / {C2['per_class_f1'][c]:.4f} / {C3['per_class_f1'][c]:.4f}")
        out[kind] = {"C1": C1, "C2": C2, "C3": C3,
                     "delta_C1_C2": d12, "delta_C2_C3": d23}

    tag = "drop_id" if drop_identifiers else "keep_id"
    with open(f"option_b_{tag}_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: option_b_{tag}_results.json   elapsed: {time.time()-t0:.1f}s")
    return out


if __name__ == "__main__":
    run_option_b()
