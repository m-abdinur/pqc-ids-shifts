"""Replay macro F1, benign false-alarm rate and attack->benign rate from saved predictions
(operator-sensitivity run, historical arm = locked main predictions for C2/C3; original = C1)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
root = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/sensrun/r/results"
rows = []
for seed in range(10):
    for m in ["rf", "xgb", "mlp"]:
        for cond, arm in [("C1", "original"), ("C2", "historical"), ("C3", "historical")]:
            z = np.load(f"{root}/predictions/seed{seed}_{m}_{arm}_{cond}.npz")
            yt, yp = z["y_true"], z["y_pred"]
            ben = yt == 0; att = ~ben
            rows.append(dict(seed=seed, model=m, condition=cond,
                             weighted_f1=f1_score(yt, yp, average="weighted"),
                             macro_f1=f1_score(yt, yp, average="macro"),
                             benign_false_alarm=float(np.mean(yp[ben] != 0)),
                             attack_to_benign=float(np.mean(yp[att] == 0))))
df = pd.DataFrame(rows)
out = sys.argv[2] if len(sys.argv) > 2 else "."
df.to_csv(f"{out}/error_analysis_by_split.csv", index=False)
s = df.groupby(["model", "condition"]).mean(numeric_only=True).drop(columns="seed")
s.to_csv(f"{out}/error_analysis_summary.csv")
print(s.round(4))
