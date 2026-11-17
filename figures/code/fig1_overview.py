"""Figure 1 (FCS). Message: one simulated shift prepares both defences on training data only, four conditions are
evaluated on test data, and the analyses answer the paper's two questions. Study overview, read top to bottom.
Design facts only (no results). Counts come from the manuscript (Sections 3.3, 3.4, 4.1.1, 4.2; Table 2) and are
checked below against data/features_ks.csv and data/weighted_f1_by_split.csv. Footprint values: pilot_option_b.MLKEM
(main hybrid footprint); conditions: revised_pipeline_main_only.one_split."""
import pandas as pd
from matplotlib.patches import Rectangle
from style import *

# ---------------------------------------------------------------- design facts and guards
ks = pd.read_csv("data/features_ks.csv")
N_CAND, N_STABLE = len(ks), int((ks.group == "Stable").sum())
N_SEL = int((ks.selected == 10).sum())                              # selected in all ten splits
assert (N_CAND, N_STABLE, int((ks.group == "Shifted").sum()), N_SEL) == (55, 32, 23, 24)
assert sorted(pd.read_csv("data/weighted_f1_by_split.csv").seed.unique()) == list(range(10))   # seeds 0-9
N_ALL, N_TR, N_TE, N_ELIG, N_AUG = 119571, 83699, 35872, 74127, 167398    # Sections 3.3, 4.1.1, 4.2; Table 2
assert N_TR + N_TE == N_ALL and 2 * N_TR == N_AUG                   # A = training split + its shifted copy

H = 156
f = fig(H, layout=None)
cv = f.add_axes([0, 0, 1, 1]); cv.set_xlim(0, W); cv.set_ylim(0, H); cv.axis("off")
GREY = "0.35"
PT = 25.4 / 72                                                       # mm per point
BOXES = []                                                           # (rectangle, texts) for the fit check


def box(x, y, w, h, lines, edge="k", lw=0.9):
    """lines: (text, size, weight, colour) from top to bottom, centred in the box"""
    r = Rectangle((x, y), w, h, fc="white", ec=edge, lw=lw, zorder=2)
    cv.add_patch(r)
    heights = [s * 1.32 * PT for _, s, _, _ in lines]
    yy = y + h / 2 + sum(heights) / 2
    texts = []
    for (t, s, wt, c), lh in zip(lines, heights):
        yy -= lh
        texts.append(cv.text(x + w / 2, yy + lh / 2, t, ha="center", va="center", fontsize=s, fontweight=wt,
                             color=c, zorder=3))
    BOXES.append((r, texts))
    return r


def listbox(x, y, w, h, title, rows, edge="k", lw=0.9):
    """a box with a bold title and rows of (item, section); sections right-aligned in grey"""
    r = Rectangle((x, y), w, h, fc="white", ec=edge, lw=lw, zorder=2)
    cv.add_patch(r)
    texts = [cv.text(x + 2.2, y + h - 2.0, title, ha="left", va="top", fontsize=9, fontweight="bold", zorder=3),
             cv.text(x + w - 2.2, y + h - 2.2, "Section", ha="right", va="top", fontsize=8, color=GREY, zorder=3)]
    ys = []
    yy = y + h - 7.6
    for item, sec in rows:
        texts.append(cv.text(x + 2.2, yy, "• " + item, ha="left", va="center", fontsize=8, zorder=3))
        texts.append(cv.text(x + w - 2.2, yy, sec, ha="right", va="center", fontsize=8, color=GREY, zorder=3))
        ys.append(yy)
        yy -= 3.75
    BOXES.append((r, texts))
    return r, ys


def arrow(p, q, color="k", ls="-"):
    cv.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color, ls=ls, mutation_scale=9, shrinkA=0, shrinkB=0))


def line(xs, ys):
    cv.plot(xs, ys, color="k", lw=1.0, solid_capstyle="butt", zorder=1)


# ---------------------------------------------------------------- 1 data and splits
box(29, 144, 112, 11, [("CSE-CIC-IDS2018 flows (CICFlowMeter features)", 9, "normal", "k"),
                       (f"{N_ALL:,} records: Benign, DoS, Bot and Infiltration; 70 features", 8, "normal", "k")])
arrow((85, 144), (85, 137.5))
box(42, 129.5, 86, 8, [("Ten stratified 70:30 train–test splits (seeds 0–9)", 9, "normal", "k")])

# ---------------------------------------------------------------- 2 two lanes: training data | test data
XL, XR = 51, 134                                                     # lane centres
cv.add_patch(Rectangle((3, 47), 95, 78.5, fc="0.94", ec="none", zorder=0))
cv.add_patch(Rectangle((102, 47), 65, 78.5, fc="0.94", ec="none", zorder=0))
line([85, 85], [129.5, 125.5]); line([XL, XR], [125.5, 125.5])
arrow((XL, 125.5), (XL, 121)); arrow((XR, 125.5), (XR, 121))
box(XL - 30, 113, 60, 8, [(f"Training split: {N_TR:,} records", 9, "normal", "k")])
box(XR - 28, 113, 56, 8, [(f"Test split: {N_TE:,} records", 9, "normal", "k")])

# the shift T, one box across both lanes: the same T prepares the defences and shifts the test data
arrow((XL, 113), (XL, 107)); arrow((XR, 113), (XR, 107))
box(5, 87, 160, 20,
    [("Shift $T$ (main hybrid footprint), applied to a copy of each split", 9, "normal", "k"),
     ("Byte totals +1,184 B forward and +1,088 B backward (ML-KEM-768 sizes, FIPS 203); +1 packet* and", 8, "normal", "k"),
     ("+40 header bytes* per direction; maximum length at least 1,460 B*; means and rates recomputed; timing unchanged", 8,
      "normal", "k"),
     (f"Only flows with packets in both directions change: {N_ELIG:,} of {N_ALL:,} records.   * assumption", 8,
      "normal", GREY)],
    edge=COND["C2"], lw=1.6)

# training lane: the two defences, then fitting
XC3, XA = 28, 74
arrow((XC3, 87), (XC3, 81)); arrow((XA, 87), (XA, 81))
box(6, 62, 44, 19, [("C3 screening", 9, "normal", "k"),
                    (f"KS test + Holm: {N_STABLE} of {N_CAND}", 8, "normal", "k"),
                    ("candidates unchanged", 8, "normal", "k"),
                    (f"MI ≥ 0.05: {N_SEL} kept ($\\mathcal{{F}}$)", 8, "normal", "k")],
    edge=COND["C3"], lw=1.6)
box(52, 62, 44, 19, [("A augmentation", 9, "normal", "k"),
                     ("original + shifted", 8, "normal", "k"),
                     (f"training rows: {N_AUG:,}", 8, "normal", "k"),
                     ("all 70 features", 8, "normal", "k")],
    edge=COND["A"], lw=1.6)
arrow((XC3, 62), (XC3, 57.5)); arrow((XA, 62), (XA, 57.5))
box(8, 48, 86, 9.5, [("Fit RF, XGBoost and MLP", 9, "normal", "k"),
                     ("C1, C2: original rows; C3: $\\mathcal{F}$ only; A: augmented rows", 8, "normal", "k")])

# test lane: what each condition is tested on
arrow((XR, 87), (XR, 81))
box(XR - 28, 62, 56, 19, [("Test data", 9, "normal", "k"),
                          ("C1: original", 8, "normal", "k"),
                          ("C2, C3, A: shifted ($\\mathcal{F}$ only for C3)", 8, "normal", "k"),
                          ("never used for screening or fitting", 8, "normal", GREY)])

# ---------------------------------------------------------------- 3 evaluation
arrow((XL, 48), (XL, 40.5)); arrow((XR, 62), (XR, 40.5))
box(5, 31.5, 160, 9,
    [("Evaluate C1, C2, C3 and A on the ten paired splits", 9, "normal", "k"),
     ("Weighted F1; loss: C2 − C1; recovery: C3 or A within ±0.01 of C1 (TOST, also after the overlap correction)", 8,
      "normal", "k")])

# ---------------------------------------------------------------- 4 the two questions and the analyses that answer them
X1, W1, X2, W2 = 35, 62, 101, 67
arrow((X1 + W1 / 2, 31.5), (X1 + W1 / 2, 25.5)); arrow((X2 + W2 / 2, 31.5), (X2 + W2 / 2, 25.5))
_, q1 = listbox(X1, 1, W1, 24.5, "Q1  What drives the loss?",
                [("Loss under full exposure", "4.3.2"), ("Pure byte increments vs lab data", "4.3.4"),
                 ("One assumption changed at a time", "4.3.4")])
listbox(X2, 1, W2, 24.5, "Q2  Do the defences recover?",
        [("Ten non-timing conditions", "4.3.3, 4.3.5"), ("Mismatched packet structure", "4.3.5"),
         ("Delays absent from training", "4.3.5"), ("Per attack class", "4.3.6"),
         ("Predictors removed; low exposure", "4.3.7")])
box(2, 3, 29, 20.5, [("CIC-PQC_OAV v1", 8, "bold", "k"), ("lab TLS 1.3", 8, "normal", "k"),
                     ("handshakes:", 8, "normal", "k"), ("X25519 vs", 8, "normal", "k"), ("ML-KEM-768", 8, "normal", "k")])
arrow((31, q1[1]), (X1, q1[1]))

# ---------------------------------------------------------------- every text inside its box (1 mm margin)
f.canvas.draw()
r = f.canvas.get_renderer()
to_mm = cv.transData.inverted()
BAD = []
for rect, texts in BOXES:
    x0, y0 = rect.get_x(), rect.get_y(); x1, y1 = x0 + rect.get_width(), y0 + rect.get_height()
    for t in texts:
        (a, b), (c, d) = to_mm.transform(t.get_window_extent(r).get_points())
        if not (a >= x0 + 1 and c <= x1 - 1 and b >= y0 + 0.5 and d <= y1 - 0.5):
            BAD.append(f"text {t.get_text()!r} ({a:.1f}-{c:.1f}, {b:.1f}-{d:.1f}) outside box ({x0}-{x1}, {y0}-{y1})")
assert not BAD, "\n".join(BAD)
save(f, "Figure_1")
print("ok")
