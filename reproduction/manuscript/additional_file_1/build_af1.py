"""Build Additional file 1 (docx) from af1_tables. Text numbers are computed from the same sources."""
import sys

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import af1_tables as T  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "Additional_file_1.docx"
TITLE = ("A Simulated ML-KEM-768 Flow-Feature Shift Degrades Machine Learning-Based Intrusion Detection: "
         "Stable-Feature Screening versus Training Augmentation")
FONT = "Times New Roman"

tables = {t["id"]: t for t in T.all_tables()}

# ---------- numbers used in the running text (computed, not typed) ----------
mi = {tau: {m: T.raw_series(m, ["mi_sweep", tau]).mean() for m, _ in T.MODELS} for tau in ["0.0", "0.01", "0.05", "0.1", "0.2"]}
spread_low = max(max(mi[t][m] for t in ["0.0", "0.01", "0.05", "0.1"]) - min(mi[t][m] for t in ["0.0", "0.01", "0.05", "0.1"]) for m, _ in T.MODELS)
drop20 = [mi["0.05"][m] - mi["0.2"][m] for m, _ in T.MODELS]
fp = {}
for v in ["hybrid512", "pure768", "hybrid768", "hybrid768_p2"]:
    fp[v] = {m: (T.raw_series(m, ["per_variant", v, "C2", "weighted_f1"]) - T.C1[m]).mean() for m, _ in T.MODELS}
fp_range = {m: (min(fp[v][m] for v in fp), max(fp[v][m] for v in fp)) for m, _ in T.MODELS}
ps = pd.read_csv(T.VO / "packet_size/table_4_13_replay.csv")
ps["d"] = ps.C2 - ps.C1
ps_rng = ps.groupby("model").d.agg(lambda s: s.max() - s.min())
ops = pd.read_csv(T.VO / "operator_sensitivity/c2_cross_arm_summary.csv")
ops_max = ops.mean_C2_minus_hist_C2.abs().max()
err = pd.read_csv(T.VO / "error_analysis/error_analysis_summary.csv").set_index(["model", "condition"])
import json as _json
dup = _json.load(open(T.VO / "duplicates_sourceday/dup_global_replay.json"))
dup_share = dup["rows_in_duplicate_groups"] / dup["rows"]


def f(x, d=4, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


# ---------- document helpers ----------
doc = Document()
st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for s in doc.sections:
    s.page_height, s.page_width = Cm(29.7), Cm(21.0)
    s.left_margin = s.right_margin = Cm(2.0)
    s.top_margin = s.bottom_margin = Cm(2.0)


def set_heading_styles():
    for name, size in [("Heading 1", 14), ("Heading 2", 12)]:
        h = doc.styles[name]
        h.font.name = FONT
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = None
        rf = h.element.rPr.rFonts
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), FONT)
        h.paragraph_format.space_before = Pt(14 if size == 14 else 10)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True


set_heading_styles()


def para(text, bold=False, italic=False, size=None, align=None, space_after=None, keep=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def bullets(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(it)
        p.paragraph_format.space_after = Pt(3)


def border(cell, edges):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for e, sz in edges.items():
        el = OxmlElement(f"w:{e}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        b.append(el)
    tcPr.append(b)


def cell_text(cell, text, bold=False, size=8, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    if align:
        p.alignment = align


def add_table(tid, widths=None):
    t = tables[tid]
    cap = doc.add_paragraph()
    cap.paragraph_format.keep_with_next = True
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(3)
    r = cap.add_run(f"Table {tid}. ")
    r.bold = True
    r.font.size = Pt(10)
    r2 = cap.add_run(t["title"])
    r2.font.size = Pt(10)
    ncol = len(t["cols"])
    size = 8 if ncol <= 8 else 7.5
    tb = doc.add_table(rows=1 + len(t["rows"]), cols=ncol)
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    tb.autofit = True
    for j, c in enumerate(t["cols"]):
        cell = tb.rows[0].cells[j]
        cell_text(cell, c, bold=True, size=size)
        border(cell, {"top": 8, "bottom": 4})
    for i, row in enumerate(t["rows"], start=1):
        for j, v in enumerate(row):
            cell = tb.rows[i].cells[j]
            num = j > 0 and any(ch.isdigit() for ch in v[:2]) or v[:1] in "−+<[–"
            cell_text(cell, v, size=size)
            if i == len(t["rows"]):
                border(cell, {"bottom": 8})
    # repeat header row across pages
    trPr = tb.rows[0]._tr.get_or_add_trPr()
    h = OxmlElement("w:tblHeader")
    h.set(qn("w:val"), "true")
    trPr.append(h)
    for row in tb.rows:
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        cs.set(qn("w:val"), "true")
        trPr.append(cs)
    if widths is None:
        cw = 0.19 if size == 8 else 0.178  # cm per character, approx.
        pad = 0.38
        mins, prefs = [], []
        for j in range(ncol):
            cells = [r[j] for r in t["rows"]]
            tokens = [w for c in cells + [t["cols"][j]] for w in c.split(" ")]
            mins.append(max(len(w) for w in tokens) * cw + pad)
            prefs.append(max(max(len(c) for c in cells), max(len(w) for w in t["cols"][j].split())) * cw + pad)
        avail = 17.0
        if sum(prefs) <= avail:
            widths = prefs
        else:
            extra = avail - sum(mins)
            slack = [p_ - m_ for p_, m_ in zip(prefs, mins)]
            widths = [m_ + (max(extra, 0) * sl / sum(slack) if sum(slack) else 0) for m_, sl in zip(mins, slack)]
            if extra < 0:
                widths = [w * avail / sum(widths) for w in widths]
    tblPr = tb._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    for j, w in enumerate(widths):
        tb.columns[j].width = Cm(w)
    for row in tb.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
    if t["note"]:
        n = doc.add_paragraph()
        n.paragraph_format.space_before = Pt(3)
        n.paragraph_format.space_after = Pt(10)
        rn = n.add_run(t["note"])
        rn.font.size = Pt(8.5)
    return tb


# ---------- page numbers ----------
def footer_page_numbers():
    for s in doc.sections:
        p = s.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for kind, txt in [("begin", None), (None, "PAGE"), ("separate", None), ("end", None)]:
            if kind:
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), kind)
                r._r.append(fc)
            else:
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = txt
                r._r.append(it)
        r.font.size = Pt(9)
        hp = s.header.paragraphs[0]
        hr = hp.add_run("Additional file 1")
        hr.font.size = Pt(9)
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT


# =================== CONTENT ===================
para("Additional file 1", bold=True, size=16, space_after=2)
para("Supplementary analyses", bold=True, size=13, space_after=10)
para(f"Article: {TITLE}", italic=True, size=10.5, space_after=4)
para("Authors: Abdinur Mohamed Abdullahi and Vaithegy Doraisamy [authors to confirm]", size=10.5, space_after=12)

doc.add_heading("How to read this file", level=1)
para("This file gives the full results that the main text summarises. Unless a table says otherwise, values are "
     "means over the ten stratified 70:30 splits (split seeds 0–9), differences are paired within each split, and "
     "scores are weighted F1 rounded to four decimals.")
bullets([
    "C1 trains and tests on original data. C2 trains on original data and tests on transformed data. C3 trains on the "
    "selected stable features and is tested on the transformed values of those features.",
    "The identifier-free design (B3) removes Dst Port and Init Fwd Win Byts from every condition.",
    "Equivalence uses two one-sided tests (TOST; Schuirmann 1987) on the paired differences, with a 90% paired t "
    "interval and Holm (1979) adjustment across the three models. The primary margin for weighted F1 is ±0.01; ±0.005 is a stricter check.",
    "p-values below 0.001 are shown as <0.001, and adjusted values above 0.999 as 1.000.",
])
para("Contents: S1 Implementation and analysis record; S2 Selection threshold; S3 Pipeline component controls; "
     "S4 Sensitivity to the transformation; S5 Exposure and adoption; S6 Retraining reference; S7 Duplicates, grouped "
     "split and source-day holdout; S8 Feature restrictions; S9 Overlapping splits; S10 Error analysis; S11 Processing "
     "time; S12 Unanticipated timing shift; S13 Training augmentation versus screening.", size=10)

# ---- S1
doc.add_heading("S1 Implementation and analysis record", level=1)
doc.add_heading("S1.1 How the models were fitted", level=2)
para("Within each split, C1 and C2 used separate fits with the same original training data, all 70 features, the "
     "same settings and model seed 42. C2 differs from C1 only in its test inputs. C3 fits its own classifier on the "
     "original training values of the selected features. The MI-threshold sweep (Table S3) and the footprint "
     "evaluation (Table S7) each refit C3.")
para("A later run checked that the separate fits do not affect the comparison. It fitted one full-feature classifier "
     "per split and model, evaluated it on both test representations, and fitted one C3 model. This reproduced all 90 "
     "stored weighted-F1 scores (C1, C2 and C3 for three models in ten splits) exactly.")
para("The main recovery analysis uses the C3 scores stored with the footprint evaluation. For RF in split 6, the "
     "MI-sweep fit at τ = 0.05 differs from that score by 2.7 × 10⁻⁵, because of a near-tie in the averaged tree "
     "probabilities that depends on summation order. No reported conclusion depends on this.")
doc.add_heading("S1.2 Software", level=2)
para("Table S1 lists the software versions.")
add_table("S1")
doc.add_heading("S1.3 When each analysis was added", level=2)
para("Table S2 lists every statistical family, its role and when its model runs were produced. Three points follow "
     "from this record.")
bullets([
    "The ±0.01 and ±0.005 margins were defined for weighted F1. The per-attack TOST reuses the ±0.01 margin and was "
    "run after the per-class results were available, so it is reported as exploratory.",
    "The random-stable control (Table S6) was specified without a test direction, and the analysis used a one-sided "
    "test. The two-sided Holm-adjusted p-values are also below 0.001.",
    "The corrected transformation operators, the grouped split and the corrected resampled t statistic were added "
    "after all earlier results were known. They are sensitivity analyses; the recorded procedure stays primary.",
])
add_table("S2")

# ---- S2
doc.add_heading("S2 Selection threshold", level=1)
para(f"Table S3 shows how C3 changes with the MI threshold. From τ = 0 to τ = 0.10, mean C3 weighted F1 moved by at "
     f"most {f(spread_low)} for any model. At τ = 0.20 the subset fell to 16 features, and C3 was lower than at "
     f"τ = 0.05 by {f(drop20[0])} (RF), {f(drop20[1])} (XGBoost) and {f(drop20[2])} (MLP). The sweep reuses the same test partitions, so it describes sensitivity, not an "
     f"independent choice of threshold.")
add_table("S3")

# ---- S3
doc.add_heading("S3 Pipeline component controls", level=1)
para("These controls ask which stage of the pipeline restores performance (Tables S4–S6).")
bullets([
    "Stability screening alone is enough. Keeping all 32 stable candidates (KS-only) returned weighted F1 to within "
    "±0.01 of C1 for all three models. Adding the MI filter changed the mean by less than 0.001 while reducing the set "
    "to a mean of 24.1 features.",
    "Label relevance alone fails. Without screening, MI kept all 23 shifted features at τ = 0.05, and 13–14 of them "
    "at equal set size, because the shifted features are also label-relevant. Weighted F1 stayed far below C1.",
    "After screening, MI chooses better than chance. The MI subset beat the average of equal-size random stable "
    "subsets in all ten splits for every model (Table S6). The gain is small, and single random subsets sometimes "
    "scored higher.",
])
add_table("S4")
add_table("S5")
add_table("S6")

# ---- S4
doc.add_heading("S4 Sensitivity to the transformation", level=1)
doc.add_heading("S4.1 Footprint", level=2)
para(f"The size of the C2 loss depended on the assumed footprint (Table S7). Mean C2 − C1 ranged from "
     f"{f(fp_range['rf'][0])} to {f(fp_range['rf'][1])} for RF, {f(fp_range['xgb'][0])} to {f(fp_range['xgb'][1])} for "
     f"XGBoost and {f(fp_range['mlp'][0])} to {f(fp_range['mlp'][1])} for MLP. C3 reused the subset selected under the "
     f"main footprint. Its inputs are unchanged by every footprint, so C3 − C1 is the same in every row. This shows "
     f"transfer within one family of transformations, not robustness to changes in the retained features.")
add_table("S7")
doc.add_heading("S4.2 Added-packet size", level=2)
para(f"Changing the added-packet size from 1,460 to 1,330 or 1,200 bytes changed only the three maximum packet-length "
     f"features. The screen classed all three as shifted at every size, so the stable and selected sets were "
     f"identical and C3 did not change (Table S8). C2 stayed below C1 in every split at every size, and mean C2 − C1 "
     f"moved by at most {f(ps_rng.max(), 3)}. Because C3 is identical by construction, no C3 equivalence test across "
     f"packet sizes is reported.")
add_table("S8")
doc.add_heading("S4.3 Corrected operators", level=2)
para(f"The original operator leaves 58,242 of the 74,127 eligible flows (78.57%) with a directional maximum above the "
     f"directional byte total, and 17,354 flows with a zero byte rate despite added bytes. We repeated C1, C2 and C3 "
     f"with each correction and with both. Every one of the 90 corrected C2 scores stayed below its C1 score, and mean "
     f"C2 changed by at most {f(ops_max, 3)} (Table S9). The corrections do not model timing, variances or packet "
     f"framing.")
add_table("S9")

# ---- S5
doc.add_heading("S5 Exposure and adoption", level=1)
para("The main transformation reaches every bidirectional flow, 62.0% of the data, with unequal coverage across "
     "classes (Table S10). A TLS-likely rule reaches only 5.5% of rows and no DoS or Bot flows.")
para("When only TLS-likely flows were transformed, C2 fell by at most 0.003, so there was little to recover, and C3 "
     "stayed within ±0.005 of C1 (Table S11). The main condition is therefore a high-exposure stress test. Because "
     "destination port defines this exposure, the identifier-free feature set is the primary analysis.")
para("The screen was also less sensitive at this exposure. When the simulated training copy was shifted only on "
     "TLS-likely rows, eight altered features passed as stable and were selected (Table S12). When the whole copy was "
     "shifted, as in the main pipeline, the screen flagged exactly the 23 altered features in every split. The screen "
     "should therefore be run against a fully shifted copy, not against the adoption level seen at the time.")
para("In the partial-adoption analysis, transformed rows were mixed into the test set only. The full-feature model "
     "lost performance steadily as the transformed share grew, while C3 stayed constant by construction (Table S13).")
add_table("S10")
add_table("S11")
add_table("S12")
add_table("S13")

# ---- S6
doc.add_heading("S6 Retraining reference", level=1)
para("As a reference, each classifier was retrained on transformed training data with all 70 features and "
     "tested on transformed test data (Table S14). It used the original labels on transformed training data, so, like "
     "the pipeline, it needs the simulated transformation but no captured post-quantum traffic. It was trained only on "
     "transformed data and was not tested on original or mixed traffic. "
     "Retraining reached C1-level performance, which shows that the C2 loss comes from the mismatch between training "
     "and test data, not from lost information. C3 was within ±0.01 of the retrained reference in weighted F1 for all "
     "three models.")
add_table("S14")

# ---- S7
doc.add_heading("S7 Duplicates, grouped split and source-day holdout", level=1)
para(f"Exact duplicates are common: {100 * dup_share:.1f}% of prepared rows ({dup['rows_in_duplicate_groups']:,} of "
     f"{dup['rows']:,}) share their 70-feature vector with at least one other row (53.6% of DoS and 48.9% of Bot rows). "
     f"There are {dup['conflicting_label_groups']:,} groups ({dup['rows_in_conflicting_label_groups']:,} rows: 1,745 Infiltration and 1,395 Benign) "
     "whose identical vectors carry both labels. Under the random splits, 28.5% of test rows had an exact copy in "
     "training (Table S15).")
para("The grouped split keeps every duplicate group on one side of the train–test boundary (Table S16). The baseline "
     "hardly changed, C2 still fell for every model, and C3 stayed within ±0.01 of C1 for all three models. At ±0.005, "
     "RF and XGBoost remained equivalent, but MLP did not.")
para("The source-day holdout trains on one Infiltration capture day and tests on the other (Table S17). RF and "
     "XGBoost C3 were equivalent to their cross-day C1 in both directions. MLP C3 scored above its cross-day C1, so "
     "equivalence failed because C3 was higher, not lower; this is not a recovery failure.")
add_table("S15")
add_table("S16")
add_table("S17")

# ---- S8
doc.add_heading("S8 Feature restrictions", level=1)
para("The main text reports weighted F1 for the identifier-free, timing-free and combined restrictions (Table 4 of "
     "the main text). Table S18 adds the per-attack results. In the identifier-free design, MLP stayed below its "
     "baseline for Bot and Infiltration. In the combined design, only the DoS comparisons for RF and XGBoost met "
     "±0.01. Table S19 shows that no MI threshold removed the identifier-free MLP gap.")
add_table("S18")
add_table("S19")

# ---- S9
doc.add_heading("S9 Overlapping splits", level=1)
para("The ten splits reuse the same rows, so the recorded tests understate uncertainty. Table S20 repeats the tests "
     "with the corrected resampled t statistic (Nadeau and Bengio 2003). Every degradation and ±0.01 recovery decision "
     "stays the same. At ±0.005, the MLP results for the main design and for C3 versus retrained are no longer "
     "established, and the timing-free results remain not established.")
add_table("S20")

# ---- S10
doc.add_heading("S10 Error analysis", level=1)
r1, r2 = err.loc[("rf", "C1")], err.loc[("rf", "C2")]
para(f"Under C2, the benign false-alarm rate fell for every model, while the share of attacks classified as Benign "
     f"rose (for example, RF: false alarms {f(r1.benign_false_alarm, 3)} → {f(r2.benign_false_alarm, 3)}; attacks "
     f"classified as Benign {f(r1.attack_to_benign, 3)} → {f(r2.attack_to_benign, 3)}). Fewer alerts therefore came "
     f"with more missed attacks. Under C3, both rates returned close to their C1 values (Table S21). Macro F1 followed "
     f"the same pattern as weighted F1.")
add_table("S21")

# ---- S11
doc.add_heading("S11 Processing time", level=1)
para("C3 trained faster than C1 for every model in the main design. Inference was not always faster: RF inference "
     "was slower with C3 than with C2 in both designs, and MLP inference was slower in the identifier-free design "
     "(Table S22). Timing depends on hardware and software, so these ratios describe the original execution "
     "environment only.")
add_table("S22")

# ---- S12
ts = pd.read_csv(T.TS / "timing_shift_statistics.csv")
tsel = pd.read_csv(T.TS / "anticipated_selection_summary.csv")
def tsv(cond, var, d, m, col="mean_difference"):
    return ts[(ts.condition == cond) & (ts.variant == var) & (ts.delay_us == d) & (ts.model == m)].iloc[0][col]
new_shift = sorted(set(";".join(tsel[tsel.delay_us == 1000].newly_shifted.dropna()).split(";")))
doc.add_heading("S12 Unanticipated timing shift", level=1)
para("This test asks what happens when a shift also moves features that C3 keeps. On top of the main "
     "transformation, an assumed handshake delay d of 0.1, 1, 10 or 100 ms was added to eligible flows. It lengthens "
     "Flow Duration, one flow gap and one gap in each direction with at least two packets, and the rates and gap "
     "means, maxima and single-gap minima derived from them. IAT standard deviations and the Active and Idle "
     "summaries are unchanged. The delays are assumptions, not measurements. The plan, including the delays, the "
     "two ways of updating the maxima and the decision rules, was fixed before the run.")
para("Two versions of C3 were compared. In the unanticipated version, C3 is the paper's model: its screen saw only "
     "the main transformation. In the anticipated version, the screen compares the training data with a copy that "
     "also includes the delay, and the model is refitted on the new selection.")
para(f"When the screen had not seen the delay, all three models stayed within ±0.01 of C1 only at 0.1 ms "
     f"(Table S23). At 1 ms, RF fell by {f(-tsv('C3u','additive_max',1000,'rf'), 3)}; at 10 ms, RF fell by "
     f"{f(-tsv('C3u','additive_max',10000,'rf'), 3)} and XGBoost by {f(-tsv('C3u','additive_max',10000,'xgb'), 3)}. "
     f"The floor variant gave the same pattern. MLP stayed within {f(max(abs(tsv('C3u','additive_max',d,'mlp')) for d in (100,1000,10000,100000)), 4)} "
     f"of C1 at every delay. A likely reason, not tested, is that the MLP standardises its inputs and the timing features "
     f"vary by seconds, so even a 100 ms delay moves the scaled inputs very little, whereas the tree models split at "
     f"small thresholds.")
para(f"When the screen was run against a copy that included the delay, it classed the {len(new_shift)} changed timing "
     f"features in the selected set as shifted (a twelfth, Fwd IAT Min, at 100 ms), kept 12–14 features, and C3 stayed "
     f"within ±0.01 of C1 for all three models at every delay (Table S24). These values are close to the timing-free "
     f"result in the main text. Under the corrected resampled t statistic, MLP at 0.1 ms and 1 ms is borderline "
     f"(Holm-adjusted p = {f(tsv('C3a','additive_max',100,'mlp','corr_tost_0.01_holm'), 4)} and "
     f"{f(tsv('C3a','additive_max',1000,'mlp','corr_tost_0.01_holm'), 4)}).")
para("Recovery therefore depends on the screen seeing the shift the detector will face. The reported run was made in "
     "Colab and reproduces the locked main scores exactly at zero delay. An earlier run on different hardware gave the "
     "same ±0.01 decisions; it differed only in one XGBoost baseline split.")
add_table("S23")
add_table("S24")

# ---- S13
aug = pd.read_csv(T.AUG / "augmentation_statistics.csv")
INF = ["original", "well_specified", "hybrid512", "pure768", "hybrid768_p2", "packet_1200", "packet_1330",
       "adoption_25", "adoption_50", "adoption_75"]
ain = aug[(aug.contrast == "A-C1") & aug.test_set.isin(INF)]
cin = aug[(aug.contrast == "C3-A") & aug.test_set.isin(INF)]
g = lambda ts, c, m, col="mean_difference": aug[(aug.test_set == ts) & (aug.contrast == c) & (aug.model == m)].iloc[0][col]
doc.add_heading("S13 Training augmentation versus screening", level=1)
para("Retraining on transformed data needs only the simulation, not captured post-quantum traffic. That points to a "
     "simpler defence than screening: train on the original training data stacked with its simulated copy, with the "
     "same labels and all 70 features (A). A plan fixed before the run compared A with the paper's C3 on 14 test sets: "
     "original data, the simulated shift as assumed, three other footprints, two other packet sizes, three adoption "
     "levels and four timing delays that neither defence anticipated (Table S25).")
para(f"A stayed within ±0.01 of C1 for all three models in all ten simulated test sets, including those where the "
     f"simulated size of the change was wrong (A − C1 from {f(ain.mean_difference.min(), 4, True)} to "
     f"{f(ain.mean_difference.max(), 4, True)}). C3 − A ranged from {f(cin.mean_difference.min(), 4, True)} to "
     f"{f(cin.mean_difference.max(), 4, True)}. XGBoost scored slightly higher with A in nine of the ten simulated sets; "
     f"Random Forest scored slightly higher with C3 in four; no set showed C3 within the margin while A was not. The "
     f"rule fixed before the run therefore does not support a claim that screening is the more robust defence.")
para(f"Neither defence withstood the unsimulated delays. At 1 ms, Random Forest fell by {f(-g('delay_1000us','A-C1','rf'), 3)} "
     f"with A and {f(-g('delay_1000us','C3-C1','rf'), 3)} with C3; at 10 ms, XGBoost fell by "
     f"{f(-g('delay_10000us','A-C1','xgb'), 3)} and {f(-g('delay_10000us','C3-C1','xgb'), 3)}. How well the migration is "
     f"simulated therefore matters more than which of the two defences is used. Screening still shows which features "
     f"the migration changes and gives a smaller model (24 of 70 features).")
para("The reported run was made in Colab; a replication in the session workspace gave identical scores for all 1,260 "
     "evaluations.")
add_table("S25")

# ---- references
doc.add_heading("References", level=1)
for ref in [
    "Holm S (1979) A simple sequentially rejective multiple test procedure. Scand J Stat 6(2):65–70",
    "Nadeau C, Bengio Y (2003) Inference for the generalization error. Mach Learn 52(3):239–281. "
    "https://doi.org/10.1023/A:1024068626366",
    "Schuirmann DJ (1987) A comparison of the two one-sided tests procedure and the power approach for assessing the "
    "equivalence of average bioavailability. J Pharmacokinet Biopharm 15(6):657–680. https://doi.org/10.1007/BF01068419",
]:
    para(ref, size=10)

footer_page_numbers()
doc.core_properties.title = "Additional file 1: Supplementary analyses"
doc.save(OUT)
print("saved", OUT)
