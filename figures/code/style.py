"""Figure style for the paper, Frontiers of Computer Science (FCS) version: the plain Matplotlib look (as in the
Matplotlib brochure notebook) with
- the FCS colour guide (Instructions for Authors 2025): blue #004385, orange #C66219, gold #FFC000, red #B7282C,
  purple #742984, teal #1A7C6B,
- Arial (or Liberation Sans, which has the same letter widths; bundled in fonts/ for Binder),
- subfigure labels (a), (b), (c) in the top-left corner of each panel, as FCS asks ("Use (a), (b), ..."),
sized for print at full page width (170 mm)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from pathlib import Path
from matplotlib.transforms import Bbox

HERE = Path(__file__).parent
for _ttf in sorted((HERE / "fonts").glob("*.ttf")):                 # make the bundled fonts available
    fm.fontManager.addfont(str(_ttf))
FONT = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "Liberation Sans"

plt.style.use("default")
plt.rcParams.update({
    "font.family": FONT, "font.size": 9,                            # print size; journal lettering 8-12 pt
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
    "mathtext.bf": FONT + ":bold", "mathtext.cal": "cmsy10",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",  # fonts embedded; SVG text editable
    "axes.prop_cycle": plt.cycler(color=["#004385", "#C66219", "#FFC000", "#742984", "#1A7C6B", "#B7282C"]),
})

MM = 1 / 25.4
W = 170                                     # full page width; FCS sets no limit, these keep the earlier journal sizes
MAX_W, MAX_H = 170.0, 225.0                 # largest figure in mm

# FCS colour guide. One colour per condition, kept the same in every figure. Blue and purple look alike to
# readers with red-green colour blindness, so C1 is always drawn dashed or labelled where A also appears.
FCS = {"blue": "#004385", "orange": "#C66219", "gold": "#FFC000", "red": "#B7282C", "purple": "#742984",
       "teal": "#1A7C6B"}
COND = {"C1": FCS["blue"], "C2": FCS["orange"], "C3": FCS["teal"], "A": FCS["purple"]}
# sequential map for heatmaps, from FCS colours with lightness falling monotonically (white, gold, orange, red),
# so the order of values also survives colour blindness and greyscale printing
from matplotlib.colors import LinearSegmentedColormap
FCS_SEQ = LinearSegmentedColormap.from_list("fcs_seq", ["#FFFFFF", FCS["gold"], FCS["orange"], FCS["red"]])


def ink(rgba):
    """black or white text, whichever contrasts more with a cell colour"""
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "black" if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else "white"
NAME = {"C1": "C1 Original", "C2": "C2 Shifted", "C3": "C3 Screening", "A": "A Augmentation"}
MARK = {"C1": "o", "C2": "s", "C3": "^", "A": "D"}
MODEL_NAMES = [("rf", "RF"), ("xgb", "XGBoost"), ("mlp", "MLP")]


def fig(h_mm, w_mm=W, layout="constrained"):
    """a figure of the final printed size (mm)"""
    return plt.figure(figsize=(w_mm * MM, h_mm * MM), layout=layout)


def panel(ax, letter, title=None):
    """subfigure label (a), (b), ... in the top-left corner; optional plain title in the centre"""
    ax.set_title("(%s)" % letter, loc="left", fontsize=10)
    if title:
        ax.set_title(title)


def _content_bbox(f):
    """Bounding box (inches) of everything drawn; axes with the axis turned off count only for what is drawn on
    them, so the figure can be cropped closely."""
    import matplotlib.spines as mspines, matplotlib.axis as maxis
    r = f.canvas.get_renderer()
    boxes = []
    for a in f.get_axes():
        if not a.get_visible():
            continue
        if a.axison:
            bb = a.get_tightbbox(r)
            if bb is not None:
                boxes.append(bb)
            continue
        for c in a.get_children():
            if c is a.patch or isinstance(c, (mspines.Spine, maxis.Axis)) or not c.get_visible():
                continue
            if hasattr(c, "get_text") and not c.get_text():
                continue
            try:
                bb = c.get_window_extent(r)
            except Exception:
                continue
            if bb is not None and np.isfinite(bb.get_points()).all() and (bb.width > 0 or bb.height > 0):
                boxes.append(bb)
    for a in list(f.legends) + list(f.texts):
        if a.get_visible():
            boxes.append(a.get_window_extent(r))
    for sf in getattr(f, "subfigs", []):
        for a in list(sf.legends) + list(sf.texts):
            if a.get_visible():
                boxes.append(a.get_window_extent(r))
    return Bbox.union(boxes).transformed(f.dpi_scale_trans.inverted())


def _align_left_titles(f):
    """Start each left title (the panel letter) above the left edge of the panel's tick labels and axis label,
    the top-left corner of the panel's bounding box."""
    r = f.canvas.get_renderer()
    for ax in f.get_axes():
        t = ax._left_title
        if not (ax.axison and t.get_text()):
            continue
        yb = ax.yaxis.get_tightbbox(r)                                # None when the panel shows no y labels
        x0 = min(ax.bbox.x0, yb.x0) if yb is not None else ax.bbox.x0
        t.set_x((x0 - ax.bbox.x0) / ax.bbox.width)


SHOW, SHOW_WIDTH = True, 900                # in a notebook, save() also shows the figure at this width (pixels)


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def save(f, name, outdir=HERE / "out"):
    """Close-cropped vector (PDF, SVG), 600 dpi PNG and 600 dpi TIFF output; refuses figures wider or taller than the journal allows."""
    outdir.mkdir(exist_ok=True)
    f.canvas.draw()                                                   # run the layout before measuring
    f.set_layout_engine("none")                                       # freeze it, so every file has the same layout
    _align_left_titles(f)
    pad = 0.04                                                        # ~1 mm white margin (inches)
    bb = _content_bbox(f)
    px = max(0.0, min(pad, (MAX_W / 25.4 - bb.width) / 2))
    bb = Bbox.from_extents(bb.x0 - px, bb.y0 - pad, bb.x1 + px, bb.y1 + pad)
    w_mm, h_mm = bb.width * 25.4, bb.height * 25.4
    assert w_mm <= MAX_W + 1e-6, f"{name} is {w_mm:.1f} mm wide (limit {MAX_W} mm)"
    assert h_mm <= MAX_H, f"{name} is {h_mm:.1f} mm high (limit {MAX_H} mm)"
    for ext in ("pdf", "svg"):
        f.savefig(outdir / f"{name}.{ext}", bbox_inches=bb)
    f.savefig(outdir / f"{name}.png", dpi=600, bbox_inches=bb)
    plt.close(f)
    from PIL import Image as _PIL                                     # TIFF for the final files FCS asks for
    with _PIL.open(outdir / f"{name}.png") as im:
        im.convert("RGB").save(outdir / f"{name}.tif", compression="tiff_lzw", dpi=(600, 600))
    if SHOW and _in_notebook():                                       # in Jupyter: show the saved figure
        from IPython.display import Image, display
        display(Image(filename=str(outdir / f"{name}.png"), width=SHOW_WIDTH))
    return w_mm, h_mm
