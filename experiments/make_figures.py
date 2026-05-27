import os
"""Generate clean figures summarizing the experiments (representation analysis).
No AI-slop styling: white bg, muted palette, minimal spines, real data only. Saves PNGs."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

R = Path(os.environ.get("PROJECT_ROOT", ".")); D = R / "results/diagnostics"
FIG = R / "notes/figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight", "axes.titlesize": 12})
BLUE, GRAY, RED, GREEN = "#3b6ea5", "#9aa0a6", "#b5651d", "#4a7c59"

# ---- Fig 1: recoverability atlas (transport Cohen's d per modality) ----
arms = ["Endocrine\ntrajectory\n(AMH direct)", "Ovarian US\n(reserve\nphenotype)", "Embryo\nmorphology\n(AMH)"]
d_vals = [1.14, 2.44, 0.05]; colors = [GREEN, BLUE, GRAY]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
b = ax.bar(arms, d_vals, color=colors, width=0.6)
for rect, v, lab in zip(b, d_vals, ["p<0.001", "p<0.001", "n.s."]):
    ax.text(rect.get_x()+rect.get_width()/2, v+0.05, f"d={v}\n{lab}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Generative transport effect (Cohen's d)")
ax.set_title("Recoverability atlas: where AMH-linked signal lives")
ax.set_ylim(0, 2.9); ax.axhline(0, color="k", lw=0.6)
fig.savefig(FIG/"fig1_atlas.png"); plt.close(fig)

# ---- Fig 2: confound robustness (probe fragile vs transport robust) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
# (A) ovary probe AUROC across residualization spectrum
labels = ["raw", "−resolution\n/aspect", "−global\nintensity", "−full\nhistogram"]
auroc = [0.88, 0.90, 0.85, 0.72]
a1.bar(labels, auroc, color=BLUE, width=0.62)
a1.axhline(0.68, color=RED, ls="--", lw=1.2, label="metadata-only baseline")
a1.axhline(0.5, color=GRAY, ls=":", lw=1, label="chance")
a1.set_ylim(0.45, 0.95); a1.set_ylabel("Discriminative AUROC")
a1.set_title("(A) Probe: not a resolution artifact\n(survives geometry removal)")
a1.legend(fontsize=8, loc="lower left")
# (B) OT-CFM transport d raw vs residualized, two cohorts
x = np.arange(2); w = 0.35
a2.bar(x-w/2, [2.44, 1.34], w, label="raw", color=BLUE)
a2.bar(x+w/2, [1.90, 0.79], w, label="metadata-residualized", color=GREEN)
a2.axhline(0.58, color=RED, ls="--", lw=1.1, label="perm-null level")
a2.set_xticks(x); a2.set_xticklabels(["fuid", "PCOSGen"])
a2.set_ylabel("OT-CFM transport (Cohen's d)")
a2.set_title("(B) Transport: survives confound removal\n(both cohorts, p<0.001)")
a2.legend(fontsize=8)
fig.suptitle("Generative transport is confound-robust where discriminative probing is fragile",
             fontsize=12, y=1.04)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG/"fig2_confound_robustness.png"); plt.close(fig)

# ---- Fig 3: layer-wise AUROC (deep-layer emergence) ----
try:
    lw = json.load(open(D/"ovary_layerwise_patchloc.json"))["layerwise_auroc"]
    layers = [d["layer"] for d in lw]; au = [d["auroc"] for d in lw]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(layers, au, "-o", color=BLUE, ms=4)
    ax.axhline(0.5, color=GRAY, ls=":", lw=1)
    ax.set_xlabel("DINOv2 transformer block (depth)"); ax.set_ylabel("PCO-vs-DF AUROC")
    ax.set_title("Reserve signal emerges in deep layers")
    fig.savefig(FIG/"fig3_layerwise.png"); plt.close(fig)
except Exception as e:
    print("fig3 skip:", e)

# ---- Fig 4: example ovarian US (PCO vs DF), grayscale ----
base = R/"data/external/ovarian_us/fuid/extracted"
def two(cls):
    ps = [p for p in sorted(glob.glob(f"{base}/**/{cls}/*", recursive=True))
          if p.lower().endswith((".jpg",".png",".jpeg"))]
    return ps[:2]
sel = [("PCO", two("PCO")), ("Dominant_Follicle", two("Dominant_Follicle"))]
fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.4))
for r, (name, ps) in enumerate(sel):
    for c in range(2):
        ax = axes[r][c]
        try:
            ax.imshow(np.asarray(Image.open(ps[c]).convert("L")), cmap="gray")
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0: ax.set_ylabel(name.replace("_","\n"), fontsize=10)
fig.suptitle("Ovarian ultrasound: PCO (high-reserve) vs dominant follicle", fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG/"fig4_us_examples.png"); plt.close(fig)

print("saved figures to", FIG)
for f in sorted(FIG.glob("*.png")):
    print(f"  {f.name}  {f.stat().st_size//1024} KB")
