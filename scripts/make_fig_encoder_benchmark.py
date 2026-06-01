"""Publication figure for the unified 10-encoder recoverability benchmark.

Reads results/diagnostics/unified_encoder_benchmark.json directly (no hardcoded
result numbers) and renders a dot-and-CI (forest-style) plot:
  - left group: 10 frozen encoders, embryo morphology -> serum AMH tertile (the null)
  - right group: ovary-US FUID reserve phenotype positive control

Design: clean, muted, single-accent. No dark/neon, no monospace-everywhere,
no decorative elements. Honest null shown as null.
Outputs figures/fig_encoder_benchmark.png.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

JSON = "results/diagnostics/unified_encoder_benchmark.json"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

d = json.load(open(JSON))

# --- collect embryo encoders, sorted by R/H ascending (most negative at bottom) ---
emb_items = []
for name, m in d["embryo_amh"]["encoders"].items():
    if m.get("R/H") is None:
        continue
    ci = m.get("bootstrap_CI") or [None, None]
    emb_items.append((name, m["R/H"], ci[0], ci[1]))
emb_items.sort(key=lambda r: r[1])  # ascending

ov_items = []
for name, m in d["ovary_fuid_phenotype"]["encoders"].items():
    if m.get("R/H") is None:
        continue
    ci = m.get("bootstrap_CI") or [None, None]
    ov_items.append((name, m["R/H"], ci[0], ci[1]))
ov_items.sort(key=lambda r: r[1])

n_emb = len(emb_items)
n_ov = len(ov_items)

# muted palette
COL_NULL = "#5b6b7a"     # slate for null embryo points
COL_POS = "#2f7d5b"      # muted green for positive control
COL_ZERO = "#b3382c"     # muted red zero reference
GRID = "#dfe3e8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#9aa3ad",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(8.2, 6.4))

# y positions: embryo block at bottom, a gap, ovary block on top
y_emb = list(range(n_emb))
gap = 1.5
y_ov = [n_emb - 1 + gap + 1 + i for i in range(n_ov)]

# zero reference line
ax.axvline(0.0, color=COL_ZERO, lw=1.1, ls="--", zorder=1, alpha=0.8)

# embryo points + CI
for y, (name, rh, lo, hi) in zip(y_emb, emb_items):
    if lo is not None and hi is not None:
        ax.plot([lo, hi], [y, y], color=COL_NULL, lw=2.0, alpha=0.55, zorder=2,
                solid_capstyle="round")
    ax.plot(rh, y, "o", color=COL_NULL, ms=7, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8)

# ovary positive-control points + CI
for y, (name, rh, lo, hi) in zip(y_ov, ov_items):
    if lo is not None and hi is not None:
        ax.plot([lo, hi], [y, y], color=COL_POS, lw=2.4, alpha=0.6, zorder=2,
                solid_capstyle="round")
    ax.plot(rh, y, "D", color=COL_POS, ms=8, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8)

# y tick labels
all_y = y_emb + y_ov
all_labels = [n for n, *_ in emb_items] + [f"{n}" for n, *_ in ov_items]
ax.set_yticks(all_y)
ax.set_yticklabels(all_labels)

# group bracket labels on the right
y_emb_mid = sum(y_emb) / len(y_emb)
y_ov_mid = sum(y_ov) / len(y_ov)
xr = ax.get_xlim()
ax.text(1.012, y_emb_mid / (max(all_y)), "Embryo morphology\n-> serum AMH tertile",
        transform=ax.get_yaxis_transform(), va="center", ha="left",
        fontsize=9.5, color=COL_NULL, fontweight="bold")
ax.text(1.012, y_ov_mid, "Ovary US\n-> reserve phenotype\n(positive control)",
        transform=ax.get_yaxis_transform(), va="center", ha="left",
        fontsize=9.5, color=COL_POS, fontweight="bold")

# light horizontal separator between groups
sep_y = (max(y_emb) + min(y_ov)) / 2
ax.axhline(sep_y, color=GRID, lw=1.0, zorder=0)

ax.set_xlabel("Recoverability  R/H  (held-out V-information, fraction of target entropy)")
n = d["embryo_amh"]["n"]; g = d["embryo_amh"]["n_groups"]
fig.suptitle(
    "Frozen foundation-model recoverability of ovarian-reserve signal",
    fontsize=12, x=0.5, y=0.975, ha="center")
ax.set_title(
    f"embryo morphology -> serum AMH is null across {n_emb} encoders "
    f"(n={n}, {g} patients);\novary-US phenotype is recovered (positive control)",
    fontsize=9.5, loc="center", pad=8, color="#444444")

ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.margins(y=0.04)
ax.set_xlim(left=min(-0.05, xr[0]))

fig.subplots_adjust(left=0.21, right=0.79, top=0.84, bottom=0.10)
png = os.path.join(OUTDIR, "fig_encoder_benchmark.png")
fig.savefig(png, dpi=170)
print("wrote", png)
