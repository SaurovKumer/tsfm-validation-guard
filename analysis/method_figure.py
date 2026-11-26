"""Method overview figure (full text width): data split, one fine-tuning run, the interpolation path, the guarded
choice on validation windows and the optional conformal recalibration.
Run: python analysis/method_figure.py      Output: figures/fig_method.pdf
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#f6f5f2"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "pdf.fonttype": 42, "mathtext.fontset": "dejavuserif"})

fig = plt.figure(figsize=(7.2, 2.35))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 33); ax.axis("off")


def box(x, y, w, h, title, body, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=BG, ec=edge, lw=1.0))
    ax.text(x + 1.0, y + h - 1.4, title, fontsize=8, fontweight="bold", color=INK, va="top")
    ax.text(x + 1.0, y + h - 4.6, body, fontsize=7, color=INK2, va="top", linespacing=1.35)


def arrow(x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color))


# ---- (1) data split ----
ax.text(1, 31.5, "(1) Split each series", fontsize=8, fontweight="bold", color=INK, va="top")
y0, h = 22.5, 3.2
segs = [(1, 14, S1, "fine-tuning data"), (15, 7, S4, "$V$ validation"), (22, 7, S2, "$W$ test")]
for x, w, c, lab in segs:
    ax.add_patch(Rectangle((x, y0), w, h, fc=c, ec="white", lw=1.0, alpha=0.85))
    ax.text(x + w / 2, y0 - 1.3, lab, fontsize=6.5, color=INK2, ha="center", va="top")
for k in range(1, 3):
    ax.plot([15 + k * 7 / 3] * 2, [y0, y0 + h], color="white", lw=0.8)
ax.text(1, 15.8, "validation and test windows have\nlength $H$; the test windows are\nnever used for selection", fontsize=6.5,
        color=INK2, va="top", linespacing=1.3)

# ---- (2) fine-tune once, interpolation path ----
box(32, 3, 24, 27.5, "(2) Fine-tune once", "", S1)
ax.text(44, 23.5, r"$\theta_0$ zero-shot $\rightarrow$ LoRA $\rightarrow$ $\theta_1$", fontsize=7, color=INK, ha="center")
px0, px1, py = 36, 52, 15
ax.plot([px0, px1], [py, py], color=INK2, lw=1.2)
for a in [0, 0.1, 0.25, 0.5, 0.75, 1]:
    x = px0 + a * (px1 - px0)
    ax.plot(x, py, "o", ms=4.2, color=S1 if a else S3, mec="white", mew=0.6)
    if a != 0.1:
        ax.text(x, py - 2.0, f"{a:g}", fontsize=6, color=INK2, ha="center", va="top")
ax.text(px0, py + 1.6, r"$\theta_0$", fontsize=7, color=S3, ha="center")
ax.text(px1, py + 1.6, r"$\theta_1$", fontsize=7, color=S1, ha="center")
ax.text(44, 9.2, r"$\theta(\alpha)=(1-\alpha)\,\theta_0+\alpha\,\theta_1$", fontsize=7.5, color=INK, ha="center")
ax.text(44, 5.6, "score every $\\theta(\\alpha)$ on the $V$ windows", fontsize=6.5, color=INK2, ha="center")

# ---- (3) guard ----
box(58.5, 3, 24.5, 27.5, "(3) Validation guard",
    "keep $\\alpha$ only if\n"
    "  $L_V(\\alpha) < (1-m)\\,L_V(0)$\n"
    "  and last window $L_1(\\alpha) \\leq L_1(0)$\n"
    "hourly or finer data: $\\alpha = 0$\n"
    "$\\alpha^* = \\arg\\min$ admissible $L_V$\n"
    "(no admissible $\\alpha$: zero-shot)", S3)

# ---- (4) deploy ----
box(85, 3, 14.3, 27.5, "(4) Deploy",
    "$\\theta(\\alpha^*)$: one model,\nno extra inference\ncost\n\n"
    "if $\\alpha^* > 0$:\nrecalibrate the\nintervals by CQR\non the $V$ windows", S2)

arrow(29.8, 24, 31.7, 24)
arrow(56.3, 16.8, 58.2, 16.8)
arrow(83.3, 16.8, 84.7, 16.8)
fig.savefig(OUT / "fig_method.pdf")
print("->", OUT / "fig_method.pdf")
