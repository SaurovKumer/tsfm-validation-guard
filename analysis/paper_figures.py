"""Figures for the paper (PDF, print size for an Elsevier double-column page).

Run after stats_tests.py and confirm_N.py:  python analysis/paper_figures.py   Output: figures/
Colours: fixed categorical order of the validated reference palette (blue, orange, aqua, yellow, ...).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.5, "pdf.fonttype": 42})
COL1 = 3.5   # one column width in inches (cas-dc)

# ---------------- Fig. 1: alpha curves for four representative runs ----------------
COLS = ["dataset", "alpha", "val_MWQL", "test_MWQL"]
chr_ = pd.concat([pd.read_csv(RES / p / "alpha_curves_compact.csv") for p in
                  ["partA", "partB", "holdoutH1", "holdoutH2", "freshF1", "freshF2"]])
chr_ = pd.concat([chr_[chr_.method == "lora_full"][COLS],
                  pd.read_csv(RES / "chr_N/alpha_curves.csv")[COLS]]).assign(model="Chronos-2")
tfm = pd.concat([pd.read_csv(RES / f"tfm_{p}/alpha_curves_compact.csv")[COLS] for p in ["A", "BH", "F"]] +
                [pd.read_csv(RES / "tfm_N/alpha_curves.csv")[COLS]]).assign(model="TimesFM 2.5")
C = pd.concat([chr_, tfm])
cases = [("Chronos-2", "us_births/D", "(a) gain kept"), ("TimesFM 2.5", "solar/D", "(b) harm avoided"),
         ("Chronos-2", "bizitobs_service", "(c) validation misled"), ("TimesFM 2.5", "M_DENSE/H", "(d) gain missed")]
fig, axs = plt.subplots(2, 2, figsize=(COL1, 2.9), sharex=True)
for ax, (m, d, title) in zip(axs.flat, cases):
    g = C[(C.model == m) & (C.dataset == d)].sort_values("alpha")
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.plot(g.alpha, 100 * (g.val_MWQL / g.val_MWQL.iloc[0] - 1), color=S1, ls="--", marker="o", ms=3, label="validation")
    ax.plot(g.alpha, 100 * (g.test_MWQL / g.test_MWQL.iloc[0] - 1), color=S2, marker="s", ms=3, label="test")
    ax.set_title(f"{title}\n{m}, {d}", fontsize=7, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5)
for ax in axs[1]:
    ax.set_xlabel(r"interpolation weight $\alpha$")
for ax in axs[:, 0]:
    ax.set_ylabel("MWQL change (%)")
axs[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_alpha_curves.pdf")
plt.close(fig)

# ---------------- Fig. 2: risk and return of the selection methods (108 runs, cluster-bootstrap 95% CIs) ----------------
B = pd.read_csv(RES / "stats108/bootstrap_ci.csv").query("subset == 'all'").set_index("method")
order = ["LoRA", "ES", "wise", "ESguard", "Joint", "A", "B", "oracle"]
lab = {"LoRA": r"LoRA ($\alpha$ = 1)", "ES": "early stopping", "wise": r"+wise (argmin $\alpha$)",
       "ESguard": "early stopping + guard", "Joint": r"step $\times$ $\alpha$ + guard", "A": "guard-LF (A)",
       "B": "soft-frequency guard (B)", "oracle": r"oracle (best $\alpha$ on test)"}
col = {"LoRA": S2, "ES": S2, "wise": S4, "ESguard": S3, "Joint": S3, "A": S1, "B": S1, "oracle": INK2}
y = np.arange(len(order))[::-1]
fig, axs = plt.subplots(1, 2, figsize=(COL1, 2.4), sharey=True, gridspec_kw=dict(width_ratios=[1, 1]))
for ax, (est, lo, hi, xl) in zip(axs, [("mean", "lo", "hi", "mean change vs\nzero-shot (%)"),
                                      ("worse", "worse_lo", "worse_hi", "runs worse than\nzero-shot (%)")]):
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    for yi, k in zip(y, order):
        r = B.loc[k]
        ax.plot([r[lo], r[hi]], [yi, yi], color=col[k], lw=1.5, solid_capstyle="round")
        ax.plot(r[est], yi, "o", color=col[k], ms=5, mec="white", mew=0.8)
    ax.set_xlabel(xl)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.tick_params(axis="y", length=0)
axs[0].set_yticks(y, [lab[k] for k in order], fontsize=6.5)
axs[0].set_xlim(-8, 48)
axs[1].set_xlim(-5, 80)
axs[0].spines["left"].set_visible(False); axs[1].spines["left"].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_risk_return.pdf")
plt.close(fig)

# ---------------- Fig. 3: coverage, zero-shot vs LoRA, and LoRA before vs after CQR (108 runs) ----------------
V = pd.read_csv(RES / "stats108/coverage_108.csv")
W = V.pivot_table(index=["model", "dataset"], columns="at", values=["cov80", "cqr_cov80"]).reset_index()
fig, axs = plt.subplots(1, 2, figsize=(COL1, 2.05))
panels = [(("cov80", "a=0"), ("cov80", "a=1"), r"zero-shot ($\alpha$=0)", r"LoRA ($\alpha$=1)", "(a) effect of fine-tuning"),
          (("cov80", "a=1"), ("cqr_cov80", "a=1"), r"LoRA, raw", "LoRA, after CQR", "(b) recalibration")]
for ax, (xc, yc, xl, yl, title) in zip(axs, panels):
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
    ax.axvline(0.8, color=GRID, ls=":", lw=1)
    ax.axhline(0.8, color=GRID, ls=":", lw=1)
    for m, c, mkr in [("Chronos-2", S1, "o"), ("TimesFM 2.5", S2, "s"), ("Chronos-Bolt", S3, "^")]:
        g = W[W.model == m]
        ax.scatter(g[xc], g[yc], color=c, marker=mkr, s=12, edgecolor="white", lw=0.5, label=m, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.4, 0.8]); ax.set_yticks([0, 0.4, 0.8])
    ax.set_xlabel(xl, fontsize=7)
    ax.set_ylabel(yl, fontsize=7)
    ax.set_title(title, fontsize=7, color=INK, loc="left")
axs[0].legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_coverage.pdf")
plt.close(fig)
print("figures ->", OUT)
