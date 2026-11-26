"""Statistical tests on all 72 runs (Chronos-2 + TimesFM 2.5; 28 design/hold-out/fresh datasets + 8 confirmation
configurations; LoRA full, seed 0). No GPU needed.

Per run, the change in test MWQL vs zero-shot (%) is computed for each alpha-selection rule from the alpha curves.
Tests:
  * Wilcoxon signed-rank (Pratt zero handling), two-sided, paired per run:
      rule vs zero-shot, rule vs plain LoRA, A vs +wise.
  * Cluster bootstrap over datasets (both models of a dataset resampled together), 10,000 draws:
      95% CI for the mean change, the geometric-mean MWQL ratio and the share of runs worse than zero-shot (> +1%).
  * Holm correction across the primary family (A, B, +wise, LoRA vs zero-shot; A vs LoRA; A vs +wise).
Run: python analysis/stats_tests.py      Output: results/stats/
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "stats"
OUT.mkdir(exist_ok=True)
COLS = ["dataset", "alpha", "val_MWQL", "val1_MWQL", "test_MWQL"]

chr_ = pd.concat([pd.read_csv(RES / p / "alpha_curves_compact.csv") for p in
                  ["partA", "partB", "holdoutH1", "holdoutH2", "freshF1", "freshF2"]])
chr_ = chr_[chr_.method == "lora_full"][COLS].assign(model="chronos2", stage="earlier")
tfm = pd.concat([pd.read_csv(RES / f"tfm_{p}/alpha_curves_compact.csv") for p in ["A", "BH", "F"]])[COLS]
tfm = tfm.assign(model="timesfm", stage="earlier")
newc = pd.read_csv(RES / "chr_N/alpha_curves.csv").query("method == 'lora_full'")[COLS].assign(model="chronos2", stage="confirm")
newt = pd.read_csv(RES / "tfm_N/alpha_curves.csv").query("method == 'lora_full'")[COLS].assign(model="timesfm", stage="confirm")
C = pd.concat([chr_, tfm, newc, newt]).sort_values(["model", "dataset", "alpha"])


def is_hf(d):
    f = d.split("/")[-1] if "/" in d else ""
    named_hf = {"bizitobs_service", "bizitobs_application", "m4_hourly"}
    return d in named_hf or f.endswith(("H", "T", "S"))


def pick(r, m=0.03, cap=1.0, last=True, hf_zero=False, m_hf=None):
    if hf_zero and r["hf"]:
        return 0.0
    mm = m_hf if (m_hf is not None and r["hf"]) else m
    a, v, v1 = r["a"], r["v"], r["v1"]
    ok = (a <= cap + 1e-9) & (v < (1 - mm) * v[0])
    if last:
        ok &= v1 <= v1[0]
    ok[0] = True
    i = np.where(ok)[0]
    return a[i[np.argmin(v[i])]]


RULES = {"A": dict(m=0.02, hf_zero=True), "B": dict(m=0.02, m_hf=0.10, cap=0.5),
         "guardLF3": dict(m=0.03, hf_zero=True), "wise": dict(m=0.0, last=False)}
rows = []
for (mdl, d), g in C.groupby(["model", "dataset"]):
    r = dict(a=g.alpha.values, v=g.val_MWQL.values, v1=g.val1_MWQL.values, t=g.test_MWQL.values, hf=is_hf(d))
    ch = lambda al: 100 * (r["t"][np.searchsorted(r["a"], al)] / r["t"][0] - 1)
    row = dict(model=mdl, dataset=d, hf=r["hf"], stage=g.stage.iloc[0], LoRA=ch(1.0),
               oracle=100 * (r["t"].min() / r["t"][0] - 1))
    row.update({k: ch(pick(r, **p)) for k, p in RULES.items()})
    rows.append(row)
P = pd.DataFrame(rows)
assert len(P) == 72, len(P)
P.round(4).to_csv(OUT / "per_run_72.csv", index=False)
METHODS = ["LoRA", "A", "B", "guardLF3", "wise", "oracle"]

# ---------------- cluster bootstrap over datasets ----------------
rng = np.random.default_rng(0)
B_DRAWS = 10_000


def stats(x):
    return np.array([x.mean(), 100 * (np.exp(np.log1p(x / 100).mean()) - 1), 100 * (x > 1).mean()])


def boot(df, col):
    codes = pd.factorize(df.dataset)[0]
    idx = [np.where(codes == k)[0] for k in range(codes.max() + 1)]   # positions of each dataset's runs
    x = df[col].values
    est = stats(x)
    draws = np.empty((B_DRAWS, 3))
    for b in range(B_DRAWS):
        pick_ = rng.integers(0, len(idx), len(idx))
        sel = np.concatenate([idx[j] for j in pick_])
        draws[b] = stats(x[sel])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return est, lo, hi


def boot_table(df, label):
    out = []
    for c in METHODS:
        est, lo, hi = boot(df, c)
        out.append(dict(subset=label, method=c, n=len(df),
                        mean=est[0], mean_lo=lo[0], mean_hi=hi[0],
                        geo=est[1], geo_lo=lo[1], geo_hi=hi[1],
                        pct_worse=est[2], worse_lo=lo[2], worse_hi=hi[2],
                        worst=df[c].max(), no_worse=int((df[c] <= 1).sum())))
    return out


subsets = {"all 72": P, "Chronos-2": P[P.model == "chronos2"], "TimesFM 2.5": P[P.model == "timesfm"],
           "low freq.": P[~P.hf], "high freq.": P[P.hf], "confirmation 16": P[P.stage == "confirm"]}
BT = pd.DataFrame(sum((boot_table(df, k) for k, df in subsets.items()), []))
BT.round(3).to_csv(OUT / "bootstrap_ci.csv", index=False)

# ---------------- Wilcoxon signed-rank tests ----------------
def wtest(x, y=None):
    d = x if y is None else x - y
    if np.allclose(d, 0):
        return np.nan, np.nan
    s = wilcoxon(d, zero_method="pratt", alternative="two-sided")
    return s.statistic, s.pvalue


TESTS = [("A vs zero-shot", "A", None), ("B vs zero-shot", "B", None), ("+wise vs zero-shot", "wise", None),
         ("LoRA vs zero-shot", "LoRA", None), ("A vs LoRA", "A", "LoRA"), ("A vs +wise", "A", "wise"),
         ("A vs guard-LF m=3%", "A", "guardLF3")]
wrows = []
for label, df in subsets.items():
    for name, a, b in TESTS:
        x = df[a].values
        y = None if b is None else df[b].values
        d = x if y is None else x - y
        stat, p = wtest(x, y)
        wrows.append(dict(subset=label, test=name, n=len(d), n_nonzero=int((np.abs(d) > 1e-9).sum()),
                          median_diff=np.median(d), mean_diff=d.mean(), W=stat, p=p))
W = pd.DataFrame(wrows)
# Holm over the primary family (all 72 runs, first six tests)
prim = W[(W.subset == "all 72") & W.test.isin([t[0] for t in TESTS[:6]])].sort_values("p")
m = len(prim)
adj, run = [], 0.0
for k, p in enumerate(prim.p.values):
    run = max(run, min(1.0, (m - k) * p))
    adj.append(run)
W.loc[prim.index, "p_holm"] = adj
W.round(5).to_csv(OUT / "wilcoxon.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30)
print("=== cluster bootstrap 95% CI (change in test MWQL vs zero-shot, %) ===")
show = BT.copy()
for k in ["mean", "geo"]:
    show[k] = show.apply(lambda r: f"{r[k]:+.2f} [{r[k + '_lo']:+.2f}, {r[k + '_hi']:+.2f}]", axis=1)
show["worse%"] = show.apply(lambda r: f"{r.pct_worse:.0f} [{r.worse_lo:.0f}, {r.worse_hi:.0f}]", axis=1)
print(show[["subset", "method", "n", "mean", "geo", "worse%", "worst", "no_worse"]].to_string(index=False))
print("\n=== Wilcoxon signed-rank (Pratt), two-sided ===")
print(W.round(5).to_string(index=False))
