"""Paper statistics on all 108 runs (Chronos-2, TimesFM 2.5, Chronos-Bolt x 36 configurations; LoRA, seed 0).

Source: the §22 notebooks (results/{chr,tfm,bolt}_E{1,2,3}/alpha_curves.csv). Their 4000-step alpha curves reproduce
the earlier runs exactly (§22.2), and they add the checkpoint rows needed for the early-stopping methods.
Per run, the change in test MWQL vs zero-shot (%) for:
  LoRA (alpha=1), +wise, guard-LF m=3%, A, B, ES, ES+guard, Joint, oracle (best alpha on test).
Tests (as in stats_tests.py, now on 108 runs):
  * cluster bootstrap over configurations (all three models of a configuration resampled together), 10,000 draws;
  * Wilcoxon signed-rank (Pratt), two-sided; Holm over the primary family
    (A, B, +wise, LoRA vs zero-shot; A vs LoRA; A vs +wise).
Also: coverage / CQR at alpha=0, alpha=1 and the alpha chosen by A, for the paper's calibration table.
Run: python analysis/stats108.py      Output: results/stats108/
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "stats108"
OUT.mkdir(exist_ok=True)
MODELS = {"Chronos-2": "chr", "TimesFM 2.5": "tfm", "Chronos-Bolt": "bolt"}
HF = {"bizitobs_l2c/H", "bizitobs_service", "bizitobs_application", "m4_hourly", "SZ_TAXI/H",
      "electricity/H", "electricity/15T", "bitbrains_rnd/H", "bitbrains_fast_storage/H"}
JOINT_ALPHAS = [0.25, 0.5, 0.75, 1.0]


def is_hf(d):
    return d in HF or ("/" in d and d.split("/")[-1].endswith(("H", "T", "S")))


def guard(c, c0, m=0.02, hf=False, hf_zero=True, m_hf=None, cap_hf=None, last=True):
    if hf and hf_zero:
        return c0
    mm = m_hf if (hf and m_hf is not None) else m
    ok = c[c.val_MWQL < (1 - mm) * c0.val_MWQL]
    if last:
        ok = ok[ok.val1_MWQL <= c0.val1_MWQL]
    if hf and cap_hf is not None:
        ok = ok[ok.alpha <= cap_hf + 1e-9]
    if ok.empty:
        return c0
    return ok.sort_values(["val_MWQL", "step", "alpha"]).iloc[0]


rows, cov = [], []
for model, tag in MODELS.items():
    df = pd.concat([pd.read_csv(RES / f"{tag}_E{i}" / "alpha_curves.csv") for i in (1, 2, 3)])
    df = df[df.method == "lora_full"].drop_duplicates(["dataset", "step", "alpha"], keep="last")
    for d, g in df.groupby("dataset"):
        fin = g[g.step == g.step.max()].sort_values("alpha")
        c0 = fin[fin.alpha == 0.0].iloc[0]
        hf = is_hf(d)
        at1, joint, curve = g[g.alpha == 1.0], g[g.alpha.isin(JOINT_ALPHAS)], fin[fin.alpha > 0]
        sel = {
            "LoRA": fin[fin.alpha == 1.0].iloc[0],
            "wise": curve.sort_values(["val_MWQL", "alpha"]).iloc[0] if curve.val_MWQL.min() < c0.val_MWQL else c0,
            "guardLF3": guard(curve, c0, m=0.03, hf=hf),
            "A": guard(curve, c0, hf=hf),
            "B": guard(curve, c0, hf=hf, hf_zero=False, m_hf=0.10, cap_hf=0.5),
            "ES": at1.sort_values(["val_MWQL", "step"]).iloc[0],
            "ESguard": guard(at1, c0, hf=hf),
            "Joint": guard(joint, c0, hf=hf),
            "oracle": fin.sort_values("test_MWQL").iloc[0],
        }
        row = dict(model=model, dataset=d, hf=hf)
        row.update({k: 100 * (r.test_MWQL / c0.test_MWQL - 1) for k, r in sel.items()})
        row.update({f"alpha_{k}": float(r.alpha) for k, r in sel.items()})
        rows.append(row)
        for lab, r in [("a=0", c0), ("a=1", sel["LoRA"]), ("A", sel["A"]), ("ES+guard", sel["ESguard"])]:
            cov.append(dict(model=model, dataset=d, hf=hf, at=lab, cov80=r.test_cov80, width80=r.test_width80,
                            cov60=r.test_cov60, cqr_cov80=r.test_cqr_cov80, cqr_width80=r.test_cqr_width80))
P = pd.DataFrame(rows)
assert len(P) == 108, len(P)
P.round(4).to_csv(OUT / "per_run_108.csv", index=False)
V = pd.DataFrame(cov)
V.round(4).to_csv(OUT / "coverage_108.csv", index=False)
METHODS = ["LoRA", "wise", "guardLF3", "A", "B", "ES", "ESguard", "Joint", "oracle"]

rng = np.random.default_rng(0)
B_DRAWS = 10_000


def boot(df, col):
    codes = pd.factorize(df.dataset)[0]
    idx = [np.where(codes == k)[0] for k in range(codes.max() + 1)]
    x = df[col].values
    f = lambda v: np.array([v.mean(), 100 * (v > 1).mean()])
    est = f(x)
    draws = np.array([f(x[np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])]) for _ in range(B_DRAWS)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return est, lo, hi


subsets = {"all": P, "Chronos-2": P[P.model == "Chronos-2"], "TimesFM 2.5": P[P.model == "TimesFM 2.5"],
           "Chronos-Bolt": P[P.model == "Chronos-Bolt"], "LF": P[~P.hf], "HF": P[P.hf]}
bt = []
for lab, df in subsets.items():
    for c in METHODS:
        est, lo, hi = boot(df, c)
        bt.append(dict(subset=lab, method=c, n=len(df), mean=est[0], lo=lo[0], hi=hi[0], worse=est[1],
                       worse_lo=lo[1], worse_hi=hi[1], worst=df[c].max(), no_worse=int((df[c] <= 1).sum()),
                       better=int((df[c] < -1).sum())))
BT = pd.DataFrame(bt)
BT.round(3).to_csv(OUT / "bootstrap_ci.csv", index=False)


def wtest(d):
    if np.allclose(d, 0):
        return np.nan
    return wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue


TESTS = [("A vs zero-shot", "A", None), ("B vs zero-shot", "B", None), ("+wise vs zero-shot", "wise", None),
         ("LoRA vs zero-shot", "LoRA", None), ("A vs LoRA", "A", "LoRA"), ("A vs +wise", "A", "wise"),
         ("A vs guard-LF m=3%", "A", "guardLF3"), ("A vs ES+guard", "A", "ESguard"), ("A vs ES", "A", "ES")]
wr = []
for lab, df in subsets.items():
    for name, a, b in TESTS:
        d = df[a].values - (0 if b is None else df[b].values)
        wr.append(dict(subset=lab, test=name, n=len(d), mean_diff=d.mean(), p=wtest(d)))
W = pd.DataFrame(wr)
prim = W[(W.subset == "all") & W.test.isin([t[0] for t in TESTS[:6]])].sort_values("p")
adj, run = [], 0.0
for k, p in enumerate(prim.p.values):
    run = max(run, min(1.0, (len(prim) - k) * p))
    adj.append(run)
W.loc[prim.index, "p_holm"] = adj
W.to_csv(OUT / "wilcoxon.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30)
print("=== bootstrap (mean change %, 95% CI; % runs worse than zero-shot) ===")
for lab in subsets:
    t = BT[BT.subset == lab].set_index("method")
    print(f"\n[{lab}] n={int(t.n.iloc[0])}")
    for c in METHODS:
        r = t.loc[c]
        print(f"  {c:9s} {r['mean']:+7.2f} [{r.lo:+6.2f},{r.hi:+6.2f}]  worse {r.worse:5.1f}% [{r.worse_lo:4.1f},{r.worse_hi:4.1f}]"
              f"  worst {r.worst:+8.2f}  no-worse {int(r.no_worse)}/{int(r.n)}  better {int(r.better)}")
print("\n=== Wilcoxon (all 108; Holm over first six) ===")
print(W[W.subset == "all"][["test", "mean_diff", "p", "p_holm"]].to_string(index=False))
print("\nper model, A vs zero-shot / A vs LoRA:")
print(W[(W.subset != "all") & W.test.isin(["A vs zero-shot", "A vs LoRA", "A vs ES+guard"])][["subset", "test", "mean_diff", "p"]].to_string(index=False))
oa = P.oracle.mean()
print(f"\nshare of oracle gain: A {P.A.mean() / oa:.2f}, B {P.B.mean() / oa:.2f}, wise {P.wise.mean() / oa:.2f}")
print("\n=== coverage (108 runs, test windows; nominal 0.80) ===")
V["dev"] = (V.cov80 - 0.8).abs()
V["cqr_dev"] = (V.cqr_cov80 - 0.8).abs()
print(V.groupby("at")[["cov80", "cqr_cov80", "dev", "cqr_dev", "cov60"]].mean().round(3).to_string())
print(V.groupby("at")[["width80", "cqr_width80"]].median().round(3).rename(columns=lambda c: "median " + c).to_string())
print(V.groupby(["model", "at"])[["cov80", "cqr_cov80"]].mean().round(3).to_string())
w = V.pivot_table(index=["model", "dataset"], columns="at", values="cov80")
print("coverage fell > 0.05 from a=0 to a=1:", int(((w["a=0"] - w["a=1"]) > 0.05).sum()), "of", len(w),
      "| rose > 0.05:", int(((w["a=1"] - w["a=0"]) > 0.05).sum()))
print("CQR NaN rows:", int(V.cqr_cov80.isna().sum()))
