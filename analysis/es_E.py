"""§22: interpolation vs early stopping (pre-registered 2026-11-24; written before any E result was read).

Input: results/{chr,tfm,bolt}_E{1,2,3}/alpha_curves.csv from the tsfm-*-E notebooks. Rows with the final step carry
the usual alpha curve (0, .1, .25, .5, .75, 1); rows with step 500/1000/2000 carry alpha in {.25, .5, .75, 1}.
Methods (validation only; HF = §21 definition; guard = rule A: m = 2%, last window not worse, HF -> zero-shot):
  ES          argmin pooled validation over steps, alpha = 1
  ES+guard    guard over {zero-shot} + steps at alpha = 1
  A           guard over the final-step alpha curve (as before)
  Joint       guard over {zero-shot} + all (step, alpha in {.25,.5,.75,1})
Criteria (108 runs pooled): P1 ES worst > +10% and ES no-worse count < A's; P2 A vs ES+guard; P3 Joint vs A.
Run: python analysis/es_E.py      Output: results/es_E/
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path(os.environ.get("TSFM_RES", Path(__file__).resolve().parents[1] / "results"))
OUT = RES / "es_E"
OUT.mkdir(exist_ok=True)
MODELS = {"Chronos-2": "chr", "TimesFM 2.5": "tfm", "Chronos-Bolt": "bolt"}
HF = {"bizitobs_l2c/H", "bizitobs_service", "bizitobs_application", "m4_hourly", "SZ_TAXI/H",
      "electricity/H", "electricity/15T", "bitbrains_rnd/H", "bitbrains_fast_storage/H"}
JOINT_ALPHAS = [0.25, 0.5, 0.75, 1.0]
M = 0.02


def is_hf(d):
    return d in HF or ("/" in d and d.split("/")[-1].endswith(("H", "T", "S")))


def guard(c, c0, m=M, hf=False, hf_zero=True, m_hf=None, cap_hf=None):
    """c: candidate rows (DataFrame); c0: the zero-shot row. Returns the selected row."""
    if hf and hf_zero:
        return c0
    mm = m_hf if (hf and m_hf is not None) else m
    ok = c[(c.val_MWQL < (1 - mm) * c0.val_MWQL) & (c.val1_MWQL <= c0.val1_MWQL)]
    if hf and cap_hf is not None:
        ok = ok[ok.alpha <= cap_hf + 1e-9]
    if ok.empty:
        return c0
    return ok.sort_values(["val_MWQL", "step", "alpha"]).iloc[0]


def load(tag):
    parts = [RES / f"{tag}_E{i}" / "alpha_curves.csv" for i in (1, 2, 3)]
    have = [p for p in parts if p.exists()]
    if not have:
        return None
    df = pd.concat([pd.read_csv(p) for p in have])
    df = df[df.method == "lora_full"].drop_duplicates(["dataset", "step", "alpha"], keep="last")
    return df, len(have)


rows, sel_rows = [], []
for model, tag in MODELS.items():
    got = load(tag)
    if got is None:
        print(f"{model}: no E results yet")
        continue
    df, n_parts = got
    print(f"{model}: {n_parts}/3 parts, {df.dataset.nunique()} configurations")
    for d, g in df.groupby("dataset"):
        final = g.step.max()
        fin = g[g.step == final].sort_values("alpha")
        c0 = fin[fin.alpha == 0.0].iloc[0]
        zs = c0.test_MWQL
        hf = is_hf(d)
        ch = lambda r: 100 * (r.test_MWQL / zs - 1)
        at1 = g[g.alpha == 1.0]
        joint = g[g.alpha.isin(JOINT_ALPHAS)]
        sel = {
            "LoRA (4000, a=1)": fin[fin.alpha == 1.0].iloc[0],
            "ES": at1.sort_values(["val_MWQL", "step"]).iloc[0],
            "ES+guard": guard(at1, c0, hf=hf),
            "A": guard(fin[fin.alpha > 0], c0, hf=hf),
            "Joint": guard(joint, c0, hf=hf),
            "ES+guard (no HF rule)": guard(at1, c0, hf=hf, hf_zero=False),
            "Joint (no HF rule)": guard(joint, c0, hf=hf, hf_zero=False),
            "Joint (B-style HF)": guard(joint, c0, hf=hf, hf_zero=False, m_hf=0.10, cap_hf=0.5),
            "oracle-step": pd.concat([at1, c0.to_frame().T]).sort_values("test_MWQL").iloc[0],
            "oracle-joint": pd.concat([joint, c0.to_frame().T]).sort_values("test_MWQL").iloc[0],
        }
        row = dict(model=model, dataset=d, hf=hf)
        for k, r in sel.items():
            row[k] = ch(r)
            sel_rows.append(dict(model=model, dataset=d, method=k, step=int(r.step), alpha=float(r.alpha),
                                 change=ch(r), test_cov80=r.test_cov80, test_cqr_cov80=r.test_cqr_cov80))
        row["eval_sec_final_curve"] = fin.eval_sec.sum()
        row["eval_sec_extra_steps"] = g[g.step != final].eval_sec.sum()
        rows.append(row)

if not rows:
    raise SystemExit
P = pd.DataFrame(rows)
S = pd.DataFrame(sel_rows)
P.round(4).to_csv(OUT / "per_run.csv", index=False)
S.round(4).to_csv(OUT / "selected.csv", index=False)

meths = ["LoRA (4000, a=1)", "ES", "ES+guard", "A", "Joint", "ES+guard (no HF rule)", "Joint (no HF rule)",
         "Joint (B-style HF)", "oracle-step", "oracle-joint"]
summ = pd.DataFrame({c: dict(n=len(P), mean=P[c].mean(), median=P[c].median(), worst=P[c].max(),
                             no_worse=int((P[c] <= 1).sum()), better=int((P[c] < -1).sum())) for c in meths}).T
summ.round(3).to_csv(OUT / "summary.csv")
pd.set_option("display.width", 250)
print("\n=== summary (change in test MWQL vs zero-shot, %) ===")
print(summ.round(2).to_string())
print("\n=== per model (mean / worst) ===")
for c in meths:
    print(f"{c:24s}", "  ".join(f"{m}: {g[c].mean():+.2f}/{g[c].max():+.2f}" for m, g in P.groupby("model")),
          "  ", "  ".join(f"{'HF' if h else 'LF'}: {g[c].mean():+.2f}/{g[c].max():+.2f}" for h, g in P.groupby("hf")))


def one_sided(x, y):
    """p-value of H1: x < y (paired), Wilcoxon signed-rank with zero differences dropped."""
    dlt = (x - y).values
    if np.allclose(dlt, 0):
        return 1.0
    return wilcoxon(x, y, alternative="less", zero_method="wilcox").pvalue


n = len(P)
print(f"\n=== pre-registered comparisons (§22; n = {n} runs{'' if n == 108 else ' — INCOMPLETE, not the final test'}) ===")
es, a, esg, jt = P["ES"], P["A"], P["ES+guard"], P["Joint"]
p1 = es.max() > 10 and (es <= 1).sum() < (a <= 1).sum()
print(f"P1 ES worst {es.max():+.2f}% (need > +10), no worse {(es <= 1).sum()}/{n} vs A {(a <= 1).sum()}/{n}: "
      f"{'SUPPORTED' if p1 else 'NOT SUPPORTED'}")
pa, pe = one_sided(a, esg), one_sided(esg, a)
pj = one_sided(jt, a)
raw = {"P2 A<ES+guard": pa, "P2 ES+guard<A": pe, "P3 Joint<A": pj}
order = sorted(raw, key=raw.get)          # Holm over the three one-sided tests
holm, run_max = {}, 0.0
for i, k in enumerate(order):
    run_max = max(run_max, min(1.0, (len(order) - i) * raw[k]))
    holm[k] = run_max
diff = a.mean() - esg.mean()
if diff < 0 and holm["P2 A<ES+guard"] < 0.05:
    v2 = "interpolation better"
elif abs(diff) < 0.5 and holm["P2 A<ES+guard"] >= 0.05 and holm["P2 ES+guard<A"] >= 0.05:
    v2 = "equivalent"
else:
    v2 = "early stopping better" if diff > 0 else "inconclusive (A lower mean, not significant, |diff| >= 0.5)"
print(f"P2 mean A {a.mean():+.3f} vs ES+guard {esg.mean():+.3f} (diff {diff:+.3f} pp); "
      f"Holm p(A<ESg) {holm['P2 A<ES+guard']:.4f}, p(ESg<A) {holm['P2 ES+guard<A']:.4f} -> {v2}")
p3 = (jt.mean() <= a.mean() - 0.5) and holm["P3 Joint<A"] < 0.05 and jt.max() <= a.max() + 1
print(f"P3 Joint mean {jt.mean():+.3f} vs A {a.mean():+.3f}; worst {jt.max():+.2f} vs {a.max():+.2f}; "
      f"Holm p {holm['P3 Joint<A']:.4f} -> {'SUPPORTED' if p3 else 'NOT SUPPORTED'}")

print("\n=== secondary ===")
fail = P[(P.model == "Chronos-Bolt") & P.dataset.isin(["covid_deaths", "kdd_cup_2018_with_missing/D"])]
if len(fail):
    print("Bolt rule-A failures under each method:\n", fail[["dataset", "A", "ES+guard", "Joint", "ES"]].round(2).to_string(index=False))
st = S[S.method.isin(["ES", "ES+guard", "A", "Joint"])]
print("\nselected step (share of runs):\n", pd.crosstab(st.method, st.step, normalize="index").round(2).to_string())
print("\ncoverage at the selected point (mean test cov80 raw / CQR):\n",
      st.groupby("method")[["test_cov80", "test_cqr_cov80"]].mean().round(3).to_string())
print(f"\ncompute: evaluation of the final alpha curve {P.eval_sec_final_curve.sum() / 3600:.2f} h, "
      f"extra checkpoints {P.eval_sec_extra_steps.sum() / 3600:.2f} h (all runs)")

# reproducibility of the final-step curve against the earlier runs (alpha = 1 change, same seed and settings)
OLD = {"chr": ["chr_C1", "chr_C2", "chr_N"], "tfm": ["tfm_A", "tfm_BH", "tfm_F", "tfm_N"], "bolt": ["bolt_B1", "bolt_B2"]}
rep = []
for model, tag in MODELS.items():
    olds = [pd.read_csv(RES / p / "alpha_curves.csv") for p in OLD[tag] if (RES / p / "alpha_curves.csv").exists()]
    if not olds:
        continue
    old = pd.concat(olds)
    old = old[old.method == "lora_full"].drop_duplicates(["dataset", "alpha"], keep="last")
    for _, r in P[P.model == model].iterrows():
        o = old[old.dataset == r.dataset].sort_values("alpha")
        if len(o):
            rep.append(dict(model=model, dataset=r.dataset, old=100 * (o.test_MWQL.iloc[-1] / o.test_MWQL.iloc[0] - 1),
                            new=r["LoRA (4000, a=1)"]))
R = pd.DataFrame(rep)
if len(R):
    R["diff"] = R.new - R.old
    print("\nreproducibility of LoRA (a=1) change vs earlier runs (pp): median |diff| per model\n",
          R.groupby("model")["diff"].apply(lambda x: x.abs().median()).round(3).to_string())
    R.round(4).to_csv(OUT / "reproducibility.csv", index=False)
