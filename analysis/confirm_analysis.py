"""Confirmatory analysis of the pre-registered '+guard' rule (both_improve, m=3%) on LoRA full, seeds 1-2.

Seed 0 (exploratory) values for +guard are recomputed from the seed-0 alpha curves with the same rule.
Run from D:\\01 research:  python analysis/confirm_analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
conf = pd.concat([pd.read_csv(R / p / "results.csv") for p in ("confirmA", "confirmB")])
zs = conf[conf.method == "zero_shot"].set_index("dataset").MWQL
sn = conf[conf.method == "seasonal_naive"].set_index("dataset").MWQL

# seed 0 (exploratory run): same rule applied to the seed-0 alpha curves
cur0 = pd.concat([pd.read_csv(R / p / "alpha_curves_compact.csv") for p in ("partA", "partB")])
cur0 = cur0[cur0.method == "lora_full"]
res0 = pd.concat([pd.read_csv(R / p / "results.csv") for p in ("partA", "partB")])
rows0 = []
for d, g in cur0.groupby("dataset"):
    g = g.sort_values("alpha")
    v0, v10 = g.val_MWQL.iloc[0], g.val1_MWQL.iloc[0]
    ok = g[(g.alpha == 0) | ((g.val_MWQL < 0.97 * v0) & (g.val1_MWQL <= v10))]
    best = ok.sort_values(["val_MWQL", "alpha"]).iloc[0]
    rows0.append(dict(dataset=d, method="lora_full+guard", seed=0, alpha=best.alpha, MWQL=best.test_MWQL))
for m in ("lora_full", "lora_full+wise", "lora_full+wise1"):
    r = res0[res0.method == m][["dataset", "method", "seed", "alpha", "MWQL"]]
    rows0 += r.to_dict("records")
seed0 = pd.DataFrame(rows0)

methods = ["lora_full", "lora_full+wise", "lora_full+wise1", "lora_full+guard"]
allr = pd.concat([conf[conf.method.isin(methods)][["dataset", "method", "seed", "alpha", "MWQL"]], seed0])
allr["change"] = 100 * (allr.MWQL / allr.dataset.map(zs) - 1)
allr["MWQL_n"] = allr.MWQL / allr.dataset.map(sn)
allr.to_csv(R / "confirm_per_run.csv", index=False)


def stats(df):
    return pd.Series(dict(n=len(df), no_worse=int((df.change <= 1.0).sum()),
                          pct_no_worse=100 * (df.change <= 1.0).mean(), better=int((df.change < -1.0).sum()),
                          mean_change=df.change.mean(), median_change=df.change.median(),
                          worst=df.change.max(), best=df.change.min(), mean_MWQL_n=df.MWQL_n.mean()))


pd.set_option("display.width", 200)
print("=== CONFIRMATORY: seeds 1-2 (24 runs per method) ===")
c = allr[allr.seed.isin([1, 2])]
T = c.groupby("method").apply(stats).loc[methods]
print(T.round(2).to_string())
g = T.loc["lora_full+guard"]
crit1, crit2 = g.pct_no_worse >= 80, g.mean_change <= 0
print(f"\nPre-registered criteria for +guard: no-worse {g.pct_no_worse:.1f}% (>=80%: {'PASS' if crit1 else 'FAIL'}), "
      f"mean change {g.mean_change:+.2f}% (<=0: {'PASS' if crit2 else 'FAIL'})")
zs_n = (zs / sn).mean()
print(f"zero-shot mean MWQL_n over the 12 datasets: {zs_n:.4f}")

print("\n=== Seed 0 (exploratory) for comparison ===")
print(allr[allr.seed == 0].groupby("method").apply(stats).loc[methods].round(2).to_string())

print("\n=== All seeds 0-2 (36 runs per method) ===")
print(allr.groupby("method").apply(stats).loc[methods].round(2).to_string())

print("\n=== Chosen alpha for +guard by dataset and seed ===")
P = allr[allr.method == "lora_full+guard"].pivot_table(index="dataset", columns="seed", values="alpha")
P["stable"] = P.nunique(axis=1) == 1
print(P.to_string())
print("\n=== +guard % change vs zero-shot by dataset and seed ===")
print(allr[allr.method == "lora_full+guard"].pivot_table(index="dataset", columns="seed", values="change").round(2).to_string())

# paired comparison guard vs zero-shot across the 12 datasets (mean over seeds 1-2)
from math import comb
per_ds = c[c.method == "lora_full+guard"].groupby("dataset").change.mean()
nz = per_ds[per_ds.abs() > 0.01]   # changes within ±0.01% are ties (alpha=0 → identical up to float noise)
k = int((nz < 0).sum()); n = len(nz)
p_sign = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else float("nan")
print(f"\nSign test (seeds 1-2 mean per dataset, ties dropped): {k}/{n} datasets better, one-sided p = {p_sign:.4f}")
