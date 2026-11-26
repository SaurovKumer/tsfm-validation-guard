"""Offline study of alpha-selection rules for guarded interpolation (no GPU needed).

Inputs: results/partA|partB/alpha_curves_compact.csv (val/test MWQL for every alpha) and results.csv
(Seasonal Naive MWQL per dataset, used for normalisation).
Every rule picks alpha using validation information only; the test set is used only for scoring.
To avoid tuning a rule's hyper-parameter on the test set, rules with a margin m are also evaluated with
leave-one-dataset-out (LODO): for each held-out dataset, m is chosen on the other 11 datasets.

Run:  python analysis/selection_rules.py   (from D:\\01 research)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "selection_rules"
OUT.mkdir(exist_ok=True)

curves = pd.concat([pd.read_csv(ROOT / p / "alpha_curves_compact.csv") for p in ("partA", "partB")])
res = pd.concat([pd.read_csv(ROOT / p / "results.csv") for p in ("partA", "partB")])
sn = res[res.method == "seasonal_naive"].set_index("dataset").MWQL
zs = res[res.method == "zero_shot"].set_index("dataset").MWQL
DATASETS = list(dict.fromkeys(curves.dataset))
MODES = ["fft", "lora_full", "lora_few"]
MARGINS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]


def pick(g: pd.DataFrame, rule: str, m: float = 0.0) -> float:
    """g: alpha curve of one (dataset, mode), sorted by alpha. Returns the chosen alpha."""
    g = g.sort_values("alpha")
    v, v1, a = g.val_MWQL.values, g.val1_MWQL.values, g.alpha.values
    if rule == "zero_shot":
        return 0.0
    if rule == "full_ft":
        return 1.0
    if rule == "oracle_test":
        return a[np.argmin(g.test_MWQL.values)]
    if rule == "argmin_V3":                      # = +wise
        return a[np.argmin(v)]
    if rule == "argmin_V1":                      # = +wise1
        return a[np.argmin(v1)]
    if rule == "margin_V3":                      # leave alpha=0 only if pooled val improves by > m
        i = np.argmin(v)
        return a[i] if v[i] < (1 - m) * v[0] else 0.0
    if rule == "shrink_half":                    # halve the argmin alpha, snapped down to the grid
        t = a[np.argmin(v)] / 2
        return a[a <= t + 1e-9].max()
    if rule == "agree_min":                      # take the smaller of the V3 and V1 choices
        return min(a[np.argmin(v)], a[np.argmin(v1)])
    if rule == "both_improve":                   # alpha must improve BOTH pooled val (by > m) and the last window
        ok = (v < (1 - m) * v[0]) & (v1 <= v1[0])
        ok[0] = True
        cand = np.where(ok)[0]
        return a[cand[np.argmin(v[cand])]]
    raise ValueError(rule)


def evaluate(rule: str, m: float = 0.0) -> pd.DataFrame:
    rows = []
    for (d, mode), g in curves.rename(columns={"method": "mode"}).groupby(["dataset", "mode"]):
        alpha = pick(g, rule, m)
        t = g.set_index("alpha").test_MWQL[alpha]
        rows.append(dict(dataset=d, mode=mode, alpha=alpha, test_MWQL=t,
                         MWQL_n=t / sn[d], change_vs_zs=100 * (t / zs[d] - 1)))
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    return dict(mean_MWQL_n=df.MWQL_n.mean(),
                mean_change_vs_zs=df.change_vs_zs.mean(),
                n_no_worse=int((df.change_vs_zs <= 1.0).sum()),
                n_strictly_better=int((df.change_vs_zs < -1.0).sum()),
                worst=df.change_vs_zs.max(), best=df.change_vs_zs.min(),
                frac_alpha0=(df.alpha == 0).mean())


rules = [("zero_shot", 0), ("full_ft", 0), ("argmin_V3", 0), ("argmin_V1", 0), ("shrink_half", 0),
         ("agree_min", 0)] + [("margin_V3", m) for m in MARGINS[1:]] + \
        [("both_improve", m) for m in MARGINS] + [("oracle_test", 0)]
table, per_pair = [], {}
for r, m in rules:
    df = evaluate(r, m)
    name = r if r in ("zero_shot", "full_ft", "argmin_V3", "argmin_V1", "shrink_half", "agree_min", "oracle_test") \
        else f"{r}(m={m:.0%})"
    per_pair[name] = df
    for mode in MODES + ["all"]:
        sub = df if mode == "all" else df[df["mode"] == mode]
        table.append(dict(rule=name, mode=mode, n=len(sub), **summary(sub)))
T = pd.DataFrame(table)

# ---- leave-one-dataset-out choice of the margin m (honest estimate for margin-based rules) ----
lodo = []
for r in ("margin_V3", "both_improve"):
    grid = {m: evaluate(r, m) for m in MARGINS}
    for d in DATASETS:
        best_m = min(MARGINS, key=lambda m: grid[m][grid[m].dataset != d].MWQL_n.mean())
        lodo.append(grid[best_m][grid[best_m].dataset == d].assign(rule=f"{r}_LODO", m=best_m))
L = pd.concat(lodo)
for r, sub in L.groupby("rule"):
    for mode in MODES + ["all"]:
        s = sub if mode == "all" else sub[sub["mode"] == mode]
        T.loc[len(T)] = dict(rule=r, mode=mode, n=len(s), **summary(s))

T.to_csv(OUT / "rule_summary.csv", index=False)
L.to_csv(OUT / "lodo_choices.csv", index=False)
for name, df in per_pair.items():
    df.to_csv(OUT / f"pairs_{name.replace('(', '_').replace(')', '').replace('=', '').replace('%', 'pct')}.csv",
              index=False)

pd.set_option("display.width", 200)
allm = T[T["mode"] == "all"].drop(columns=["mode"]).round(3)
print("=== All 36 (dataset, mode) pairs; change in % vs zero-shot; 'no worse' allows +1% ===")
print(allm.to_string(index=False))
print("\nLODO-chosen margins:", L.groupby("rule").m.value_counts().to_dict())
