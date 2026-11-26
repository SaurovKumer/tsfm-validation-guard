"""Offline study of less conservative alpha-selection rules on 56 runs (Chronos-2 + TimesFM 2.5, 28 datasets each, LoRA full, seed 0).

No GPU needed. Every rule picks alpha from validation information only (pooled V-window MWQL and last-window MWQL).
The test set is used only for scoring. Hyper-parameters are chosen by leave-one-dataset-out (LODO):
for each dataset, the setting is tuned on the other 27 datasets (both models) and applied to the held-out one.

Run: python analysis/rules_v2.py      Output: results/rules_v2/
"""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "rules_v2"
OUT.mkdir(exist_ok=True)

COLS = ["dataset", "alpha", "val_MWQL", "val1_MWQL", "test_MWQL"]
chr_ = pd.concat([pd.read_csv(RES / p / "alpha_curves_compact.csv") for p in
                  ["partA", "partB", "holdoutH1", "holdoutH2", "freshF1", "freshF2"]])
chr_ = chr_[chr_.method == "lora_full"][COLS].assign(model="chronos2")
tfm = pd.concat([pd.read_csv(RES / f"tfm_{p}/alpha_curves_compact.csv") for p in ["A", "BH", "F"]])[COLS].assign(model="timesfm")
C = pd.concat([chr_, tfm]).sort_values(["model", "dataset", "alpha"])
HF = {"bizitobs_l2c/H", "bizitobs_service", "bizitobs_application", "m4_hourly", "SZ_TAXI/H",
      "electricity/H", "electricity/15T", "bitbrains_rnd/H", "bitbrains_fast_storage/H"}
HOLD = {"hospital", "covid_deaths", "car_parts_with_missing", "m4_weekly", "bizitobs_service",
        "m4_hourly", "SZ_TAXI/H", "kdd_cup_2018_with_missing/D", "bizitobs_application"}
FRESH = {"electricity/H", "electricity/15T", "electricity/W", "bitbrains_rnd/H", "electricity/D", "m4_daily",
         "bitbrains_fast_storage/H"}

RUNS = {}
for (mdl, d), g in C.groupby(["model", "dataset"]):
    RUNS[(mdl, d)] = dict(a=g.alpha.values, v=g.val_MWQL.values, v1=g.val1_MWQL.values, t=g.test_MWQL.values, hf=d in HF)
assert len(RUNS) == 56, len(RUNS)
DATASETS = sorted({d for _, d in RUNS})


def pick(r, m=0.03, cap=1.0, last=True, hf_zero=False, m_hf=None):
    """Generalised guard: among alpha <= cap, take the pooled-validation argmin among the alphas that
    improve pooled validation by more than m (m_hf for high-frequency data) and, if `last`,
    do not worsen the most recent validation window. Otherwise alpha = 0."""
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


def change(r, alpha):
    return 100 * (r["t"][np.searchsorted(r["a"], alpha)] / r["t"][0] - 1)


def score(params, keys):
    return np.array([change(RUNS[k], pick(RUNS[k], **params)) for k in keys])


def summ(x):
    s = np.sort(x)[::-1]
    return dict(mean=x.mean(), median=np.median(x), worst=x.max(), cvar10=s[:max(1, len(x) // 10)].mean(),
                no_worse=int((x <= 1).sum()), better=int((x < -1).sum()), n=len(x))


KEYS = sorted(RUNS)
oracle = np.array([100 * (RUNS[k]["t"].min() / RUNS[k]["t"][0] - 1) for k in KEYS])

# ---- fixed reference rules (all settings fixed in advance; no tuning) ----
FIXED = {
    "LoRA (alpha=1)": None,
    "+wise (argmin, no margin)": dict(m=0.0, last=False),
    "+guard (m=3%, last)": dict(m=0.03),
    "guard-LF (m=3%, last, HF->0)": dict(m=0.03, hf_zero=True),
}
# ---- families with tuned hyper-parameters (LODO) ----
MS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
CAPS = [0.25, 0.5, 0.75, 1.0]
FAMILIES = {
    "guard, m tuned": [dict(m=m) for m in MS],
    "guard, m+cap tuned": [dict(m=m, cap=c) for m, c in product(MS, CAPS)],
    "guard-LF, m+cap tuned": [dict(m=m, cap=c, hf_zero=True) for m, c in product(MS, CAPS)],
    "guard, m_LF/m_HF+cap tuned": [dict(m=m, m_hf=mh, cap=c) for m, mh, c in product(MS, MS + [0.1], CAPS)],
    "full family (m, cap, last, HF->0)": [dict(m=m, cap=c, last=l, hf_zero=h)
                                          for m, c, l, h in product(MS, CAPS, [True, False], [True, False])],
}


def lodo(grid, objective="mean"):
    """Returns per-run changes (in KEYS order) and the settings chosen in each fold."""
    allsc = {i: score(p, KEYS) for i, p in enumerate(grid)}
    out, chosen = np.zeros(len(KEYS)), []
    for d in DATASETS:
        tr = np.array([k[1] != d for k in KEYS])
        def obj(i):
            x = allsc[i][tr]
            return x.mean() if objective == "mean" else x.mean() + 0.5 * np.sort(x)[::-1][:max(1, len(x) // 10)].mean()
        best = min(allsc, key=obj)
        out[~tr] = allsc[best][~tr]
        chosen.append((d, best))
    return out, chosen


rows, per_run = [], {"oracle": oracle}
for name, p in FIXED.items():
    x = np.array([change(RUNS[k], 1.0) for k in KEYS]) if p is None else score(p, KEYS)
    per_run[name] = x
    rows.append(dict(rule=name, tuned="no", **summ(x)))
choices = {}
for obj in ["mean", "mean+risk"]:
    for name, grid in FAMILIES.items():
        x, ch = lodo(grid, obj)
        nm = f"{name} [LODO {obj}]"
        per_run[nm] = x
        choices[nm] = pd.Series([str(grid[i]) for _, i in ch]).value_counts().head(3).to_dict()
        rows.append(dict(rule=nm, tuned="LODO", **summ(x)))
rows.append(dict(rule="oracle (best alpha on test)", tuned="-", **summ(oracle)))
T = pd.DataFrame(rows)
T["oracle_share"] = T["mean"] / oracle.mean()

P = pd.DataFrame(per_run, index=pd.MultiIndex.from_tuples(KEYS, names=["model", "dataset"]))
P.insert(0, "group", ["fresh" if d in FRESH else "hold-out" if d in HOLD else "design" for _, d in KEYS])
P.round(3).to_csv(OUT / "per_run_change.csv")
T.round(3).to_csv(OUT / "rule_summary.csv", index=False)

pd.set_option("display.width", 250, "display.max_colwidth", 60)
print("=== 56 runs (28 datasets x 2 models), change in test MWQL vs zero-shot (%) ===")
print(T.round(2).to_string(index=False))
print("\n=== by model / group (mean change; worst) ===")
for c in ["+guard (m=3%, last)", "guard-LF (m=3%, last, HF->0)", "+wise (argmin, no margin)"] + \
         [f"{n} [LODO mean]" for n in FAMILIES] + ["oracle"]:
    g = P.groupby(level="model")[c].agg(["mean", "max"]).round(2)
    h = P.groupby("group")[c].agg(["mean", "max"]).round(2)
    print(f"{c:50s} " + "  ".join(f"{i}: {r['mean']:+.2f}/{r['max']:+.1f}" for i, r in pd.concat([g, h]).iterrows()))
print("\n=== most frequent LODO choices ===")
for k, v in choices.items():
    print(k, v)
