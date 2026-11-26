"""Why did validation-guarded interpolation fail on m4_hourly and bizitobs_service?

Uses every alpha curve we have:
- design set, seed 0: 12 datasets x {fft, lora_full, lora_few}
- hold-out set, seed 0: 9 datasets x lora_full

For each run we compare the validation gain of fine-tuning with its test gain, and look for dataset
properties that predict when validation is misleading. EXPLORATORY: any rule suggested here must be confirmed
on fresh data before it is claimed.
Run from D:\\01 research:  python analysis/failure_diagnosis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "diagnosis"
OUT.mkdir(exist_ok=True)

cur = pd.concat([pd.read_csv(R / p / "alpha_curves_compact.csv").assign(set=s)
                 for p, s in (("partA", "design"), ("partB", "design"), ("holdoutH1", "holdout"),
                              ("holdoutH2", "holdout"))])
res = pd.concat([pd.read_csv(R / p / "results.csv")
                 for p in ("partA", "partB", "holdoutH1", "holdoutH2")])
meta = res.drop_duplicates("dataset").set_index("dataset")[["H", "windows", "n_series", "n_obs"]]

# seasonal period (gluonts get_seasonality) and sampling interval in hours, per dataset
SEASON = {"/H": 24, "_hourly": 24, "bizitobs_service": 360, "bizitobs_application": 360, "hospital": 12,
          "car_parts": 12}
STEP_H = {"/H": 1, "_hourly": 1, "bizitobs_service": 10 / 3600, "bizitobs_application": 10 / 3600,
          "/D": 24, "covid": 24, "kdd": 24, "/W": 168, "_weekly": 168, "hospital": 730, "car_parts": 730}


def lookup(d, table, default):
    for k, v in table.items():
        if k in d:
            return v
    return default


rows = []
for (d, m, s), g in cur.groupby(["dataset", "method", "set"]):
    g = g.sort_values("alpha").set_index("alpha")
    v0, v10, t0 = g.val_MWQL[0.0], g.val1_MWQL[0.0], g.test_MWQL[0.0]
    V, H = int(g.V.iloc[0]), int(meta.loc[d, "H"])
    season = lookup(d, SEASON, 1)
    rows.append(dict(
        dataset=d, mode=m, set=s, V=V, H=H, windows=int(meta.loc[d, "windows"]), season=season,
        step_hours=lookup(d, STEP_H, np.nan),
        val_gain_full=100 * (1 - g.val_MWQL[1.0] / v0),        # + = fine-tuning helps on validation
        val1_gain_full=100 * (1 - g.val1_MWQL[1.0] / v10),
        test_gain_full=100 * (1 - g.test_MWQL[1.0] / t0),     # + = fine-tuning helps on test
        best_val_gain=100 * (1 - g.val_MWQL.min() / v0),
        zs_test_over_val=t0 / v0,                             # >1: test period harder than validation
        val_span_over_season=V * H / season,                  # how many seasonal cycles validation covers
        test_span_over_val_span=int(meta.loc[d, "windows"]) / V,
    ))
D = pd.DataFrame(rows)
D["val_says_ft_helps"] = D.val_gain_full > 3
D["test_says_ft_helps"] = D.test_gain_full > 0
D["misled"] = D.val_says_ft_helps & ~D.test_says_ft_helps   # validation says >3% gain, test says loss
D["freq_group"] = np.where(D.step_hours <= 1, "high (hourly or finer)", "low (daily or coarser)")
D.to_csv(OUT / "val_vs_test_per_run.csv", index=False)

pd.set_option("display.width", 220)
print("=== 1. Does validation gain predict test gain? (fine-tuned model, alpha=1) ===")
for grp, g in D.groupby("freq_group"):
    r = np.corrcoef(g.val_gain_full, g.test_gain_full)[0, 1]
    agree = (np.sign(g.val_gain_full) == np.sign(g.test_gain_full)).mean()
    print(f"{grp:24s} n={len(g):2d}  Pearson r={r:+.2f}  sign agreement={agree:.0%}  "
          f"misled(val>3% gain, test loss)={int(g.misled.sum())}/{int(g.val_says_ft_helps.sum())}")
r_all = np.corrcoef(D.val_gain_full, D.test_gain_full)[0, 1]
print(f"{'all':24s} n={len(D):2d}  Pearson r={r_all:+.2f}")

print("\n=== 2. Runs where validation says fine-tuning helps by >3% ===")
cols = ["dataset", "mode", "set", "freq_group", "val_gain_full", "val1_gain_full", "test_gain_full",
        "zs_test_over_val", "val_span_over_season", "test_span_over_val_span"]
print(D[D.val_says_ft_helps].sort_values("test_gain_full")[cols].round(2).to_string(index=False))

print("\n=== 3. Candidate explanations (only runs with val gain > 3%) ===")
S = D[D.val_says_ft_helps]
for c in ["freq_group"]:
    print(S.groupby(c).agg(n=("misled", "size"), misled=("misled", "sum")).to_string())
for c in ["zs_test_over_val", "val_span_over_season", "test_span_over_val_span", "val_gain_full"]:
    a, b = S[S.misled][c], S[~S.misled][c]
    print(f"{c:24s} misled: {a.round(2).tolist()}   ok: median {b.median():.2f} "
          f"(range {b.min():.2f}..{b.max():.2f})")

print("\n=== 4. Exploratory rule: '+guard', but stay at zero-shot for high-frequency data ===")
gr = pd.concat([pd.read_csv(R / p / "results.csv") for p in ("partA", "partB", "holdoutH1", "holdoutH2")])
zs = gr[gr.method == "zero_shot"].set_index("dataset").MWQL
lf = gr[gr.method == "lora_full+guard"].copy()
if lf.empty:   # design set has no +guard rows: recompute guard from curves
    pass
G = []
for (d, s), g in cur[cur.method == "lora_full"].groupby(["dataset", "set"]):
    g = g.sort_values("alpha")
    v0, v10 = g.val_MWQL.iloc[0], g.val1_MWQL.iloc[0]
    ok = g[(g.alpha == 0) | ((g.val_MWQL < 0.97 * v0) & (g.val1_MWQL <= v10))]
    best = ok.sort_values(["val_MWQL", "alpha"]).iloc[0]
    hf = D[(D.dataset == d) & (D["mode"] == "lora_full")].freq_group.iloc[0].startswith("high")
    G.append(dict(dataset=d, set=s, high_freq=hf,
                  guard=100 * (best.test_MWQL / zs[d] - 1),
                  guard_lowfreq_only=0.0 if hf else 100 * (best.test_MWQL / zs[d] - 1)))
G = pd.DataFrame(G)
for s, g in G.groupby("set"):
    for col in ["guard", "guard_lowfreq_only"]:
        print(f"{s:8s} {col:20s} no-worse {int((g[col] <= 1).sum())}/{len(g)}  mean {g[col].mean():+.2f}%  "
              f"worst {g[col].max():+.2f}%")
G.to_csv(OUT / "guard_vs_lowfreq_rule.csv", index=False)
