"""TimesFM Part A (6 datasets): build results/tfm_A CSVs from the scraped Kaggle output and report change vs zero-shot.

Usage: python analysis/tfm_partA_analysis.py RAW_RESULTS RAW_CURVES   (raw files only needed once)
       python analysis/tfm_partA_analysis.py                          (re-analyse saved CSVs)
"""
import sys
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "tfm_A"
OUT.mkdir(exist_ok=True)
META = {"us_births/D": (30, 20, 1, 7305), "saugeenday/D": (30, 20, 1, 23741), "ett1/W": (8, 2, 7, 721),
        "ett2/W": (8, 2, 7, 721), "solar/W": (8, 1, 137, 7124), "hierarchical_sales/W": (8, 4, 118, 30680)}

if len(sys.argv) == 3:
    r = pd.read_csv(sys.argv[1], header=None, names=["dataset", "method", "alpha", "fit_sec", "MWQL", "MASE", "MAPE"])
    r["method"] = r.method.str.replace(r"^L", "lora_full", regex=True)
    for i, c in enumerate(["H", "windows", "n_series", "n_obs"]):
        r[c] = r.dataset.map(lambda d: META[d][i])
    r["seed"] = r.method.map(lambda m: 0 if m.startswith("lora") else -1)
    r[["dataset", "H", "windows", "n_series", "n_obs", "method", "seed", "alpha", "fit_sec", "MWQL", "MASE", "MAPE"]] \
        .to_csv(OUT / "results.csv", index=False)
    c = pd.read_csv(sys.argv[2], header=None, names=["dataset", "alpha", "V", "val_MWQL", "val1_MWQL", "test_MWQL", "test_MASE"])
    c.insert(1, "method", "lora_full")
    c.to_csv(OUT / "alpha_curves_compact.csv", index=False)

r = pd.read_csv(OUT / "results.csv")
c = pd.read_csv(OUT / "alpha_curves_compact.csv")
zs = r[r.method == "zero_shot"].set_index("dataset").MWQL
ms = ["lora_full", "lora_full+wise", "lora_full+wise1", "lora_full+guard", "lora_full+guardLF", "zero_shot_rec"]
t = (r[r.method.isin(ms)].pivot_table(index="dataset", columns="method", values="MWQL")
     .div(zs, axis=0).sub(1).mul(100)[ms].loc[list(META)])
a = r[r.method.str.contains(r"\+")].pivot_table(index="dataset", columns="method", values="alpha").loc[list(META)]
pd.set_option("display.width", 200)
print("=== change in MWQL vs zero-shot (%) ===\n", t.round(2).to_string())
print("\nmean:", t.mean().round(2).to_dict())
print("worst:", t.max().round(2).to_dict())
print("no worse (<= +1%):", (t <= 1).sum().to_dict())
print("\n=== selected alpha ===\n", a.to_string())
t.to_csv(OUT / "change_vs_zero_shot.csv")

print("\n=== validation vs test, alpha = 1 ===")
for d, g in c.groupby("dataset", sort=False):
    g = g.set_index("alpha")
    print(f"{d:22s} val gain {100 * (1 - g.val_MWQL[1.0] / g.val_MWQL[0.0]):+6.1f}%  "
          f"test gain {100 * (1 - g.test_MWQL[1.0] / g.test_MWQL[0.0]):+6.1f}%  "
          f"oracle alpha {g.test_MWQL.idxmin()} ({100 * (g.test_MWQL.min() / g.test_MWQL[0.0] - 1):+.2f}%)")
