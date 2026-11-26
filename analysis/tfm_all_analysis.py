"""TimesFM 2.5, all 28 datasets: change in test MWQL vs zero-shot per rule, by group, and the pre-registered criteria T1-T3 (§12).

Inputs: results/tfm_A/change_vs_zero_shot.csv, results/tfm_BH/change_vs_zero_shot.csv, results/tfm_F/results.csv
Output: results/tfm_all/change_vs_zero_shot.csv, summary_by_group.csv
"""
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "tfm_all"
OUT.mkdir(exist_ok=True)
M = ["lora_full", "lora_full+wise", "lora_full+wise1", "lora_full+guard", "lora_full+guardLF", "zero_shot_rec"]
SHORT = {"lora_full": "LoRA", "lora_full+wise": "+wise", "lora_full+wise1": "+wise1", "lora_full+guard": "+guard",
         "lora_full+guardLF": "guard-LF", "zero_shot_rec": "zs_rec"}

a = pd.read_csv(RES / "tfm_A/change_vs_zero_shot.csv", index_col=0)
bh = pd.read_csv(RES / "tfm_BH/change_vs_zero_shot.csv", index_col=0)
f = pd.read_csv(RES / "tfm_F/results.csv")
zs = f[f.method == "zero_shot"].set_index("dataset").MWQL
f = (f[f.method.isin(M)].pivot_table(index="dataset", columns="method", values="MWQL", sort=False)
     .div(zs, axis=0).sub(1).mul(100))
t = pd.concat([a[M], bh[M], f[M]])
assert len(t) == 28 and t.index.is_unique, t.index

HOLD = ["hospital", "covid_deaths", "car_parts_with_missing", "m4_weekly", "bizitobs_service",
        "m4_hourly", "SZ_TAXI/H", "kdd_cup_2018_with_missing/D", "bizitobs_application"]
FRESH = list(f.index)
t.insert(0, "group", ["fresh" if d in FRESH else "hold-out" if d in HOLD else "design" for d in t.index])
t.round(4).to_csv(OUT / "change_vs_zero_shot.csv")

def summ(x):
    return pd.DataFrame({"mean": x.mean(), "median": x.median(), "worst": x.max(), "best": x.min(),
                         "no_worse(<=+1%)": (x <= 1).sum(), "better(<-1%)": (x < -1).sum(), "n": x.count()})

pd.set_option("display.width", 220)
rows = []
for g, x in [("all", t)] + list(t.groupby("group", sort=False)):
    s = summ(x[M]).rename(index=SHORT)
    s.insert(0, "group", g)
    rows.append(s)
    print(f"\n=== {g} ({len(x)} datasets) ===\n{s.drop(columns='group').round(2).to_string()}")
pd.concat(rows).round(4).to_csv(OUT / "summary_by_group.csv")

print("\n=== per dataset (change in MWQL vs zero-shot, %) ===")
print(t.rename(columns=SHORT).round(2).to_string())

w = t[M].max()
print("\n=== pre-registered criteria (§12) ===")
t1 = {r: w[r] <= w["lora_full"] for r in ["lora_full+wise", "lora_full+guard", "lora_full+guardLF"]}
print("T1 worst case of each rule <= worst of plain LoRA (%.2f):" % w["lora_full"],
      {SHORT[k]: f"{w[k]:.2f} {'PASS' if v else 'FAIL'}" for k, v in t1.items()})
nw, mg = (t["lora_full+guardLF"] <= 1).sum(), t["lora_full+guardLF"].mean()
print(f"T2 guard-LF no worse on {nw}/28 (need >= 25) and mean {mg:.2f}% (need <= 0):",
      "PASS" if nw >= 25 and mg <= 0 else "FAIL")
mw, ml = t["lora_full+wise"].mean(), t["lora_full"].mean()
print(f"T3 +wise mean {mw:.2f}% < LoRA mean {ml:.2f}% and <= 0:", "PASS" if mw < ml and mw <= 0 else "FAIL")
