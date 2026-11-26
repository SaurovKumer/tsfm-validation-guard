"""Evaluate the pre-registered fresh-source test of guard-LF (phase1_analysis.md section 10).

Reads results/freshF1 and results/freshF2 (whichever exist) and prints:
- per-dataset change vs zero-shot for every method,
- primary criteria (1) no-worse (<= +1%) on >= 6/7 and (2) mean <= 0%,
- secondary (3) misled share high- vs low-frequency, (4) +guard vs guard-LF on high-frequency data.
Run from D:\\01 research:  python analysis/fresh_analysis.py
"""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "fresh"
OUT.mkdir(exist_ok=True)
HIGH = {"electricity/H", "electricity/15T", "bitbrains_fast_storage/H", "bitbrains_rnd/H"}
LOW = {"electricity/D", "electricity/W", "m4_daily"}
EXPECTED = HIGH | LOW

parts = [p for p in ("freshF1", "freshF2") if (R / p / "results.csv").exists()]
res = pd.concat([pd.read_csv(R / p / "results.csv") for p in parts])
cur = pd.concat([pd.read_csv(R / p / "alpha_curves_compact.csv") for p in parts
                 if (R / p / "alpha_curves_compact.csv").exists()])
res = res[~res.method.str.endswith("_FAILED")]
done = sorted(set(res.dataset) & EXPECTED)
missing = sorted(EXPECTED - set(done))

zs = res[res.method == "zero_shot"].set_index("dataset").MWQL
methods = ["lora_full", "lora_full+wise", "lora_full+wise1", "lora_full+guard", "lora_full+guardLF"]
tab = (res[res.method.isin(methods)].pivot_table(index="dataset", columns="method", values="MWQL")
       .div(zs, axis=0).sub(1).mul(100))
alpha = res[res.method.isin(methods[1:])].pivot_table(index="dataset", columns="method", values="alpha")
tab["group"] = ["high" if d in HIGH else "low" for d in tab.index]

pd.set_option("display.width", 200)
print(f"Datasets done: {len(done)}/7   missing: {missing or 'none'}\n")
print("=== Change in MWQL vs zero-shot (%), negative = better ===")
print(tab.round(2).to_string())
print("\n=== Selected alpha ===")
print(alpha.to_string())

g = tab["lora_full+guardLF"]
n_ok, n = int((g <= 1).sum()), len(g)
c1 = n_ok >= n - 1 if n == 7 else None
c2 = g.mean() <= 0
print("\n=== Primary criteria (guard-LF) ===")
print(f"(1) no worse (<= +1%) on {n_ok}/{n}  -> {'PASS' if c1 else ('FAIL' if c1 is False else 'pending (need 7)')}"
      f"  [needed >= 6 of 7]")
print(f"(2) mean change {g.mean():+.2f}%  -> {'PASS' if c2 else 'FAIL'}  [needed <= 0%]")

rows = []
for d, c in cur[cur.method == "lora_full"].groupby("dataset"):
    c = c.sort_values("alpha").set_index("alpha")
    vg = 100 * (1 - c.val_MWQL[1.0] / c.val_MWQL[0.0])
    tg = 100 * (1 - c.test_MWQL[1.0] / c.test_MWQL[0.0])
    rows.append(dict(dataset=d, group="high" if d in HIGH else "low", val_gain=vg, test_gain=tg,
                     val_says_helps=vg > 3, misled=(vg > 3) and (tg <= 0)))
M = pd.DataFrame(rows)
print("\n=== Secondary (3): validation vs test gain of the fine-tuned model (alpha=1) ===")
print(M.round(2).to_string(index=False))
for grp, s in M.groupby("group"):
    print(f"{grp:5s} misled {int(s.misled.sum())}/{len(s)} runs "
          f"(val claimed >3% gain in {int(s.val_says_helps.sum())})")

hf = tab[tab.group == "high"]
print("\n=== Secondary (4): +guard vs guard-LF on high-frequency datasets ===")
print(f"+guard mean {hf['lora_full+guard'].mean():+.2f}%  worst {hf['lora_full+guard'].max():+.2f}%   "
      f"guard-LF mean {hf['lora_full+guardLF'].mean():+.2f}%")
print(f"(4) {'PASS' if hf['lora_full+guard'].mean() > hf['lora_full+guardLF'].mean() + 1e-4 else 'NOT SUPPORTED'}"
      f" (+guard worse than guard-LF?)")

tab.to_csv(OUT / "fresh_change_vs_zero_shot.csv")
M.to_csv(OUT / "fresh_val_vs_test.csv", index=False)
