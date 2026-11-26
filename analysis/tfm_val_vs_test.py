"""TimesFM 2.5, 28 datasets: is validation a reliable signal for the test gain of fine-tuning? (same analysis as §9 for Chronos-2)

gain = 1 - MWQL(alpha=1) / MWQL(alpha=0), positive = fine-tuning helps.
"misled" = pooled validation gain > 3% but test gain < 0.
Output: results/tfm_all/val_vs_test.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
c = pd.concat([pd.read_csv(RES / f"tfm_{p}/alpha_curves_compact.csv") for p in ["A", "BH", "F"]])
HF = {"bizitobs_l2c/H", "bizitobs_service", "bizitobs_application", "m4_hourly", "SZ_TAXI/H",
      "electricity/H", "electricity/15T", "bitbrains_rnd/H", "bitbrains_fast_storage/H"}

rows = []
for d, g in c.groupby("dataset", sort=False):
    g = g.set_index("alpha")
    gain = lambda col: 100 * (1 - g[col][1.0] / g[col][0.0])
    best = g.test_MWQL.idxmin()
    rows.append(dict(dataset=d, freq="high" if d in HF else "low", val_gain=gain("val_MWQL"), last_gain=gain("val1_MWQL"),
                     test_gain=gain("test_MWQL"), oracle_alpha=best,
                     oracle_change=100 * (g.test_MWQL[best] / g.test_MWQL[0.0] - 1)))
t = pd.DataFrame(rows).set_index("dataset")
t["misled"] = (t.val_gain > 3) & (t.test_gain < 0)
t["val_claims_gain"] = t.val_gain > 3
t.round(3).to_csv(RES / "tfm_all/val_vs_test.csv")

pd.set_option("display.width", 200)
print(t.round(2).to_string())
print("\n=== by frequency ===")
for f, g in t.groupby("freq"):
    r = np.corrcoef(g.val_gain, g.test_gain)[0, 1]
    rho = g.val_gain.rank().corr(g.test_gain.rank())
    sign = (np.sign(g.val_gain) == np.sign(g.test_gain)).mean()
    print(f"{f:5s} n={len(g):2d}  Pearson r={r:+.2f}  Spearman rho={rho:+.2f}  sign agreement={100 * sign:.0f}%  "
          f"misled {g.misled.sum()} of {g.val_claims_gain.sum()} runs where validation claims >3% gain")
print(f"\noracle (best alpha on test), mean change over 28: {t.oracle_change.mean():.2f}%")
