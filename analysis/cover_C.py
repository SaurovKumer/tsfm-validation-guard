"""§19: coverage re-run (registered in §19). For each model whose C1/C2 files exist, the 28 configurations re-run with
interval coverage and split-conformal (CQR) recalibration for every alpha.
  R1  coverage at alpha = 0 vs alpha = 1 (and at the alpha chosen by rule A)
  R2  CQR fitted on validation windows: coverage and width on test
  R3  reproducibility: the seed-0 change at alpha = 1 and the alpha chosen by rule A vs the original runs
Run: python analysis/cover_C.py      Output: results/cover_C/
"""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "cover_C"
OUT.mkdir(exist_ok=True)
HF = {"bizitobs_l2c/H", "bizitobs_service", "bizitobs_application", "m4_hourly", "SZ_TAXI/H",
      "electricity/H", "electricity/15T", "bitbrains_rnd/H", "bitbrains_fast_storage/H"}
NEW = {"Chronos-2": ["chr_C1", "chr_C2"], "TimesFM 2.5": ["tfm_C1", "tfm_C2"]}
OLD = {"Chronos-2": ["partA", "partB", "holdoutH1", "holdoutH2", "freshF1", "freshF2"], "TimesFM 2.5": ["tfm_A", "tfm_BH", "tfm_F"]}


def pick(r, m=0.02, hf_zero=True):
    if hf_zero and r["hf"]:
        return 0.0
    a, v, v1 = r["a"], r["v"], r["v1"]
    ok = (v < (1 - m) * v[0]) & (v1 <= v1[0])
    ok[0] = True
    i = np.where(ok)[0]
    return a[i[np.argmin(v[i])]]


def runs(df):
    out = {}
    for d, g in df[df.method == "lora_full"].sort_values("alpha").groupby("dataset"):
        out[d] = g.reset_index(drop=True)
    return out


rows, rep = [], []
for model in NEW:
    if not all((RES / p / "alpha_curves.csv").exists() for p in NEW[model]):
        print(f"{model}: coverage re-run not available yet")
        continue
    new = runs(pd.concat([pd.read_csv(RES / p / "alpha_curves.csv") for p in NEW[model]]))
    old = runs(pd.concat([pd.read_csv(RES / p / f) for p in OLD[model] for f in ["alpha_curves_compact.csv"]
                          if (RES / p / f).exists()]))
    for d, g in new.items():
        r = dict(a=g.alpha.values, v=g.val_MWQL.values, v1=g.val1_MWQL.values, hf=d in HF)
        aA = pick(r)
        at = {"a=0": 0, "a=1": len(g) - 1, "A": int(np.searchsorted(r["a"], aA))}
        for lab, i in at.items():
            x = g.iloc[i].copy()
            if not np.isfinite(x.cqr_q80):   # Chronos C1/C2 bug: all-NaN context gave a NaN scale -> CQR invalid
                for c in ["test_cqr_cov80", "test_cqr_width80", "test_cqr_cov60", "test_cqr_width60"]:
                    x[c] = np.nan
            rows.append(dict(model=model, dataset=d, at=lab, alpha=x.alpha,
                             change=100 * (x.test_MWQL / g.test_MWQL.iloc[0] - 1),
                             cov80=x.test_cov80, width80=x.test_width80, cov60=x.test_cov60, width60=x.test_width60,
                             cqr_cov80=x.test_cqr_cov80, cqr_width80=x.test_cqr_width80,
                             cqr_cov60=x.test_cqr_cov60, cqr_width60=x.test_cqr_width60))
        if d in old:
            o = old[d]
            ro = dict(a=o.alpha.values, v=o.val_MWQL.values, v1=o.val1_MWQL.values, hf=d in HF)
            rep.append(dict(model=model, dataset=d,
                            zs_new=g.test_MWQL.iloc[0], zs_old=o.test_MWQL.iloc[0],
                            lora_new=100 * (g.test_MWQL.iloc[-1] / g.test_MWQL.iloc[0] - 1),
                            lora_old=100 * (o.test_MWQL.iloc[-1] / o.test_MWQL.iloc[0] - 1),
                            aA_new=aA, aA_old=pick(ro)))
V = pd.DataFrame(rows)
R = pd.DataFrame(rep)
if V.empty:
    raise SystemExit
V.round(4).to_csv(OUT / "coverage_per_run.csv", index=False)
R.round(4).to_csv(OUT / "reproducibility.csv", index=False)
pd.set_option("display.width", 220)
cols = ["cov80", "width80", "cqr_cov80", "cqr_width80", "cov60", "width60", "cqr_cov60", "cqr_width60"]
print("=== R1/R2: mean over configurations (nominal 0.80 / 0.60) ===")
print(V.groupby(["model", "at"])[cols].mean().round(3).to_string())
V["hf"] = V.dataset.isin(HF)
print(V.groupby(["model", "hf", "at"])[["cov80", "cqr_cov80", "cov60", "cqr_cov60"]].mean().round(3).to_string())
w = V.pivot_table(index=["model", "dataset"], columns="at", values="cov80")
print("\nruns where coverage (80%) fell by > 0.05 from a=0 to a=1:", int(((w["a=0"] - w["a=1"]) > 0.05).sum()), "of", len(w),
      "| rose by > 0.05:", int(((w["a=1"] - w["a=0"]) > 0.05).sum()))
print("\n=== R3: reproducibility of the seed-0 runs ===")
R["zs_rel"] = 100 * (R.zs_new / R.zs_old - 1)
R["lora_diff"] = R.lora_new - R.lora_old
print(R[["model", "dataset", "zs_rel", "lora_old", "lora_new", "aA_old", "aA_new"]].round(2).to_string(index=False))
print(f"\nzero-shot max |rel diff| {R.zs_rel.abs().max():.4f}% | LoRA change: corr {np.corrcoef(R.lora_old, R.lora_new)[0,1]:.3f}, "
      f"median |diff| {R.lora_diff.abs().median():.2f} pp | same sign {int((np.sign(R.lora_old) == np.sign(R.lora_new)).sum())}/{len(R)} | "
      f"rule A same alpha {int((R.aA_old == R.aA_new).sum())}/{len(R)}")
