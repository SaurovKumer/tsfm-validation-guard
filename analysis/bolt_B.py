"""§21: evaluate the pre-registered §20 third-model test (Chronos-Bolt base, 36 configurations, LoRA full, seed 0).

Rules are computed offline from the alpha curves exactly as in rules_v2.py / confirm_N.py:
  A = pick(m=0.02, hf_zero=True); B = pick(m=0.02, m_hf=0.10, cap=0.5).
High frequency = the rules_v2 HF set (28 configurations) or base frequency H, T or S (the 8 N configurations).
Criteria (§20): K1 worst case of each interpolation rule <= worst case of plain LoRA;
                K2 rule A no worse than zero-shot (<= +1%) on >= 34/36 and mean <= 0%;
                K3 +wise mean < plain-LoRA mean.
Run: python analysis/bolt_B.py      Output: results/bolt_B/
"""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "bolt_B"
OUT.mkdir(exist_ok=True)
HF = {"bizitobs_l2c/H", "bizitobs_service", "bizitobs_application", "m4_hourly", "SZ_TAXI/H",
      "electricity/H", "electricity/15T", "bitbrains_rnd/H", "bitbrains_fast_storage/H"}
NEW8 = {"restaurant", "hierarchical_sales/D", "solar/D", "us_births/W",
        "ett1/H", "M_DENSE/H", "kdd_cup_2018_with_missing/H", "jena_weather/H"}


def is_hf(d):
    if d in HF:
        return True
    return "/" in d and d.split("/")[-1].endswith(("H", "T", "S"))


def pick(r, m=0.03, cap=1.0, last=True, hf_zero=False, m_hf=None):
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


RULES = {
    "A: guard-LF m=2%": dict(m=0.02, hf_zero=True),
    "B: m_LF 2%/m_HF 10%/cap 0.5": dict(m=0.02, m_hf=0.10, cap=0.5),
    "guard-LF m=3%": dict(m=0.03, hf_zero=True),
    "+guard m=3%": dict(m=0.03),
    "+wise": dict(m=0.0, last=False),
}

C = pd.concat([pd.read_csv(RES / p / "alpha_curves.csv") for p in ["bolt_B1", "bolt_B2"]])
C = C[C.method == "lora_full"].sort_values(["dataset", "alpha"])
assert C.dataset.nunique() == 36, C.dataset.nunique()

rows, cov = [], []
for d, g in C.groupby("dataset"):
    g = g.reset_index(drop=True)
    r = dict(a=g.alpha.values, v=g.val_MWQL.values, v1=g.val1_MWQL.values, t=g.test_MWQL.values, hf=is_hf(d))
    ch = lambda al: 100 * (r["t"][np.searchsorted(r["a"], al)] / r["t"][0] - 1)
    row = dict(dataset=d, hf=r["hf"], new8=d in NEW8, **{"LoRA (a=1)": ch(1.0)})
    sel = {}
    for name, p in RULES.items():
        sel[name] = pick(r, **p)
        row[name] = ch(sel[name])
        row["alpha " + name.split(":")[0]] = sel[name]
    row["oracle"] = 100 * (r["t"].min() / r["t"][0] - 1)
    row["val_gain_a1"] = 100 * (r["v"][-1] / r["v"][0] - 1)
    rows.append(row)
    for lab, al in [("a=0", 0.0), ("a=1", 1.0), ("A", sel["A: guard-LF m=2%"])]:
        x = g.iloc[np.searchsorted(r["a"], al)].copy()
        if not np.isfinite(x.cqr_q80):
            for c in ["test_cqr_cov80", "test_cqr_width80", "test_cqr_cov60", "test_cqr_width60"]:
                x[c] = np.nan
        cov.append(dict(dataset=d, hf=r["hf"], at=lab, alpha=al,
                        cov80=x.test_cov80, width80=x.test_width80, cov60=x.test_cov60, width60=x.test_width60,
                        cqr_cov80=x.test_cqr_cov80, cqr_width80=x.test_cqr_width80,
                        cqr_cov60=x.test_cqr_cov60, cqr_width60=x.test_cqr_width60))

P = pd.DataFrame(rows)
V = pd.DataFrame(cov)
P.round(4).to_csv(OUT / "per_run.csv", index=False)
V.round(4).to_csv(OUT / "coverage.csv", index=False)

cols = ["LoRA (a=1)"] + list(RULES) + ["oracle"]
summ = pd.DataFrame({c: dict(mean=P[c].mean(), median=P[c].median(), worst=P[c].max(),
                             no_worse=int((P[c] <= 1).sum()), better=int((P[c] < -1).sum())) for c in cols}).T
summ.round(3).to_csv(OUT / "summary.csv")

pd.set_option("display.width", 250)
print("=== per run: change in test MWQL vs zero-shot (%) ===")
print(P.round(2).to_string(index=False))
print("\n=== summary over 36 runs ===")
print(summ.round(2).to_string())
print("\n=== LF / HF and old 28 / new 8 (mean / worst) ===")
for c in cols:
    print(f"{c:30s}", "  ".join(f"{'HF' if h else 'LF'}({len(g)}): {g[c].mean():+.2f}/{g[c].max():+.2f}" for h, g in P.groupby("hf")),
          "  ", "  ".join(f"{'new8' if n else 'old28'}: {g[c].mean():+.2f}/{g[c].max():+.2f}" for n, g in P.groupby("new8")))

print("\n=== pre-registered criteria (§20) ===")
lw = P["LoRA (a=1)"].max()
for c in RULES:
    print(f"K1 {c:30s} worst {P[c].max():+8.2f} vs LoRA {lw:+.2f}: {'PASS' if P[c].max() <= lw else 'FAIL'}")
a = P["A: guard-LF m=2%"]
nw = int((a <= 1).sum())
print(f"K2 rule A: no worse {nw}/36 (need >= 34), mean {a.mean():+.3f} (need <= 0): "
      f"{'PASS' if nw >= 34 and a.mean() <= 0 else 'FAIL'}")
print(f"K3 +wise mean {P['+wise'].mean():+.3f} vs LoRA mean {P['LoRA (a=1)'].mean():+.3f}: "
      f"{'PASS' if P['+wise'].mean() < P['LoRA (a=1)'].mean() else 'FAIL'}")

t1 = P["LoRA (a=1)"]
agree = np.sign(P.val_gain_a1) == np.sign(t1)
print("\n=== validation vs test sign agreement at alpha=1 ===")
print(f"all {agree.sum()}/36; LF {agree[~P.hf].sum()}/{(~P.hf).sum()}; HF {agree[P.hf].sum()}/{P.hf.sum()}")
print("\n=== coverage (mean over configurations; nominal 0.80 / 0.60) ===")
cc = ["cov80", "width80", "cqr_cov80", "cqr_width80", "cov60", "width60", "cqr_cov60", "cqr_width60"]
print(V.groupby("at")[cc].mean().round(3).to_string())
print(V.groupby(["hf", "at"])[["cov80", "cqr_cov80", "cov60", "cqr_cov60"]].mean().round(3).to_string())
print("CQR NaN rows:", int(V.cqr_cov80.isna().sum()))
