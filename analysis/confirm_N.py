"""§15: evaluate the pre-registered §14.3 confirmation (8 new configurations x 2 models, LoRA full, seed 0).

Rules are computed offline from the alpha curves exactly as in rules_v2.py:
  A = pick(m=0.02, hf_zero=True); B = pick(m=0.02, m_hf=0.10, cap=0.5); HF = base frequency H, T or S.
Run: python analysis/confirm_N.py      Output: results/confirm_N/
"""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "confirm_N"
OUT.mkdir(exist_ok=True)

C = pd.concat([pd.read_csv(RES / "chr_N/alpha_curves.csv").assign(model="chronos2"),
               pd.read_csv(RES / "tfm_N/alpha_curves.csv").assign(model="timesfm")])
C = C[C.method == "lora_full"].sort_values(["model", "dataset", "alpha"])


def is_hf(d):
    f = d.split("/")[-1] if "/" in d else "D"   # restaurant has no suffix: daily
    return f.endswith(("H", "T", "S"))


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
    "ref: guard-LF m=3% (frozen)": dict(m=0.03, hf_zero=True),
    "ref: +guard m=3%": dict(m=0.03),
    "ref: +wise": dict(m=0.0, last=False),
}

rows, cov = [], []
for (mdl, d), g in C.groupby(["model", "dataset"], sort=False):
    r = dict(a=g.alpha.values, v=g.val_MWQL.values, v1=g.val1_MWQL.values, t=g.test_MWQL.values, hf=is_hf(d))
    ch = lambda al: 100 * (r["t"][np.searchsorted(r["a"], al)] / r["t"][0] - 1)
    row = dict(model=mdl, dataset=d, hf=r["hf"], **{"LoRA (a=1)": ch(1.0)})
    sel = {}
    for name, p in RULES.items():
        al = pick(r, **p)
        sel[name] = al
        row[name] = ch(al)
        row["alpha " + name.split(":")[0]] = al
    row["oracle"] = 100 * (r["t"].min() / r["t"][0] - 1)
    row["val_gain_a1"] = 100 * (r["v"][-1] / r["v"][0] - 1)
    rows.append(row)
    for lab, al in [("a=0", 0.0), ("a=1", 1.0), ("A", sel["A: guard-LF m=2%"]), ("B", sel["B: m_LF 2%/m_HF 10%/cap 0.5"])]:
        x = g.iloc[np.searchsorted(r["a"], al)]
        cov.append(dict(model=mdl, dataset=d, at=lab, alpha=al, test_cov80=x.test_cov80, test_width80=x.test_width80,
                        test_cov60=x.test_cov60, test_width60=x.test_width60))

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
print("\n=== summary over 16 runs ===")
print(summ.round(2).to_string())
print("\n=== per model (mean / worst) ===")
for c in cols:
    print(f"{c:32s}", "  ".join(f"{m}: {g[c].mean():+.2f}/{g[c].max():+.2f}" for m, g in P.groupby("model")),
          "  ", "  ".join(f"{'HF' if h else 'LF'}: {g[c].mean():+.2f}/{g[c].max():+.2f}" for h, g in P.groupby("hf")))

ref = P["ref: guard-LF m=3% (frozen)"].mean()
print("\n=== pre-registered criteria (§14.3) ===")
for c in ["A: guard-LF m=2%", "B: m_LF 2%/m_HF 10%/cap 0.5"]:
    x = P[c]
    c1, c2, c3 = (x <= 1).sum() >= 15, x.mean() <= ref + 1e-12, x.max() <= 1
    print(f"{c}: C1 {(x <= 1).sum()}/16 {'PASS' if c1 else 'FAIL'} | C2 mean {x.mean():+.3f} vs {ref:+.3f} "
          f"{'PASS' if c2 else 'FAIL'} | C3 worst {x.max():+.3f} {'PASS' if c3 else 'FAIL'}")

t1 = P["LoRA (a=1)"]
agree = np.sign(P.val_gain_a1) == np.sign(t1)
print("\n=== validation vs test sign agreement at alpha=1 ===")
print(f"all {agree.sum()}/16; LF {agree[~P.hf].sum()}/{(~P.hf).sum()}; HF {agree[P.hf].sum()}/{P.hf.sum()}")
print("\n=== coverage (mean over 16 runs; nominal 0.80 / 0.60) ===")
print(V.groupby("at")[["test_cov80", "test_width80", "test_cov60", "test_width60"]].mean().round(3).to_string())
print(V.groupby(["model", "at"])[["test_cov80", "test_cov60"]].mean().round(3).to_string())
