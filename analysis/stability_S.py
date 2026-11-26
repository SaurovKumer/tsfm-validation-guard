"""§18: stability run (registered in §17). Seeds 0-2 at 4000 steps and seed 0 at 500 / 1000 / 4000 steps,
both models, on us_births/D, ett1/W, solar/W and bizitobs_service. Rules are computed offline from the alpha curves
exactly as in rules_v2.py / confirm_N.py.
Run: python analysis/stability_S.py      Output: results/stability_S/
"""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "stability_S"
OUT.mkdir(exist_ok=True)
DS = ["us_births/D", "ett1/W", "solar/W", "bizitobs_service"]
HF = {"bizitobs_service"}
COLS = ["dataset", "method", "seed", "alpha", "val_MWQL", "val1_MWQL", "test_MWQL"]


def load(path, model, steps=None):
    d = pd.read_csv(path)
    if "seed" not in d:
        d["seed"] = 0
    d = d[d.dataset.isin(DS)][COLS].copy()
    d["model"] = model
    d["steps"] = d.method.map({"lora_s500": 500, "lora_s1000": 1000}).fillna(4000).astype(int)
    return d[d.method.isin(["lora_full", "lora_s500", "lora_s1000"])]


C = pd.concat([
    load(RES / "partA/alpha_curves_compact.csv", "Chronos-2"),                  # seed 0, 4000 steps
    load(RES / "holdoutH1/alpha_curves_compact.csv", "Chronos-2"),
    load(RES / "confirmA/alpha_curves.csv", "Chronos-2"),                       # seeds 1-2, 4000 steps
    load(RES / "chr_S/alpha_curves.csv", "Chronos-2"),                          # 500/1000 steps; bizitobs seeds 1-2
    load(RES / "tfm_A/alpha_curves_compact.csv", "TimesFM 2.5"),
    load(RES / "tfm_BH/alpha_curves_compact.csv", "TimesFM 2.5"),
    load(RES / "tfm_S/alpha_curves.csv", "TimesFM 2.5"),
]).drop_duplicates(["model", "dataset", "steps", "seed", "alpha"]).sort_values(["model", "dataset", "steps", "seed", "alpha"])


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


RULES = {"A": dict(m=0.02, hf_zero=True), "B": dict(m=0.02, m_hf=0.10, cap=0.5),
         "guard m3": dict(m=0.03), "wise": dict(m=0.0, last=False)}
rows = []
for (mdl, d, st, sd), g in C.groupby(["model", "dataset", "steps", "seed"]):
    if len(g) != 6:
        continue
    r = dict(a=g.alpha.values, v=g.val_MWQL.values, v1=g.val1_MWQL.values, t=g.test_MWQL.values, hf=d in HF)
    ch = lambda al: 100 * (r["t"][np.searchsorted(r["a"], al)] / r["t"][0] - 1)
    row = dict(model=mdl, dataset=d, steps=st, seed=sd, LoRA=ch(1.0), val_gain_a1=100 * (r["v"][-1] / r["v"][0] - 1),
               oracle=100 * (r["t"].min() / r["t"][0] - 1))
    for k, p in RULES.items():
        al = pick(r, **p)
        row[k] = ch(al); row["a_" + k] = al
    rows.append(row)
P = pd.DataFrame(rows)
P.round(3).to_csv(OUT / "per_run.csv", index=False)
pd.set_option("display.width", 250)

seeds = P[P.steps == 4000]
print("=== S1: seeds 0-2 at 4000 steps (change vs zero-shot %, chosen alpha) ===")
print(seeds[["model", "dataset", "seed", "LoRA", "A", "a_A", "B", "a_B", "guard m3", "a_guard m3", "wise", "a_wise", "oracle"]]
      .round(2).to_string(index=False))
stab = seeds.groupby(["model", "dataset"]).agg(n=("seed", "nunique"), A_alphas=("a_A", lambda x: sorted(set(x))),
                                                B_alphas=("a_B", lambda x: sorted(set(x))),
                                                A_worst=("A", "max"), LoRA_min=("LoRA", "min"), LoRA_max=("LoRA", "max"))
print(stab.round(2).to_string())
print("\n=== S2: fine-tuning length, seed 0 ===")
s2 = P[P.seed == 0].pivot_table(index=["model", "dataset"], columns="steps", values=["LoRA", "A", "wise", "oracle"])
print(s2.round(2).to_string())
print("\n=== S3: bizitobs_service by seed (4000 steps) ===")
print(seeds[seeds.dataset == "bizitobs_service"][["model", "seed", "val_gain_a1", "LoRA", "guard m3", "a_guard m3", "A", "wise"]]
      .round(2).to_string(index=False))
n = len(P)
print(f"\nall {n} runs: A worst {P.A.max():+.2f}, B worst {P.B.max():+.2f}, guard m3 worst {P['guard m3'].max():+.2f}, "
      f"wise worst {P.wise.max():+.2f}, LoRA worse than ZS in {(P.LoRA > 1).sum()}/{n}")
