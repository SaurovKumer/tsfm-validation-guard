"""Save one scraped TimesFM part (Kaggle output) as results/tfm_<NAME>/{results.csv, alpha_curves_compact.csv}.

Usage: python analysis/tfm_save_part.py NAME RAW_RESULTS RAW_CURVES [FILL_JSON]

RAW_RESULTS rows: dataset,method,seed,alpha,fit_sec,MWQL,MASE,MAPE  (method "L..." = "lora_full...")
RAW_CURVES rows:  dataset,alpha,V,val_MWQL,val1_MWQL,test_MWQL,test_MASE,test_MAPE
FILL_JSON (optional): rows the Kaggle data viewer did not return, rebuilt from the alpha curves.
  {"dataset": {"alpha": {"lora_full+wise": 0.0, ...}, "zero_shot_rec_change_pct": 16.2}}
Dataset metadata (H, windows, n_series, n_obs) is taken from the Chronos-2 result files, which use the same splits.
"""
import json
import sys
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
name, raw_r, raw_c = sys.argv[1:4]
OUT = RES / f"tfm_{name}"
OUT.mkdir(exist_ok=True)

meta = (pd.concat([pd.read_csv(p) for p in RES.glob("*/results.csv") if not p.parent.name.startswith("tfm")])
        .drop_duplicates("dataset").set_index("dataset")[["H", "windows", "n_series", "n_obs"]])

r = pd.read_csv(raw_r, header=None, names=["dataset", "method", "seed", "alpha", "fit_sec", "MWQL", "MASE", "MAPE"])
r["method"] = r.method.str.replace(r"^L", "lora_full", regex=True)
c = pd.read_csv(raw_c, header=None, names=["dataset", "alpha", "V", "val_MWQL", "val1_MWQL", "test_MWQL", "test_MASE", "test_MAPE"])

if len(sys.argv) > 4:  # rebuild rows missing from the scraped results table
    for d, spec in json.loads(Path(sys.argv[4]).read_text()).items():
        g = c[c.dataset == d].set_index("alpha")
        zs = g.loc[0.0]
        add = [dict(dataset=d, method="zero_shot", seed=-1, alpha=0.0, fit_sec=0, MWQL=zs.test_MWQL, MASE=zs.test_MASE, MAPE=zs.test_MAPE),
               dict(dataset=d, method="zero_shot_rec", seed=-1, alpha=0.0, fit_sec=0,
                    MWQL=zs.test_MWQL * (1 + spec["zero_shot_rec_change_pct"] / 100), MASE=None, MAPE=None)]
        for m, a in {"lora_full": 1.0, **spec["alpha"]}.items():
            row = g.loc[a]
            add.append(dict(dataset=d, method=m, seed=0, alpha=a, fit_sec=None, MWQL=row.test_MWQL, MASE=row.test_MASE, MAPE=row.test_MAPE))
        r = pd.concat([r, pd.DataFrame(add)], ignore_index=True)

r = r.join(meta, on="dataset")
order = list(dict.fromkeys(r.dataset))
r["dataset"] = pd.Categorical(r.dataset, order)
mo = ["seasonal_naive", "zero_shot", "zero_shot_rec", "lora_full", "lora_full+wise", "lora_full+wise1", "lora_full+guard", "lora_full+guardLF"]
r["method"] = pd.Categorical(r.method, mo)
r = r.sort_values(["dataset", "method"])
r[["dataset", "H", "windows", "n_series", "n_obs", "method", "seed", "alpha", "fit_sec", "MWQL", "MASE", "MAPE"]].to_csv(OUT / "results.csv", index=False)
c.insert(1, "method", "lora_full")
c.to_csv(OUT / "alpha_curves_compact.csv", index=False)
print(f"saved {OUT}: {len(r)} result rows, {len(c)} curve rows, {r.dataset.nunique()} datasets")
