"""Appendix table: per-configuration change in test MWQL vs zero-shot (%) for plain LoRA and rule A (with its alpha),
for the three models. Source: results/stats108/per_run_108.csv (run stats108.py first).
Run: python analysis/appendix_table.py      Output: figures/appendix_table.tex
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = pd.read_csv(ROOT / "results/stats108/per_run_108.csv")
GROUPS = [
    ("Design", ["us_births/D", "saugeenday/D", "ett1/W", "ett2/W", "solar/W", "hierarchical_sales/W",
                "bizitobs_l2c/H", "ett1/D", "ett2/D", "M_DENSE/D", "LOOP_SEATTLE/D", "jena_weather/D"]),
    ("Hold-out", ["hospital", "covid_deaths", "car_parts_with_missing", "m4_weekly", "bizitobs_service",
                  "m4_hourly", "SZ_TAXI/H", "kdd_cup_2018_with_missing/D", "bizitobs_application"]),
    ("Fresh", ["electricity/H", "electricity/15T", "electricity/W", "electricity/D", "m4_daily",
               "bitbrains_rnd/H", "bitbrains_fast_storage/H"]),
    ("Confirmation", ["restaurant", "hierarchical_sales/D", "solar/D", "us_births/W", "ett1/H", "M_DENSE/H",
                      "kdd_cup_2018_with_missing/H", "jena_weather/H"]),
]
MODELS = ["Chronos-2", "TimesFM 2.5", "Chronos-Bolt"]
SHORT = {"car_parts_with_missing": "car_parts", "kdd_cup_2018_with_missing/D": "kdd_cup_2018/D",
         "kdd_cup_2018_with_missing/H": "kdd_cup_2018/H", "bitbrains_fast_storage/H": "bitbrains_fast/H"}


def tex(name):
    return SHORT.get(name, name).replace("_", r"\_")


def num(x):
    if abs(x) < 0.005:
        return "0.0"
    s = f"{x:+.1f}"
    return f"${s}$"


def a_cell(chg, alpha):
    if alpha == 0:
        return r"0.0 \hfill(0)"
    a = f"{alpha:g}"
    s = num(chg)
    return (r"\textbf{" + s + "}" if chg > 1 else s) + rf" \hfill({a})"


assert sum(len(v) for _, v in GROUPS) == 36 and set(sum((v for _, v in GROUPS), [])) == set(P.dataset)
lines = [r"\begin{table*}[width=\textwidth,cols=8,pos=p]",
         r"\caption{Change in test MWQL against zero-shot (\%) for every configuration and model: plain LoRA ($\alpha = 1$) "
         r"and rule A, with the weight $\alpha$ that rule A selected in brackets. Negative values are improvements; "
         r"bold marks the two runs in which rule A was more than 1\% worse than zero-shot. HF: hourly or finer "
         r"(rule A keeps the zero-shot model).}\label{tab:perconfig}",
         r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}llrp{1.55cm}rp{1.55cm}rp{1.55cm}@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Chronos-2} & \multicolumn{2}{c}{TimesFM 2.5} & \multicolumn{2}{c}{Chronos-Bolt} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Group & Configuration & LoRA & A ($\alpha$) & LoRA & A ($\alpha$) & LoRA & A ($\alpha$) \\", r"\midrule"]
for gi, (g, ds) in enumerate(GROUPS):
    for k, d in enumerate(ds):
        cells = [g if k == 0 else "", tex(d) + (r"$^{\mathrm{HF}}$" if P[P.dataset == d].hf.iloc[0] else "")]
        for m in MODELS:
            r = P[(P.dataset == d) & (P.model == m)].iloc[0]
            cells += [num(r.LoRA), a_cell(r.A, r.alpha_A)]
        lines.append(" & ".join(cells) + r" \\")
    if gi < len(GROUPS) - 1:
        lines.append(r"\midrule")
lines.append(r"\midrule")
mean = ["", r"\textit{Mean}"]
for m in MODELS:
    g = P[P.model == m]
    mean += [num(g.LoRA.mean()), num(g.A.mean())]
lines += [" & ".join(mean) + r" \\", r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
out = ROOT / "figures/appendix_table.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("->", out, len(lines), "lines")
