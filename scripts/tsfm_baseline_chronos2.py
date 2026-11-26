# %% [markdown]
# # Phase 1: Chronos-2 fine-tuning baseline on GIFT-Eval small datasets
#
# **Goal:**
# 1. Reproduce the "small datasets" result in Table 4 of Laglil et al. (2026), arXiv:2607.23146:
#    zero-shot vs full fine-tuning (FFT) vs LoRA full-shot vs LoRA few-shot (20%).
# 2. First test of our core idea: **interpolating weights between the zero-shot and fine-tuned models**,
#    θ(α) = (1-α)·θ_zero-shot + α·θ_fine-tuned, with α **selected on a held-out validation window** (never on test).
#
# **Kaggle settings:** Accelerator = **GPU T4 x1 or P100**, Internet = **On** (needs a phone-verified account).
# Use *Save Version -> Save & Run All (Commit)* so the run continues in the background.
# Results are appended to `/kaggle/working/results/*.csv` after every run. If the session stops,
# re-running skips runs that are already finished.
#
# **Protocol (follows GIFT-Eval):** short-term horizon; prediction length from the frequency (D=30, W=8, H=48, ...);
# number of test windows = min(max(1, ceil(0.1·min_len/H)), 20); training split ends (windows+1)·H before the series end;
# the validation window is the H points between the training split and the first test window.
# Metrics are normalised by Seasonal Naive per dataset, then averaged (as in the paper).

# %%
import os, sys, subprocess
SMOKE = os.environ.get("TSFM_SMOKE", "0") == "1"   # local CPU smoke test only
IN_KAGGLE = os.path.exists("/kaggle")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if IN_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "chronos-forecasting>=2.1", "peft", "gluonts", "datasets", "huggingface_hub"], check=True)
    # Kaggle ships torchao 0.10, which peft rejects ("incompatible version of torchao") -> LoRA fails.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

# %%
import copy, gc, json, math, shutil, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import datasets
from huggingface_hub import snapshot_download
from gluonts.time_feature import get_seasonality, norm_freq_str
from pandas.tseries.frequencies import to_offset
from chronos import Chronos2Pipeline
from transformers import TrainerCallback

warnings.filterwarnings("ignore")
assert hasattr(Chronos2Pipeline, "fit"), "chronos-forecasting is too old: Chronos2Pipeline.fit is missing"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__)
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# %% [markdown]
# ## Configuration
# Hyperparameters follow Table 3 of the base paper (Chronos-2): FFT lr 1e-4, LoRA lr 1e-5, batch 32,
# 4000 steps, LoRA r=8, alpha=16. Note: the paper used AutoGluon; here we call `Chronos2Pipeline.fit` directly
# (the same underlying trainer). Few-shot = the last 20% of each training series (at least 3·H points so the
# series is not dropped by the trainer).

# %%
WORK = Path("/kaggle/working") if IN_KAGGLE else Path(os.environ.get("TSFM_WORK", "./tsfm_work"))
TMP = Path("/kaggle/temp") if IN_KAGGLE else WORK / "tmp"   # data + checkpoints, not saved as output
DATA_DIR = TMP / "gift_eval_data"
RES_DIR = WORK / "results"
CKPT_DIR = TMP / "ckpt"
for d in (DATA_DIR, RES_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# GIFT-Eval configurations labelled [Small] in GIFT-Eval (Table 2 of the base paper), short-term horizon.
DATASETS = [
    "ett1/D", "ett2/D", "ett1/W", "ett2/W",
    "us_births/D", "saugeenday/D", "jena_weather/D",
    "solar/W", "hierarchical_sales/W", "M_DENSE/D",
    "LOOP_SEATTLE/D", "bizitobs_l2c/H",
]
TERM = "short"
SEEDS = [0]                      # use [0, 1, 2] for the final paper runs
MODES = ["fft", "lora_full", "lora_few"]
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
HP = dict(
    fft=dict(finetune_mode="full", learning_rate=1e-4),
    lora_full=dict(finetune_mode="lora", learning_rate=1e-5),
    lora_few=dict(finetune_mode="lora", learning_rate=1e-5),
)
NUM_STEPS, BATCH_SIZE, FEWSHOT_FRAC = 4000, 32, 0.2
# Max context during fine-tuning. 8192 (default) ran out of memory on a 16 GB T4; 2048 cost ~1.1 s/step on
# long series (us_births/D: 569 s per 500 steps), i.e. >12 h for the full run. 1024 keeps each part < ~6 h.
FT_CONTEXT = 1024
# Number of held-out validation windows (each H long) between the training split and the test windows.
# The quick test showed one window is too noisy to pick alpha (us_births/D lora_full: val picked 0, test best at 1).
VAL_WINDOWS = 3
MODEL_ID = "amazon/chronos-2"

# The full run is split into two parts so each Kaggle session stays well under the 12 h limit.
PART = None    # None = all datasets, "A" or "B" = one half
PARTS = {"A": ["us_births/D", "saugeenday/D", "ett1/W", "ett2/W", "solar/W", "hierarchical_sales/W"],
         "B": ["bizitobs_l2c/H", "ett1/D", "ett2/D", "M_DENSE/D", "LOOP_SEATTLE/D", "jena_weather/D"]}
# Confirmatory run (pre-registered 2026-11-19): only LoRA full, new seeds 1 and 2, same datasets/settings.
# Primary method fixed in advance: "+guard" = both_improve rule with margin GUARD_M (chosen by LODO on seed 0).
CONFIRM = False
GUARD_M = 0.03
if CONFIRM:
    MODES, SEEDS = ["lora_full"], [1, 2]
# Dataset-level hold-out (pre-registered 2026-11-20): GIFT-Eval configurations never used to design the rule
# (no overlap in source data with the 12 small configurations). Same frozen rule, LoRA full, seed 0.
HOLDOUT = False
PARTS.update({
    "H1": ["hospital", "covid_deaths", "car_parts_with_missing", "m4_weekly", "bizitobs_service"],
    "H2": ["m4_hourly", "SZ_TAXI/H", "kdd_cup_2018_with_missing/D", "bizitobs_application"],
})
if HOLDOUT:
    MODES, SEEDS = ["lora_full"], [0]
# Fresh-source confirmation of guard-LF (pre-registered 2026-11-20): sources never used before.
FRESH = False
PARTS.update({
    "F1": ["electricity/H", "electricity/15T", "electricity/W", "bitbrains_rnd/H"],
    "F2": ["electricity/D", "m4_daily", "bitbrains_fast_storage/H"],
})
if FRESH:
    MODES, SEEDS = ["lora_full"], [0]
# New-configuration confirmation of the §14 rules (pre-registered 2026-11-22): 8 GIFT-Eval configurations not used
# by any earlier run. LoRA full, seed 0; the alpha curves now also record interval coverage (RQ4).
NEWCFG = False
PARTS.update({"N": ["restaurant", "hierarchical_sales/D", "solar/D", "us_births/W",
                    "ett1/H", "M_DENSE/H", "kdd_cup_2018_with_missing/H", "jena_weather/H"]})
if NEWCFG:
    MODES, SEEDS = ["lora_full"], [0]
# Stability run (2026-11-22): fine-tuning-length ablation (LoRA full with 500 / 1000 steps, seed 0; 4000 steps
# already exist) on 4 datasets, plus seeds 1-2 for bizitobs_service (the other three already have seeds 0-2, §6).
STAB = False
PARTS["S"] = ["us_births/D", "ett1/W", "solar/W", "bizitobs_service"]
STEPS = {"lora_s500": 500, "lora_s1000": 1000}   # LoRA-full variants with fewer steps; other modes use NUM_STEPS
HP.update(lora_s500=HP["lora_full"], lora_s1000=HP["lora_full"])
SEEDS_BY_MODE, MODE_DATASETS = {}, {}             # optional per-mode seeds / per-mode dataset restriction
if STAB:
    MODES = ["lora_s500", "lora_s1000", "lora_full"]
    SEEDS_BY_MODE = {"lora_full": [1, 2], "lora_s500": [0], "lora_s1000": [0]}
    MODE_DATASETS = {"lora_full": {"bizitobs_service"}}
# Coverage re-run (2026-11-22): the 28 configurations again (LoRA full, seed 0), now recording interval coverage
# and conformal (CQR) recalibration for every alpha (RQ4). Also a reproducibility check of the seed-0 results.
COVER = False
PARTS["C1"] = PARTS["A"] + PARTS["B"] + PARTS["H1"]
PARTS["C2"] = PARTS["H2"] + PARTS["F1"] + PARTS["F2"]
if COVER:
    MODES, SEEDS = ["lora_full"], [0]
# Interpolation vs early stopping (pre-registered 2026-11-24, §22): the LoRA update is also recorded at steps
# CKPT_STEPS of the same 4000-step run, and theta0 + alpha * dLoRA(step) is scored for alpha in ES_ALPHAS.
# All 36 configurations in three parts balanced by fine-tuning time.
ES = os.environ.get("TSFM_ES", "0") == "1"   # True in the tsfm-*-E notebooks
CKPT_STEPS = [500, 1000, 2000]
ES_ALPHAS = [0.25, 0.5, 0.75, 1.0]
PARTS["E1"] = ["SZ_TAXI/H", "bizitobs_l2c/H", "bizitobs_service", "covid_deaths", "electricity/H", "hospital",
               "jena_weather/D", "m4_daily", "restaurant", "solar/W", "us_births/D", "us_births/W"]
PARTS["E2"] = ["M_DENSE/D", "bitbrains_rnd/H", "bizitobs_application", "electricity/15T", "ett1/D", "ett1/W",
               "hierarchical_sales/W", "jena_weather/H", "kdd_cup_2018_with_missing/D", "m4_weekly", "saugeenday/D",
               "solar/D"]
PARTS["E3"] = ["LOOP_SEATTLE/D", "M_DENSE/H", "bitbrains_fast_storage/H", "car_parts_with_missing", "electricity/D",
               "electricity/W", "ett1/H", "ett2/D", "ett2/W", "hierarchical_sales/D", "kdd_cup_2018_with_missing/H",
               "m4_hourly"]
if ES:
    MODES, SEEDS = ["lora_full"], [0]
if PART:
    DATASETS = PARTS[PART]
    tag = "es_" if ES else "confirm_" if CONFIRM else ("holdout_" if HOLDOUT else ("fresh_" if FRESH else ("newcfg_" if NEWCFG else ("stab_" if STAB else ("cover_" if COVER else "")))))
    RES_DIR = WORK / f"results_{tag}{PART}"
    RES_DIR.mkdir(parents=True, exist_ok=True)

QUICK = False   # True = first check on Kaggle: 2 datasets, 500 steps (~15-20 min)
if QUICK:
    DATASETS, NUM_STEPS = ["ett1/W", "us_births/D"], 500
    RES_DIR = WORK / "results_quick"
    shutil.rmtree(RES_DIR, ignore_errors=True)   # the quick test always starts fresh
    RES_DIR.mkdir(parents=True, exist_ok=True)

if SMOKE:   # tiny settings for a CPU check of the code path
    DATASETS, NUM_STEPS, BATCH_SIZE, ALPHAS = ["ett1/W"], 3, 4, [0.0, 0.5, 1.0]
    CKPT_STEPS, ES_ALPHAS = [1, 2], [0.5, 1.0]

# %% [markdown]
# ## Download only the selected GIFT-Eval folders

# %%
snapshot_download(repo_id="Salesforce/GiftEval", repo_type="dataset", local_dir=str(DATA_DIR),
                  allow_patterns=[f"{d}/*" for d in DATASETS])

# %% [markdown]
# ## Data loading (replicates `gift_eval.data.Dataset`, univariate)

# %%
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
TERM_MULT = {"short": 1, "medium": 10, "long": 15}
NEW_TO_OLD = {"ME": "M", "MS": "M", "QE": "Q", "QS": "Q", "YE": "A", "Y": "A", "YS": "A",
              "h": "H", "min": "T", "s": "S", "D": "D", "W": "W"}

def base_freq(freq: str) -> str:
    f = norm_freq_str(to_offset(freq).name)
    return NEW_TO_OLD.get(f, f)

def load_config(name: str, term: str = TERM) -> dict:
    hf = datasets.load_from_disk(str(DATA_DIR / name)).with_format("numpy")
    freq = str(hf[0]["freq"])
    series = []
    for row in hf:
        tgt = np.asarray(row["target"], dtype=np.float32)
        if tgt.ndim == 1:
            series.append(tgt)
        else:                                   # multivariate -> univariate
            series.extend([np.asarray(v, dtype=np.float32) for v in tgt])
    bf = base_freq(freq)
    H = (M4_PRED_LENGTH_MAP if "m4" in name.lower() else PRED_LENGTH_MAP)[bf] * TERM_MULT[term]
    min_len = min(len(s) for s in series)
    windows = 1 if "m4" in name.lower() else min(max(1, math.ceil(0.1 * min_len / H)), 20)
    season = get_seasonality(freq)
    # Use VAL_WINDOWS validation windows, but keep at least 3*H points of training data per series.
    V = max(1, min(VAL_WINDOWS, (min_len - windows * H - 3 * H) // H))
    train, val_ctx, val_lab, test_ctx, test_lab, val_last = [], [], [], [], [], []
    for y in series:
        L = len(y)
        n_tr = L - (windows + V) * H
        train.append(y[:n_tr])
        for v in range(V):
            c = n_tr + v * H
            val_ctx.append(y[:c]); val_lab.append(y[c:c + H]); val_last.append(v == V - 1)
        for w in range(windows):
            c = L - (windows - w) * H
            test_ctx.append(y[:c]); test_lab.append(y[c:c + H])
    return dict(name=name, freq=freq, base_freq=bf, H=H, windows=windows, season=season, train=train, V=V,
                val_ctx=val_ctx, val_lab=val_lab, val_last=np.array(val_last),
                test_ctx=test_ctx, test_lab=test_lab,
                n_series=len(series), n_obs=int(sum(len(s) for s in series)))

def fewshot(train, H, frac=FEWSHOT_FRAC):
    return [s[-max(int(frac * len(s)), 3 * H):] for s in train]

# %% [markdown]
# ## Metrics: MWQL (CRPS proxy, as in GIFT-Eval), MASE and MAPE on the median

# %%
def mwql(qpred: np.ndarray, y: np.ndarray, qs=QUANTILES) -> float:
    """qpred: (N, H, Q); y: (N, H). Dataset-level weighted quantile loss, mean over quantiles."""
    m = np.isfinite(y)
    denom = np.abs(y[m]).sum()
    out = []
    for k, q in enumerate(qs):
        e = y - qpred[..., k]
        ql = np.where(e >= 0, q * e, (q - 1) * e)
        out.append(2 * ql[m].sum() / denom)
    return float(np.mean(out))

def mase(med, y, ctxs, season):
    vals = []
    for i, c in enumerate(ctxs):
        c = c[np.isfinite(c)]
        lag = season if len(c) > season else 1
        if len(c) <= lag:
            continue
        scale = np.mean(np.abs(c[lag:] - c[:-lag]))
        m = np.isfinite(y[i])
        if scale > 0 and m.any():
            vals.append(np.mean(np.abs(y[i][m] - med[i][m])) / scale)
    return float(np.mean(vals)) if vals else np.nan

def mape(med, y):
    vals = []
    for i in range(len(y)):
        m = np.isfinite(y[i]) & (y[i] != 0)
        if m.any():
            vals.append(100 * np.mean(np.abs((y[i][m] - med[i][m]) / y[i][m])))
    return float(np.mean(vals)) if vals else np.nan

def score(qpred, labels, ctxs, season):
    y = np.stack(labels).astype(np.float64)
    med = qpred[..., QUANTILES.index(0.5)]
    return dict(MWQL=mwql(qpred, y), MASE=mase(med, y, ctxs, season), MAPE=mape(med, y))

def interval_stats(qpred, labels, prefix):
    """Calibration (RQ4): empirical coverage of the central 80% (q0.1-q0.9) and 60% (q0.2-q0.8) intervals,
    and their width relative to the mean |y|. Reported only; no selection rule uses it."""
    y = np.stack(labels).astype(np.float64)
    m = np.isfinite(y)
    out = {}
    for nom, lo, hi in ((80, 0.1, 0.9), (60, 0.2, 0.8)):
        l, h = qpred[..., QUANTILES.index(lo)], qpred[..., QUANTILES.index(hi)]
        out[f"{prefix}_cov{nom}"] = float(((y >= l) & (y <= h))[m].mean())
        out[f"{prefix}_width{nom}"] = float((h - l)[m].sum() / np.abs(y[m]).sum())
    return out


def conformal_stats(vq, vlab, vctx, tq, tlab, tctx):
    """Split-conformal (CQR) recalibration of the central intervals, fitted on the validation windows and
    evaluated on the test windows. Scores are scaled per window by the mean |y| of its context so that
    series of different scale can be pooled. Reported only (RQ4); no selection rule uses it."""
    def scale(ctx):
        s = np.array([np.nanmean(np.abs(np.asarray(c, dtype=np.float64))) if np.isfinite(c).any() else np.nan for c in ctx])
        return np.where(np.isfinite(s) & (s > 1e-8), s, 1.0)[:, None]   # all-NaN or zero context -> unit scale
    vy, ty = np.stack(vlab).astype(np.float64), np.stack(tlab).astype(np.float64)
    vs, ts = scale(vctx), scale(tctx)
    vm, tm = np.isfinite(vy), np.isfinite(ty)
    out = {}
    for nom, lo, hi in ((80, 0.1, 0.9), (60, 0.2, 0.8)):
        il, ih = QUANTILES.index(lo), QUANTILES.index(hi)
        e = np.maximum(vq[..., il] - vy, vy - vq[..., ih]) / vs
        e = e[vm & np.isfinite(e)]
        n = len(e)
        lvl = min(1.0, np.ceil((n + 1) * nom / 100) / n) if n else 1.0
        q = float(np.quantile(e, lvl, method="higher")) if n else 0.0
        l, h = tq[..., il] - q * ts, tq[..., ih] + q * ts
        out[f"test_cqr_cov{nom}"] = float(((ty >= l) & (ty <= h))[tm].mean())
        out[f"test_cqr_width{nom}"] = float((h - l)[tm].sum() / np.abs(ty[tm]).sum())
        out[f"cqr_q{nom}"] = q
    return out

# %% [markdown]
# ## Forecasting helpers

# %%
def predict_q(pipe, ctxs, H, chunk=256):
    outs = []
    for i in range(0, len(ctxs), chunk):
        batch = [torch.tensor(c, dtype=torch.float32) for c in ctxs[i:i + chunk]]
        with torch.no_grad():
            qs, _ = pipe.predict_quantiles(batch, prediction_length=H, quantile_levels=QUANTILES)
        for q in qs:
            q = q.detach().float().cpu().numpy()
            q = q.reshape(-1, *q.shape[-2:])[0]          # (H, Q) for a univariate input
            if q.shape[0] == len(QUANTILES) and q.shape[1] == H and H != len(QUANTILES):
                q = q.T
            outs.append(q)
    return np.stack(outs).astype(np.float64)              # (N, H, Q)

def seasonal_naive_q(ctxs, H, season):
    preds = []
    for c in ctxs:
        c = pd.Series(c).ffill().bfill().to_numpy()
        if season > 1 and len(c) >= season:
            last = c[-season:]
            p = np.tile(last, math.ceil(H / season))[:H]
        else:
            p = np.repeat(c[-1], H)
        preds.append(np.repeat(p[:, None], len(QUANTILES), axis=1))
    return np.stack(preds).astype(np.float64)

def to_cpu_sd(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

class CkptCallback(TrainerCallback):
    """§22: record the LoRA update (as weight deltas on the base model's keys) at the given training steps."""
    def __init__(self, steps):
        self.steps, self.deltas = set(steps), {}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self.steps:
            out = {}
            for n, m in model.named_modules():
                la = getattr(m, "lora_A", None)
                if isinstance(la, torch.nn.ModuleDict) and "default" in la:
                    d = (m.lora_B["default"].weight @ la["default"].weight) * m.scaling["default"]
                    out[n.removeprefix("base_model.model.") + ".weight"] = d.detach().float().cpu()
            assert out, "no LoRA layers found"
            self.deltas[state.global_step] = out

def es_rows(model, pipe, sd0, deltas, D, meta, mode, seed):
    """§22: score theta0 + alpha * dLoRA(step) for every recorded intermediate step and alpha in ES_ALPHAS."""
    rows, last = [], D["val_last"]
    vy = np.stack(D["val_lab"]).astype(np.float64)
    for st in CKPT_STEPS:
        d = deltas[st]
        for a in ES_ALPHAS:
            te = time.time()
            model.load_state_dict({k: (v + a * d[k] if k in d else v) for k, v in sd0.items()})
            vq = predict_q(pipe, D["val_ctx"], D["H"])
            v = score(vq, D["val_lab"], D["val_ctx"], D["season"])
            v1 = mwql(vq[last], vy[last])
            tq = predict_q(pipe, D["test_ctx"], D["H"])
            t = score(tq, D["test_lab"], D["test_ctx"], D["season"])
            rows.append(dict(**meta, method=mode, seed=seed, alpha=a, V=D["V"],
                             val_MWQL=v["MWQL"], val_MASE=v["MASE"], val1_MWQL=v1,
                             test_MWQL=t["MWQL"], test_MASE=t["MASE"], test_MAPE=t["MAPE"],
                             **interval_stats(vq, D["val_lab"], "val"),
                             **interval_stats(tq, D["test_lab"], "test"),
                             **conformal_stats(vq, D["val_lab"], D["val_ctx"], tq, D["test_lab"], D["test_ctx"]),
                             step=st, eval_sec=time.time() - te))
        print(f"    step {st}: val MWQL " + " ".join(f"a={r['alpha']}:{r['val_MWQL']:.4f}" for r in rows[-len(ES_ALPHAS):]))
    model.load_state_dict(sd0)
    return rows

def check_final_delta(sd0, sd1, d):
    """theta0 + dLoRA(final) must reproduce the merged fine-tuned weights."""
    err = max(float((sd0[k] + d[k] - sd1[k]).abs().max()) if k in d else float((sd0[k] - sd1[k]).abs().max())
              for k in sd1 if sd1[k].is_floating_point())
    print(f"  checkpoint check: max |theta0 + dLoRA(final) - merged| = {err:.2e}"
          + ("" if err < 1e-4 else "  !! WARNING: checkpoint deltas do not reproduce the merged weights"))

def interpolate_into(model, sd0, sd1, alpha):
    new = {}
    for k, v1 in sd1.items():
        v0 = sd0.get(k)
        if v0 is not None and v0.shape == v1.shape and v1.is_floating_point():
            new[k] = (1 - alpha) * v0 + alpha * v1
        else:
            new[k] = v1
    model.load_state_dict(new)
    return model

# %% [markdown]
# ## Result files (resume-safe)

# %%
MAIN_CSV, ALPHA_CSV = RES_DIR / "results.csv", RES_DIR / "alpha_curves.csv"

def done_keys():
    if not MAIN_CSV.exists():
        return set()
    df = pd.read_csv(MAIN_CSV)
    return set(zip(df.dataset, df.method, df.seed))

def append(path, rows):
    df = pd.DataFrame(rows)
    if path.exists():   # keep the column order of the existing file
        cols = pd.read_csv(path, nrows=0).columns
        df = df.reindex(columns=cols)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)

# %% [markdown]
# ## Main loop
# For each dataset: Seasonal Naive, zero-shot, then each fine-tuning mode. For each fine-tuned model we evaluate
# α ∈ ALPHAS on the validation window **and** on the test windows (`alpha_curves.csv`). The method `<mode>+wise`
# uses the α with the lowest validation MWQL (ties go to the smaller α), so the test set is never used for selection.

# %%
base = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=DEVICE)
base.model.eval()
SD0 = to_cpu_sd(base.model)
done = done_keys()

for name in DATASETS:
    D = load_config(name)
    print(f"\n=== {name}: series={D['n_series']} obs={D['n_obs']} H={D['H']} windows={D['windows']} season={D['season']} V={D['V']}")
    meta = dict(dataset=name, H=D["H"], windows=D["windows"], n_series=D["n_series"], n_obs=D["n_obs"])

    if (name, "seasonal_naive", -1) not in done:
        s = score(seasonal_naive_q(D["test_ctx"], D["H"], D["season"]), D["test_lab"], D["test_ctx"], D["season"])
        append(MAIN_CSV, [dict(**meta, method="seasonal_naive", seed=-1, alpha=np.nan, fit_sec=0, **s)])

    if (name, "zero_shot", -1) not in done:
        s = score(predict_q(base, D["test_ctx"], D["H"]), D["test_lab"], D["test_ctx"], D["season"])
        append(MAIN_CSV, [dict(**meta, method="zero_shot", seed=-1, alpha=0.0, fit_sec=0, **s)])
        print("zero-shot", s)

    for mode in MODES:
        if mode in MODE_DATASETS and name not in MODE_DATASETS[mode]:
            continue
        for seed in SEEDS_BY_MODE.get(mode, SEEDS):
            if (name, mode, seed) in done and (name, mode + "+wise", seed) in done:
                continue
            train = fewshot(D["train"], D["H"]) if mode == "lora_few" else D["train"]
            out_dir = CKPT_DIR / f"{name.replace('/', '_')}_{mode}_{seed}"
            torch.manual_seed(seed); np.random.seed(seed)
            t0 = time.time()
            fit_kw = dict(inputs=train, prediction_length=D["H"], num_steps=STEPS.get(mode, NUM_STEPS),
                          batch_size=BATCH_SIZE, output_dir=str(out_dir), context_length=FT_CONTEXT,
                          remove_printer_callback=True, **HP[mode])
            ckpt_cb = CkptCallback(CKPT_STEPS + [STEPS.get(mode, NUM_STEPS)]) if ES else None
            if ES:
                fit_kw["callbacks"] = [ckpt_cb]
            try:
                try:   # seed / report_to are forwarded to transformers.TrainingArguments
                    ft = base.fit(**fit_kw, seed=seed, report_to="none")
                except TypeError as te:
                    if "multiple values" not in str(te):
                        raise
                    ft = base.fit(**fit_kw, seed=seed)
            except Exception as e:
                print(f"  !! fit failed for {mode}: {e!r}")
                append(MAIN_CSV, [dict(**meta, method=mode + "_FAILED", seed=seed, alpha=1.0, fit_sec=np.nan,
                                       MWQL=np.nan, MASE=np.nan, MAPE=np.nan)])
                shutil.rmtree(out_dir, ignore_errors=True)
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue
            fit_sec = time.time() - t0
            model = ft.model.merge_and_unload() if hasattr(ft.model, "merge_and_unload") else ft.model
            model.eval()
            SD1 = to_cpu_sd(model)
            pipe = Chronos2Pipeline(model=model)

            curve = []
            last = D["val_last"]
            vy = np.stack(D["val_lab"]).astype(np.float64)
            for a in ALPHAS:
                te = time.time()
                interpolate_into(model, SD0, SD1, a)
                vq = predict_q(pipe, D["val_ctx"], D["H"])
                v = score(vq, D["val_lab"], D["val_ctx"], D["season"])          # all V windows
                v1 = mwql(vq[last], vy[last])                                    # last window only
                tq = predict_q(pipe, D["test_ctx"], D["H"])
                t = score(tq, D["test_lab"], D["test_ctx"], D["season"])
                curve.append(dict(**meta, method=mode, seed=seed, alpha=a, V=D["V"],
                                  val_MWQL=v["MWQL"], val_MASE=v["MASE"], val1_MWQL=v1,
                                  test_MWQL=t["MWQL"], test_MASE=t["MASE"], test_MAPE=t["MAPE"],
                                  **interval_stats(vq, D["val_lab"], "val"),
                                  **interval_stats(tq, D["test_lab"], "test"),
                                  **conformal_stats(vq, D["val_lab"], D["val_ctx"], tq, D["test_lab"], D["test_ctx"])))
                curve[-1].update(step=STEPS.get(mode, NUM_STEPS), eval_sec=time.time() - te)
            append(ALPHA_CSV, curve)
            if ES:
                check_final_delta(SD0, SD1, ckpt_cb.deltas[STEPS.get(mode, NUM_STEPS)])
                append(ALPHA_CSV, es_rows(model, pipe, SD0, ckpt_cb.deltas, D, meta, mode, seed))
                ckpt_cb.deltas.clear()
            c = pd.DataFrame(curve)
            full = c[c.alpha == 1.0].iloc[0]
            best = c.sort_values(["val_MWQL", "alpha"]).iloc[0]
            best1 = c.sort_values(["val1_MWQL", "alpha"]).iloc[0]   # ablation: 1 validation window
            # Pre-registered guard rule (both_improve): leave alpha=0 only if pooled validation MWQL improves by
            # more than GUARD_M AND the most recent validation window does not get worse.
            c0 = c[c.alpha == 0.0].iloc[0]
            ok = c[(c.alpha == 0.0) | ((c.val_MWQL < (1 - GUARD_M) * c0.val_MWQL) & (c.val1_MWQL <= c0.val1_MWQL))]
            guard = ok.sort_values(["val_MWQL", "alpha"]).iloc[0]
            # guard-LF (pre-registered 2026-11-20): zero-shot for hourly-or-finer data, +guard otherwise.
            high_freq = D["base_freq"] in ("H", "T", "S")
            guard_lf = c0 if high_freq else guard
            append(MAIN_CSV, [
                dict(**meta, method=mode, seed=seed, alpha=1.0, fit_sec=fit_sec,
                     MWQL=full.test_MWQL, MASE=full.test_MASE, MAPE=full.test_MAPE),
                dict(**meta, method=mode + "+wise", seed=seed, alpha=best.alpha, fit_sec=fit_sec,
                     MWQL=best.test_MWQL, MASE=best.test_MASE, MAPE=best.test_MAPE),
                dict(**meta, method=mode + "+wise1", seed=seed, alpha=best1.alpha, fit_sec=fit_sec,
                     MWQL=best1.test_MWQL, MASE=best1.test_MASE, MAPE=best1.test_MAPE),
                dict(**meta, method=mode + "+guard", seed=seed, alpha=guard.alpha, fit_sec=fit_sec,
                     MWQL=guard.test_MWQL, MASE=guard.test_MASE, MAPE=guard.test_MAPE),
                dict(**meta, method=mode + "+guardLF", seed=seed, alpha=guard_lf.alpha, fit_sec=fit_sec,
                     MWQL=guard_lf.test_MWQL, MASE=guard_lf.test_MASE, MAPE=guard_lf.test_MAPE),
            ])
            print(f"  {mode} seed={seed} fit={fit_sec:.0f}s  test MWQL a=1: {full.test_MWQL:.4f} | "
                  f"wise(V={D['V']}) a={best.alpha}: {best.test_MWQL:.4f} | wise1 a={best1.alpha}: {best1.test_MWQL:.4f}"
                  f" | guard a={guard.alpha}: {guard.test_MWQL:.4f}")
            del ft, model, pipe, SD1
            shutil.rmtree(out_dir, ignore_errors=True)
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

# %% [markdown]
# ## Summary: metrics normalised by Seasonal Naive, averaged over datasets
# Paper (Table 4, Chronos-2, small datasets), MWQL_n: zero-shot 0.423 | FFT 0.577 | LoRA-full 0.477 | LoRA-few 0.471.
# MAPE_n: 0.544 | 0.732 | 0.647 | 0.612. Our dataset list and size split differ slightly, so expect the same
# *pattern* (fine-tuning worse than zero-shot on small data), not identical numbers.

# %%
df = pd.read_csv(MAIN_CSV)
failed = df[df.method.str.endswith("_FAILED")]
if len(failed):
    print("FAILED runs:\n", failed[["dataset", "method", "seed"]])
df = df[~df.method.str.endswith("_FAILED")].dropna(subset=["MWQL"])
df = df.drop_duplicates(["dataset", "method", "seed"], keep="last")
sn = df[df.method == "seasonal_naive"].set_index("dataset")[["MWQL", "MASE", "MAPE"]]
rows = []
for _, r in df[df.method != "seasonal_naive"].iterrows():
    if r.dataset in sn.index:
        rows.append(dict(dataset=r.dataset, method=r.method, seed=r.seed, alpha=r.alpha,
                         MWQL_n=r.MWQL / sn.loc[r.dataset, "MWQL"],
                         MASE_n=r.MASE / sn.loc[r.dataset, "MASE"],
                         MAPE_n=r.MAPE / sn.loc[r.dataset, "MAPE"]))
norm = pd.DataFrame(rows)
order = ["zero_shot", "fft", "fft+wise", "fft+wise1", "fft+guard", "lora_full", "lora_full+wise",
         "lora_full+wise1", "lora_full+guard", "lora_full+guardLF", "lora_few", "lora_few+wise", "lora_few+wise1", "lora_few+guard"]
summary = (norm.groupby("method")[["MWQL_n", "MASE_n", "MAPE_n"]].mean()
           .reindex([m for m in order if m in set(norm.method)] + sorted(set(norm.method) - set(order))))
print(summary.round(3))
summary.to_csv(RES_DIR / "summary_normalised.csv")

# Per-dataset view: does fine-tuning beat zero-shot? (negative = better than zero-shot)
zs = norm[norm.method == "zero_shot"].set_index("dataset").MWQL_n
gain = norm[norm.method != "zero_shot"].copy()
gain["rel_change_vs_zero_shot_%"] = 100 * (gain.MWQL_n / gain.dataset.map(zs) - 1)
per_ds = gain.pivot_table(index="dataset", columns="method", values="rel_change_vs_zero_shot_%")
print(per_ds.round(1))
per_ds.to_csv(RES_DIR / "per_dataset_change_vs_zero_shot.csv")
print("\nSelected alpha per dataset:")
print(norm[norm.method.str.contains(r"\+(?:wise|guard)")].pivot_table(index="dataset", columns="method", values="alpha"))
