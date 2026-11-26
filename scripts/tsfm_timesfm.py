# %% [markdown]
# # Second model: TimesFM 2.5 (200M) — zero-shot vs LoRA fine-tuning with validation-selected weight interpolation
#
# Same protocol as the Chronos-2 runs (`tsfm_baseline_chronos2.ipynb`): GIFT-Eval short-term split,
# V = 3 validation windows before the test windows, θ(α) = (1-α)·θ_zero-shot + α·θ_fine-tuned for
# α ∈ {0, .1, .25, .5, .75, 1}, and the frozen selection rules `+wise`, `+wise1`, `+guard` (m = 3%), `+guardLF`.
# Only the model and its fine-tuning loop change. Pre-registration: `results/phase1_analysis.md` §12.
#
# **Kaggle settings:** Accelerator = GPU T4, Internet = On. Use *Save Version -> Save & Run All (Commit)*.

# %%
import os, sys, subprocess
SMOKE = os.environ.get("TSFM_SMOKE", "0") == "1"   # local CPU smoke test only (random weights)
IN_KAGGLE = os.path.exists("/kaggle")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if IN_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "timesfm[torch]>=2.5", "gluonts", "datasets", "huggingface_hub", "safetensors"], check=True)

# %%
import copy, gc, json, math, shutil, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import datasets
from huggingface_hub import snapshot_download
from gluonts.time_feature import get_seasonality, norm_freq_str
from pandas.tseries.frequencies import to_offset
import timesfm
from timesfm.torch import util as tfm_util

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__, "| timesfm", getattr(timesfm, "__version__", "?"))
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# %% [markdown]
# ## Configuration
# TimesFM has no official fine-tuning recipe, so we mirror the Chronos-2 LoRA setup: r = 8, alpha = 16, adapters
# on every attention projection and on the point-output layer, batch 32, 4000 steps, linear decay, AdamW,
# gradient clipping 1.0. Learning rate 1e-4 (the usual LoRA rate). Context 1024 for both fine-tuning and
# forecasting. Loss: pinball loss on the nine quantile outputs of the last input patch, in the model's own
# patch-normalised space (exactly the path used at inference).

# %%
WORK = Path("/kaggle/working") if IN_KAGGLE else Path(os.environ.get("TSFM_WORK", "./tsfm_work"))
TMP = Path("/kaggle/temp") if IN_KAGGLE else WORK / "tmp"
DATA_DIR = TMP / "gift_eval_data"
RES_DIR = WORK / "results_tfm"
for d in (DATA_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)

TERM = "short"
SEEDS = [0]
MODES = ["lora_full"]
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
NUM_STEPS, BATCH_SIZE, LR = 4000, 32, 1e-4
LORA_R, LORA_ALPHA = 8, 16
LORA_TARGETS = ("attn.qkv_proj", "attn.out", "output_projection_point.output_layer")
CONTEXT = 1024
VAL_WINDOWS = 3
GUARD_M = 0.03
MODEL_ID = "google/timesfm-2.5-200m-pytorch"

# Same 28 GIFT-Eval configurations as the Chronos-2 study (design 12 + hold-out 9 + fresh 7), in parts.
PARTS = {
    "A": ["us_births/D", "saugeenday/D", "ett1/W", "ett2/W", "solar/W", "hierarchical_sales/W"],
    "B": ["bizitobs_l2c/H", "ett1/D", "ett2/D", "M_DENSE/D", "LOOP_SEATTLE/D", "jena_weather/D"],
    "H1": ["hospital", "covid_deaths", "car_parts_with_missing", "m4_weekly", "bizitobs_service"],
    "H2": ["m4_hourly", "SZ_TAXI/H", "kdd_cup_2018_with_missing/D", "bizitobs_application"],
    "F1": ["electricity/H", "electricity/15T", "electricity/W", "bitbrains_rnd/H"],
    "F2": ["electricity/D", "m4_daily", "bitbrains_fast_storage/H"],
    # New-configuration confirmation of the §14 rules (pre-registered 2026-11-22); not part of the 28 above.
    "N": ["restaurant", "hierarchical_sales/D", "solar/D", "us_births/W",
          "ett1/H", "M_DENSE/H", "kdd_cup_2018_with_missing/H", "jena_weather/H"],
}
# Stability run (2026-11-22): extra seeds for LoRA full and a fine-tuning-length ablation (500 / 1000 steps,
# seed 0; 4000 steps already exist) on 3 low-frequency datasets and the high-frequency case where validation misled.
PARTS["S"] = ["us_births/D", "ett1/W", "solar/W", "bizitobs_service"]
STEPS = {"lora_s500": 500, "lora_s1000": 1000}   # LoRA-full variants with fewer steps; other modes use NUM_STEPS
SEEDS_BY_MODE = {}                                # optional per-mode seed lists; default SEEDS
STAB = False
if STAB:
    MODES = ["lora_s500", "lora_s1000", "lora_full"]
    SEEDS_BY_MODE = {"lora_full": [1, 2], "lora_s500": [0], "lora_s1000": [0]}
# Coverage re-run (2026-11-22): the 28 configurations again (LoRA full, seed 0), now recording interval coverage
# and conformal (CQR) recalibration for every alpha (RQ4). Also a reproducibility check of the seed-0 results.
PARTS["C1"] = PARTS["A"] + PARTS["B"]
PARTS["C2"] = PARTS["H1"] + PARTS["H2"] + PARTS["F1"] + PARTS["F2"]
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
PART = None          # None = all 28; or a key of PARTS; or a "+"-joined list, e.g. "A+B"
DATASETS = sum((v for k, v in PARTS.items() if k not in ("N", "S", "C1", "C2", "E1", "E2", "E3")), [])
if PART:
    DATASETS = sum((PARTS[p] for p in PART.split("+")), [])
    RES_DIR = WORK / f"results_tfm_{PART.replace('+', '')}"
    RES_DIR.mkdir(parents=True, exist_ok=True)

QUICK = False   # True = first check on Kaggle: 2 datasets, 500 steps
if QUICK:
    DATASETS, NUM_STEPS = ["ett1/W", "us_births/D"], 500
    RES_DIR = WORK / "results_tfm_quick"
    shutil.rmtree(RES_DIR, ignore_errors=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

if SMOKE:
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
P = 32   # TimesFM input patch length

def load_timesfm():
    if SMOKE:   # random weights: only checks the code path
        m = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        m.model.to(m.model.device)
    else:
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID, torch_compile=False)
    m.model.eval()
    return m

def forecast_config(recommended=False):
    if recommended:   # the authors' suggested inference flags (reference row only)
        return timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=128, normalize_inputs=True,
                                      per_core_batch_size=4 if SMOKE else 256, use_continuous_quantile_head=True,
                                      force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True)
    # Used for every alpha, including alpha=0 (zero-shot): exactly the output path that LoRA is trained on.
    return timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=128, normalize_inputs=True,
                                  per_core_batch_size=4 if SMOKE else 256, use_continuous_quantile_head=False,
                                  force_flip_invariance=False, infer_is_positive=False, fix_quantile_crossing=True)

def predict_q(tfm, ctxs, H):
    inputs = []
    for c in ctxs:
        c = np.asarray(c, dtype=np.float32)
        inputs.append(c if np.isfinite(c).any() else np.zeros(1, dtype=np.float32))
    _, q = tfm.forecast(horizon=H, inputs=inputs)
    return np.asarray(q[:len(ctxs), :H, 1:10], dtype=np.float64)          # (N, H, 9): quantiles 0.1..0.9

# ---------- LoRA (plain PyTorch, same shape as peft's default: B = 0 at start, no dropout) ----------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=LORA_R, alpha=LORA_ALPHA):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    def merged(self):
        with torch.no_grad():
            self.base.weight += (self.B @ self.A) * self.scale
        return self.base

def _swap(root, name, new):
    parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
    setattr(parent, name.rsplit(".", 1)[-1], new)

def add_lora(mod):
    names = [n for n, m in mod.named_modules() if isinstance(m, nn.Linear) and n.endswith(LORA_TARGETS)]
    for n in names:
        _swap(mod, n, LoRALinear(mod.get_submodule(n)))
    return names

def merge_lora(mod):
    for n in [n for n, m in mod.named_modules() if isinstance(m, LoRALinear)]:
        _swap(mod, n, mod.get_submodule(n).merged())

# ---------- training batches ----------
def fill_nans(s):
    s = pd.Series(s, dtype="float64").interpolate(limit_direction="both").to_numpy()
    return np.nan_to_num(s, nan=0.0).astype(np.float32)

def sample_batch(series, isnan, H, rng, B):
    x = np.zeros((B, CONTEXT), np.float32); m = np.ones((B, CONTEXT), bool)
    y = np.zeros((B, H), np.float32); ym = np.zeros((B, H), bool)
    for b in range(B):
        i = rng.integers(len(series)); s, nn_ = series[i], isnan[i]
        t = rng.integers(min(H, len(s) - H), len(s) - H + 1)           # context s[:t], target s[t:t+H]
        c = s[max(0, t - CONTEXT):t]
        x[b, CONTEXT - len(c):] = c; m[b, CONTEXT - len(c):] = False
        y[b] = s[t:t + H]; ym[b] = ~nn_[t:t + H]
    return x, m, y, ym

def tfm_loss(mod, x, m, y, ym):
    """Pinball loss of the last-patch forecast; mirrors TimesFM's compiled decode (global + per-patch RevIN)."""
    mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)          # normalize_inputs=True
    x = tfm_util.revin(x, mu, sd); y = tfm_util.revin(y, mu, sd)
    Bn, C = x.shape; N = C // P; H = y.shape[1]
    px, pm = x.view(Bn, N, P), m.view(Bn, N, P)
    n = torch.zeros(Bn, device=x.device); pmu = torch.zeros_like(n); psd = torch.zeros_like(n)
    mus, sds = [], []
    for i in range(N):
        (n, pmu, psd), _ = tfm_util.update_running_stats(n, pmu, psd, px[:, i], pm[:, i])
        mus.append(pmu); sds.append(psd)
    cmu, csd = torch.stack(mus, 1), torch.stack(sds, 1)
    nx = torch.where(pm, 0.0, tfm_util.revin(px, cmu, csd))
    (_, _, out, _), _ = mod(nx, pm)
    out = out.view(Bn, N, mod.o, mod.q)[:, -1, :H, 1:10]               # (B, H, 9), normalised space
    yn = torch.nan_to_num(tfm_util.revin(y, cmu[:, -1], csd[:, -1]), nan=0.0, posinf=1e4, neginf=-1e4)
    e = yn.unsqueeze(-1) - out
    q = torch.tensor(QUANTILES, device=x.device)
    pl = torch.maximum(q * e, (q - 1) * e)
    w = ym.unsqueeze(-1).float()
    return (pl * w).sum() / (w.sum() * len(QUANTILES)).clamp(min=1.0)

CKPT_DELTAS = {}   # §22: step -> {weight key: LoRA delta}, filled by finetune when ES

def finetune(base_module, train, H, seed, num_steps=None):
    num_steps = num_steps or NUM_STEPS
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    series = [fill_nans(s) for s in train if len(s) >= H + 2]
    isnan = [~np.isfinite(np.asarray(s, dtype=np.float64)) for s in train if len(s) >= H + 2]
    mod = copy.deepcopy(base_module)
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    names = add_lora(mod)
    params = [p_ for p_ in mod.parameters() if p_.requires_grad]
    print(f"  LoRA on {len(names)} layers, {sum(p_.numel() for p_ in params) / 1e6:.2f}M trainable params")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / num_steps))
    mod.train()
    run = 0.0
    for step in range(1, num_steps + 1):
        x, m, y, ym = (torch.from_numpy(a).to(DEVICE) for a in sample_batch(series, isnan, H, rng, BATCH_SIZE))
        loss = tfm_loss(mod, x, m, y, ym)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run += loss.item()
        if step % 500 == 0 or step == num_steps:
            print(f"    step {step}: loss {run / (500 if step % 500 == 0 else step % 500):.4f}")
            run = 0.0
        if ES and (step in CKPT_STEPS or step == num_steps):   # §22: LoRA update at this step, as weight deltas
            CKPT_DELTAS[step] = {n + ".weight": (m.B @ m.A).detach().mul(m.scale).float().cpu()
                                 for n, m in mod.named_modules() if isinstance(m, LoRALinear)}
    merge_lora(mod)
    mod.eval()
    return {k: v.detach().to("cpu", copy=True) for k, v in mod.state_dict().items()}

# %%
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
# Seasonal Naive, zero-shot (plus a reference row with the authors' recommended inference flags), then LoRA
# fine-tuning. For every α we score the V validation windows and the test windows; the frozen rules pick α
# from validation only.

# %%
tfm = load_timesfm()
tfm.compile(forecast_config())
SD0 = to_cpu_sd(tfm.model)
done = done_keys()

for name in DATASETS:
    D = load_config(name)
    print(f"\n=== {name}: series={D['n_series']} obs={D['n_obs']} H={D['H']} windows={D['windows']} season={D['season']} V={D['V']}")
    meta = dict(dataset=name, H=D["H"], windows=D["windows"], n_series=D["n_series"], n_obs=D["n_obs"])
    tfm.model.load_state_dict(SD0)

    if (name, "seasonal_naive", -1) not in done:
        s = score(seasonal_naive_q(D["test_ctx"], D["H"], D["season"]), D["test_lab"], D["test_ctx"], D["season"])
        append(MAIN_CSV, [dict(**meta, method="seasonal_naive", seed=-1, alpha=np.nan, fit_sec=0, **s)])

    if (name, "zero_shot", -1) not in done:
        s = score(predict_q(tfm, D["test_ctx"], D["H"]), D["test_lab"], D["test_ctx"], D["season"])
        tfm.compile(forecast_config(recommended=True))
        r = score(predict_q(tfm, D["test_ctx"], D["H"]), D["test_lab"], D["test_ctx"], D["season"])
        tfm.compile(forecast_config())
        append(MAIN_CSV, [dict(**meta, method="zero_shot", seed=-1, alpha=0.0, fit_sec=0, **s),
                          dict(**meta, method="zero_shot_rec", seed=-1, alpha=0.0, fit_sec=0, **r)])
        print("zero-shot", s, "| recommended flags", r)

    for mode in MODES:
        for seed in SEEDS_BY_MODE.get(mode, SEEDS):
            if (name, mode, seed) in done and (name, mode + "+wise", seed) in done:
                continue
            t0 = time.time()
            try:
                SD1 = finetune(tfm.model, D["train"], D["H"], seed, STEPS.get(mode))
            except Exception as e:
                print(f"  !! fine-tuning failed for {mode}: {e!r}")
                append(MAIN_CSV, [dict(**meta, method=mode + "_FAILED", seed=seed, alpha=1.0, fit_sec=np.nan,
                                       MWQL=np.nan, MASE=np.nan, MAPE=np.nan)])
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue
            fit_sec = time.time() - t0

            curve = []
            last = D["val_last"]
            vy = np.stack(D["val_lab"]).astype(np.float64)
            for a in ALPHAS:
                te = time.time()
                interpolate_into(tfm.model, SD0, SD1, a)
                vq = predict_q(tfm, D["val_ctx"], D["H"])
                v = score(vq, D["val_lab"], D["val_ctx"], D["season"])
                v1 = mwql(vq[last], vy[last])
                tq = predict_q(tfm, D["test_ctx"], D["H"])
                t = score(tq, D["test_lab"], D["test_ctx"], D["season"])
                curve.append(dict(**meta, method=mode, seed=seed, alpha=a, V=D["V"],
                                  val_MWQL=v["MWQL"], val_MASE=v["MASE"], val1_MWQL=v1,
                                  test_MWQL=t["MWQL"], test_MASE=t["MASE"], test_MAPE=t["MAPE"],
                                  **interval_stats(vq, D["val_lab"], "val"),
                                  **interval_stats(tq, D["test_lab"], "test"),
                                  **conformal_stats(vq, D["val_lab"], D["val_ctx"], tq, D["test_lab"], D["test_ctx"])))
                curve[-1].update(step=STEPS.get(mode, NUM_STEPS), eval_sec=time.time() - te)
            tfm.model.load_state_dict(SD0)
            append(ALPHA_CSV, curve)
            if ES:
                check_final_delta(SD0, SD1, CKPT_DELTAS[STEPS.get(mode, NUM_STEPS)])
                append(ALPHA_CSV, es_rows(tfm.model, tfm, SD0, CKPT_DELTAS, D, meta, mode, seed))
                CKPT_DELTAS.clear()
            c = pd.DataFrame(curve)
            full = c[c.alpha == 1.0].iloc[0]
            best = c.sort_values(["val_MWQL", "alpha"]).iloc[0]
            best1 = c.sort_values(["val1_MWQL", "alpha"]).iloc[0]
            c0 = c[c.alpha == 0.0].iloc[0]
            ok = c[(c.alpha == 0.0) | ((c.val_MWQL < (1 - GUARD_M) * c0.val_MWQL) & (c.val1_MWQL <= c0.val1_MWQL))]
            guard = ok.sort_values(["val_MWQL", "alpha"]).iloc[0]
            guard_lf = c0 if D["base_freq"] in ("H", "T", "S") else guard
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
            print(f"  {mode} seed={seed} fit={fit_sec:.0f}s  test MWQL a=0: {c0.test_MWQL:.4f} a=1: {full.test_MWQL:.4f} | "
                  f"wise a={best.alpha}: {best.test_MWQL:.4f} | wise1 a={best1.alpha}: {best1.test_MWQL:.4f}"
                  f" | guard a={guard.alpha}: {guard.test_MWQL:.4f} | guardLF a={guard_lf.alpha}: {guard_lf.test_MWQL:.4f}")
            del SD1
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

# %% [markdown]
# ## Summary: change vs zero-shot per dataset (%, negative = better) and selected alpha

# %%
df = pd.read_csv(MAIN_CSV)
failed = df[df.method.str.endswith("_FAILED")]
if len(failed):
    print("FAILED runs:\n", failed[["dataset", "method", "seed"]])
df = df[~df.method.str.endswith("_FAILED")].drop_duplicates(["dataset", "method", "seed"], keep="last")
zs = df[df.method == "zero_shot"].set_index("dataset").MWQL
rel = df[df.method != "seasonal_naive"].copy()
rel["chg"] = 100 * (rel.MWQL / rel.dataset.map(zs) - 1)
per_ds = rel.pivot_table(index="dataset", columns="method", values="chg")
print(per_ds.round(2))
per_ds.to_csv(RES_DIR / "per_dataset_change_vs_zero_shot.csv")
print("\nmean:\n", per_ds.mean().round(2))
print("\nSelected alpha:\n", df[df.method.str.contains(r"\+")].pivot_table(index="dataset", columns="method", values="alpha"))
