# Pre-registration and analysis log

This is the dated log that was kept during the study. Every pre-registration section was written and committed before the runs it describes, and every result section after them. It is reproduced unedited except for removing the names of the people who ran Kaggle sessions. File paths refer to this repository (`results/`, `analysis/`, `runs/`).

---

# Phase 1 Results: Chronos-2 Fine-Tuning vs Guarded Interpolation (`+wise`)

**Date:** 2026-11-19
**Runs:** Kaggle notebooks `tsfm-phase1-partA` (18,633 s) and `tsfm-phase1-partB` (16,793 s), both on a T4 GPU.
**Setup:**
- Model: Chronos-2 (120M parameters), seed 0
- Fine-tuning: 4,000 steps, batch size 32, fine-tuning context length 1,024
- Learning rates: 1e-4 for FFT, 1e-5 for LoRA; LoRA r = 8, α = 16
- Few-shot: last 20% of each training series
- Data: 12 GIFT-Eval configurations marked "small", short horizon, univariate
**Selection methods:**
- `+wise`: α chosen from {0, 0.1, 0.25, 0.5, 0.75, 1} using V = 3 held-out validation windows (V = 2 for solar/W)
- `+wise1`: the same, but using only the last validation window (ablation)
**Metric:** MWQL (a CRPS proxy) divided by the Seasonal Naive MWQL, then averaged over datasets (`MWQL_n`). Lower is better.

Raw summaries are in `partA/` and `partB/`. Per-dataset % changes for all 12 datasets are in `per_dataset_change_vs_zero_shot_all.csv`.

## 1. Mean over the 12 datasets

| Method | MWQL_n | vs zero-shot | MASE_n | MAPE_n | Paper (Table 4, small) |
|---|---|---|---|---|---|
| Zero-shot | **0.534** | — | 0.684 | 0.610 | 0.423 |
| FFT | 0.981 | +83.7% | 1.112 | 1.021 | 0.577 (+36%) |
| FFT + wise | 0.551 | +3.2% | 0.699 | 0.628 | — |
| FFT + wise1 | 0.535 | +0.1% | 0.691 | 0.607 | — |
| LoRA full | 0.737 | +37.9% | 0.965 | 0.823 | 0.477 (+13%) |
| LoRA full + wise | 0.545 | +2.0% | 0.694 | 0.618 | — |
| LoRA full + wise1 | 0.538 | +0.7% | 0.697 | 0.637 | — |
| LoRA few | 0.641 | +19.9% | 0.808 | 0.859 | 0.471 (+11%) |
| LoRA few + wise | 0.539 | +0.9% | 0.693 | 0.632 | — |
| LoRA few + wise1 | 0.541 | +1.2% | 0.690 | 0.684 | — |

**Reproduction of the base paper.**
- We reproduce the qualitative finding: on small datasets, every fine-tuning mode is worse than zero-shot.
- The ranking also matches the paper: FFT is worst, then LoRA full, then LoRA few.
- Our degradation is larger than the paper's (FFT +84% vs +36%). Likely reasons:
  - We use a different set of datasets, and our size split differs from the paper's.
  - Fine-tuning context is limited to 1,024 (needed for T4 memory and time).
  - There is no early stopping.
  - We have a single seed.
- The absolute MWQL_n values are not comparable to the paper's, because the dataset sets differ.

## 2. Per dataset: how often is the method no worse than zero-shot?

A change of at most +1% counts as a tie.

| Method | ≤ zero-shot (of 12) | Median change | Worst | Best |
|---|---|---|---|---|
| FFT | 1 | +57.9% | +339.5% | −0.7% |
| **FFT + wise** | **10** | 0.0% | +32.1% (saugeenday/D) | −8.5% (M_DENSE/D) |
| FFT + wise1 | 9 | 0.0% | +9.5% | −8.5% |
| LoRA full | 4 | +6.8% | +185.6% | −9.1% |
| **LoRA full + wise** | **9** | 0.0% | +22.7% (ett2/W) | −9.1% (M_DENSE/D) |
| LoRA full + wise1 | 9 | 0.0% | +11.5% | −8.9% |
| LoRA few | 3 | +4.0% | +164.2% | −2.8% |
| **LoRA few + wise** | **7** | +0.1% | +8.6% (ett1/W) | −2.8% |
| LoRA few + wise1 | 7 | +0.1% | +12.9% | −2.8% |

Across all 36 (dataset, mode) pairs, `+wise` is no worse than zero-shot in **26/36 (72%)**. `+wise1` also gets 25/36.

**Cases where `+wise` beats zero-shot:**
- M_DENSE/D: −8.5% (FFT), −9.1% (LoRA full)
- us_births/D: −7.2% (LoRA full)
- saugeenday/D: −4.4% (LoRA full)
- ett2/W: −1.1% (FFT)

**Failure cases:**
- saugeenday/D, FFT: V = 3 picked α = 1 and scored +32%, while `wise1` picked α = 0.
- ett2/W, LoRA full: +22.7%.
- ett1/W, LoRA few: +8.6%.
- solar/W, LoRA few: +3.7%. Here V = 2 because the series are short.
- bizitobs_l2c/H: +3–5% in all three modes.

## 3. Go/no-go check (Day-2 criteria of the study plan)

| Criterion | Result | Pass? |
|---|---|---|
| `+wise` ≤ zero-shot on ≥ 80% of datasets | FFT 83%, LoRA full 75%, LoRA few 58%; pooled 72% | **Partly**: FFT only |
| `+wise` better on average than the best fine-tuning mode | 0.539–0.551 vs 0.641 (LoRA few) | **Yes** |
| Selected α not always only 0 or 1 | 0.1, 0.25, 0.5 and 0.75 are all chosen | **Yes** |

**Verdict: conditional GO.**
- The core idea works as a guard. It removes almost all the damage from fine-tuning (FFT goes from +84% to +3%, LoRA full from +38% to +2%). It also keeps real gains where they exist (M_DENSE/D, us_births/D, saugeenday/D).
- But the claim "never worse than zero-shot" does not yet hold, and on average `+wise` is still 1–3% worse than zero-shot.
- Using 3 validation windows did **not** help on average over 1 window. So the selection rule itself, not only the amount of validation data, is the bottleneck.

## 4. Next steps (from the fallback options in the plan)

1. **Offline study of selection rules (no GPU needed).** Use the full `alpha_curves.csv` files (validation and test MWQL for every α) to test better rules, then pick one before any new GPU run:
   - **Margin rule:** move away from α = 0 only if validation MWQL improves by more than m% (try m = 1–5%), or only if a paired bootstrap over series and windows shows the gain is significant.
   - **Shrinkage:** take the α with the best validation score, then halve it.
   - **Choose α per series group,** where there are many series.
2. **Prediction ensembling:** average the quantile forecasts of α = 0 and the selected α (cheap, done at inference).
3. **Layer-wise α** (for example, the output head only) as planned for weeks 3–4.
4. **Seeds 1 and 2** for the final tables. Each seed costs about 10 GPU-hours; Kaggle quota used so far is about 11 h of 30 h this week.
5. **Framing for the paper:** the strongest current message is *safe adaptation*, meaning large average degradation turned into near-zero-shot performance, with gains in some regimes. The offline rule study should aim to remove the remaining 1–3% average gap.

## 4b. Offline study of selection rules (done 2026-11-19; no GPU)

**Data and code:**
- Full `results.csv` and `alpha_curves.csv` were downloaded for both parts and saved as `partA|partB/results.csv` and `alpha_curves_compact.csv` (6 significant digits).
- Script: `kaggle/analysis/selection_rules.py`. Outputs: `results/selection_rules/`.

**Rules tested.** Each rule picks α using validation information only.
- `argmin_V3`: the current `+wise`.
- `argmin_V1`: the current `+wise1`.
- `shrink_half`: halve the chosen α.
- `agree_min`: take the smaller of the V3 and V1 choices.
- `margin_V3(m)`: leave α = 0 only if pooled validation MWQL improves by more than m.
- `both_improve(m)`: α must improve pooled validation by more than m **and** must not worsen the most recent validation window.

**Choosing m.** The margin m was chosen by leave-one-dataset-out (LODO). LODO picked m = 3% for `both_improve` and m = 5% for `margin_V3`, in every fold.

**Results over all 36 (dataset, mode) pairs:**

| Rule | Mean MWQL_n | Mean Δ vs zero-shot | No worse (≤ +1%) | Better (< −1%) | Worst |
|---|---|---|---|---|---|
| Zero-shot | 0.534 | 0 | 36 | 0 | 0 |
| `argmin_V3` (+wise) | 0.545 | +1.33% | 26 | 8 | +32.1% |
| `argmin_V1` (+wise1) | 0.538 | +0.66% | 25 | 7 | +12.9% |
| `shrink_half` | 0.537 | +0.23% | 33 | 5 | +16.4% |
| `agree_min` | 0.533 | −0.47% | 30 | 7 | +4.3% |
| `margin_V3`, LODO (m = 5%) | 0.536 | −0.13% | 34 | 5 | +22.7% |
| **`both_improve`, LODO (m = 3%)** | **0.531** | **−0.81%** | **36** | 4 | **0.0%** |
| Oracle (best α on test; upper bound) | 0.526 | −1.72% | 36 | 15 | 0.0% |

**`both_improve` (m = 3%) by mode:**

| Mode | Mean Δ vs zero-shot | No worse | Oracle Δ |
|---|---|---|---|
| FFT | −0.72% | 12/12 | −1.70% |
| **LoRA full** | **−1.73%** | **12/12** | −2.13% |
| LoRA few | 0.00% (always picks α = 0) | 12/12 | −1.32% |

Pairs where `both_improve` moves away from zero-shot:

| Dataset | Mode | α | Δ vs zero-shot |
|---|---|---|---|
| M_DENSE/D | FFT | 0.1 | −8.5% |
| M_DENSE/D | LoRA full | 1.0 | −9.1% |
| us_births/D | LoRA full | 0.75 | −7.2% |
| saugeenday/D | LoRA full | 0.75 | −4.4% |
| hierarchical_sales/W | FFT | 0.25 | 0.0% |

**Interpretation.**
- With **LoRA full + `both_improve` (m = 3%)**, the procedure is never worse than zero-shot on any of the 12 datasets.
- On average it is 1.7% better than zero-shot, which recovers about 80% of the oracle gain.
- All three Day-2 criteria now pass for this configuration:
  - ≥ 80% of datasets no worse than zero-shot: 100%.
  - Better on average than the best fine-tuning mode.
  - α is not always only 0 or 1.

**Important caveat (post-hoc selection).** The *family* of rules was designed after seeing these curves. Only m was tuned honestly, through LODO. The rule must therefore be confirmed on data it has not seen:
- new seeds (1, 2),
- other GIFT-Eval configurations (medium size),
- the energy datasets planned for paper 2.

The rule is fixed now, in advance, as the primary method: `both_improve`, m = 3%, applied to LoRA full. Results on that new data are the confirmatory evidence for the paper.

## 5. Data caveats

- Values were read from the Kaggle output viewer:
  - `summary_normalised.csv` at 4 d.p.
  - per-dataset changes at 2 d.p.
- The full `results.csv` files were later downloaded (exact values). `alpha_curves_compact.csv` keeps 6 significant digits.
- Single seed, so no variance estimates yet.

## 6. Confirmatory run (seeds 1–2): pre-registered `+guard` rule

**Runs:**
- Kaggle notebooks `tsfm-confirm-partA` (12,067 s) and `tsfm-confirm-partB` (12,325 s), 2026-11-19/20.
- LoRA full only, seeds 1 and 2. All other settings are identical to Phase 1.

**The rule was fixed before these runs** (`GUARD_M = 0.03`, `both_improve`):
- Leave α = 0 only if pooled validation MWQL (V = 3) improves by more than 3%, **and** the most recent validation window does not get worse.
- Otherwise use the zero-shot model.

**Pre-registered success criteria** over 24 runs (12 datasets × 2 seeds):
- (1) ≥ 80% of runs no worse than zero-shot (≤ +1%);
- (2) mean change vs zero-shot ≤ 0%.

**Files:**
- Raw results: `confirmA/results.csv`, `confirmB/results.csv`
- Per-run table: `confirm_per_run.csv`
- Script: `kaggle/analysis/confirm_analysis.py`

### 6.1 Result: both criteria pass

Seeds 1–2, 24 runs per method:

| Method | No worse than zero-shot | Better (< −1%) | Mean Δ vs zero-shot | Median Δ | Worst | Best | Mean MWQL_n |
|---|---|---|---|---|---|---|---|
| LoRA full (α = 1) | 10/24 (42%) | 8 | +29.3% | +5.8% | +190.2% | −11.4% | 0.72 |
| + wise (argmin, V = 3) | 18/24 (75%) | 7 | +0.47% | 0.0% | +20.8% | −11.4% | 0.54 |
| + wise1 (argmin, V = 1) | 18/24 (75%) | 6 | +0.46% | 0.0% | +29.5% | −9.3% | 0.54 |
| **+ guard (pre-registered)** | **24/24 (100%)** | 6 | **−1.91%** | 0.0% | **0.0%** | −11.4% | **0.53** |

Zero-shot mean MWQL_n is 0.534.

- **Criterion (1): 100% ≥ 80%, PASS.**
- **Criterion (2): −1.91% ≤ 0%, PASS.**

For comparison, seed 0 (exploratory, where the rule was found) gave 12/12 and −1.73%. Over all 36 runs (seeds 0–2), `+guard` is no worse than zero-shot in 36/36, with a mean of −1.85%.

### 6.2 Stability across seeds

The α chosen by `+guard` is **identical across seeds 0, 1 and 2 on all 12 datasets**:

| Dataset | Chosen α | Δ vs zero-shot (seeds 0 / 1 / 2) |
|---|---|---|
| M_DENSE/D | 1.0 | −9.1% / −11.4% / −10.6% |
| us_births/D | 0.75 | −7.2% / −7.2% / −7.5% |
| saugeenday/D | 0.75 | −4.4% / −4.7% / −4.5% |
| The other 9 datasets | 0 | exactly zero-shot |

### 6.3 Honest limitations

1. **Out-of-sample in seeds, not in datasets.** The rule family and m were chosen on seed 0 of the *same* 12 datasets. Seeds 1–2 test robustness to fine-tuning randomness, but not generalisation to new datasets. A dataset-level hold-out (GIFT-Eval medium configurations, or the energy datasets for paper 2) is still needed.
2. **The rule is conservative.**
   - It falls back to zero-shot on 9 of 12 datasets, and gains come from 3 datasets only.
   - Mean gain is about 1.9%, versus about 2.1% for the oracle on seed 0.
   - A sign test over datasets (3 better, 0 worse, 9 ties) gives one-sided p = 0.125. That is not significant with so few non-tie datasets.
   - The paper's main claim should therefore be **safety**: never worse than zero-shot, with large protection against fine-tuning damage (+29% → −1.9%). Accuracy gains should be a secondary claim.
3. **Fine-tuning setup.** Context is 1,024, there is no early stopping, and the paper's AutoGluon pipeline was not used. Stronger fine-tuning might change how often α > 0 is chosen.

### 6.4 Updated go/no-go: **GO**

All three Day-2 criteria are met by the pre-registered `+guard` rule on held-out seeds:
- ≥ 80% of runs no worse than zero-shot: 100%;
- better than the best fine-tuning mode on average;
- α is not trivially 0 or 1 everywhere.

**Next steps:**
1. Dataset-level confirmation on GIFT-Eval medium configurations (LoRA full, seed 0, the same frozen rule).
2. A second model (TTM-R3 or TiRex) to show the rule is model-agnostic.
3. Start writing the paper (Method and Results sections can be drafted now).

## 7. Dataset-level hold-out: pre-registration (written 2026-11-20, before the runs)

**Goal:** test the frozen `+guard` rule (`both_improve`, m = 3%) on GIFT-Eval configurations whose source data were **never** used to design or tune it.

**Datasets** (9 hold-out configurations, short horizon):
- **H1:** hospital, covid_deaths, car_parts_with_missing, m4_weekly, bizitobs_service
- **H2:** m4_hourly, SZ_TAXI/H, kdd_cup_2018_with_missing/D, bizitobs_application

**Setup:** Chronos-2 LoRA full, seed 0, 4,000 steps, context 1,024. Validation windows V = 3 (V = 1 for car_parts, because the series are short). Everything else is identical to the confirm run.

**Success criteria** (fixed now):
- (1) `+guard` no worse than zero-shot (≤ +1% MWQL) on ≥ 8 of 9 datasets;
- (2) mean change vs zero-shot ≤ 0%.

**Secondary analysis (not a criterion):** compare with LoRA full (α = 1), `+wise` and `+wise1`.

## 8. Hold-out results (9 new GIFT-Eval configurations, seed 0)

**Runs and files:**
- Kaggle notebooks `tsfm-holdout-H1` (5,886 s) and `tsfm-holdout-H2` (5,190 s), 2026-11-20.
- Raw results: `holdoutH1/results.csv`, `holdoutH2/results.csv`
- Per-dataset table: `holdout_per_dataset.csv`

### 8.1 Per-dataset change in MWQL vs zero-shot (%)

| Dataset | LoRA full (α = 1) | + wise | + wise1 | **+ guard** | guard α |
|---|---|---|---|---|---|
| covid_deaths | +35.5 | −25.4 | −17.0 | **−17.0** | 0.25 |
| car_parts_with_missing | −5.9 | −5.9 | −5.9 | **−5.9** | 1.0 |
| m4_weekly | −2.1 | −2.1 | −2.1 | **−2.1** | 1.0 |
| hospital | −7.6 | −7.6 | −7.6 | 0.0 (missed the gain) | 0 |
| SZ_TAXI/H | +0.8 | −0.7 | −0.7 | 0.0 | 0 |
| kdd_cup_2018_with_missing/D | +5.9 | +1.2 | 0.0 | 0.0 | 0 |
| bizitobs_application | +53.3 | 0.0 | +23.6 | 0.0 | 0 |
| **m4_hourly** | +11.0 | +11.0 | +11.0 | **+11.0** ✗ | 1.0 |
| **bizitobs_service** | +17.5 | +17.5 | +9.9 | **+17.5** ✗ | 1.0 |

### 8.2 Summary over the 9 datasets

| Method | No worse (≤ +1%) | Better (< −1%) | Mean Δ | Median Δ | Worst | Mean MWQL_n |
|---|---|---|---|---|---|---|
| Zero-shot | 9/9 | 0 | 0 | 0 | 0 | 0.507 |
| LoRA full | 4/9 | 3 | +12.0% | +5.9% | +53.3% | 0.54 |
| + wise (V = 3) | 6/9 | 4 | −1.33% | −0.7% | +17.5% | 0.50 |
| + wise1 | 6/9 | 4 | +1.24% | −0.7% | +23.6% | 0.51 |
| **+ guard (pre-registered)** | **7/9** | 3 | **+0.39%** | 0.0% | +17.5% | 0.51 |

### 8.3 Verdict against the pre-registered criteria: **FAIL**

- **Criterion (1):** `+guard` was no worse than zero-shot on ≥ 8 of 9 datasets. Result: **7/9 → FAIL.**
- **Criterion (2):** mean change ≤ 0%. Result: **+0.39% → FAIL** (narrowly).

### 8.4 What the failures show

- On **m4_hourly** and **bizitobs_service**, *every* validation-based rule chose α = 1. Validation said fine-tuning helped, but on the test windows it hurt (+11% and +17.5%). This is a failure of validation as a proxy for test performance, not of the specific margin.
  - m4_hourly has a single test window per series, at the end of each M4 series.
  - bizitobs_service consists of very long, high-frequency series. Its V = 3 validation windows cover only 3 × 60 points just before 15 × 60 test points.
- The guard still **cuts the risk of fine-tuning sharply**: the worst case falls from +53% to +17.5%, the mean from +12% to +0.4%, and it avoids 3 of the 5 harmful fine-tunes. But it does **not** guarantee "never worse than zero-shot" on new datasets.
- `+wise` (plain argmin) had the best mean on the hold-out (−1.3%), the opposite of what we saw on the design datasets. With 9 datasets these differences are small and noisy.

### 8.5 Consequences for the paper

1. The claim "never worse than zero-shot" **must be dropped**. The supportable claim is: *validation-guarded interpolation removes most of the damage from fine-tuning and keeps most of its gains.* Across 21 configurations:
   - LoRA full: mean +23%, worst +190%.
   - +guard: mean −0.9%, worst +17.5%.
2. Any new rule designed after seeing these hold-out results needs **another fresh hold-out**, for example the remaining GIFT-Eval configurations not yet used (electricity/H, solar/H, LOOP_SEATTLE/H, m4_daily, m4_monthly, temperature_rain), before it can be claimed.
3. The honest negative result is itself useful to report: the pre-registration, the confirmation on new seeds, and the partial failure on new datasets. It shows that validation windows can mislead on series with regime changes near the end.

## 9. Why the hold-out failed: diagnosis (exploratory, 2026-11-20)

**Data:** every α-curve we have, 45 runs in total:
- design set, seed 0: 12 datasets × 3 modes;
- hold-out: 9 datasets × LoRA full.

**Script and outputs:** `kaggle/analysis/failure_diagnosis.py`, with outputs in `results/diagnosis/`.

### 9.1 Validation is a reliable signal for daily-or-coarser data, but not for high-frequency data

We compare the gain from fine-tuning (α = 1 vs α = 0) on the validation windows with the gain on the test windows.

| Data frequency | Runs | Pearson r (val vs test gain) | Sign agreement | "Misled" runs* |
|---|---|---|---|---|
| Daily or coarser | 38 | +0.86 | 84% | 1 of 6 |
| **Hourly or finer** | 7 | +0.68 | **43%** | **3 of 3** |

\* A run is "misled" when validation says fine-tuning helps by more than 3% but it hurts on test.

**All three high-frequency runs where validation reported a clear gain were misled:**

| Dataset | Validation gain | Last-window gain | Test result |
|---|---|---|---|
| bizitobs_service (10 s) | +13.2% | +12.0% | **−17.5%** |
| m4_hourly | +10.8% | +16.9% | **−11.0%** |
| bizitobs_l2c/H | +4.4% | −51.2% | −10.2% |

- In bizitobs_l2c/H, the `+guard` last-window check caught the problem (−51.2% on the last window), so `+guard` stayed at zero-shot.
- In bizitobs_service and m4_hourly, the last window also looked good, so nothing stopped the mistake. These are exactly the two hold-out failures.

The only misled low-frequency run, saugeenday/D with FFT, was also caught by the last-window check (−38%).

### 9.2 Mechanism (hypothesis)

The validation block covers V × H steps immediately after the fine-tuning data ends.
- **High-frequency data:** this is a tiny slice of calendar time, about 30 minutes for bizitobs_service and 6 days for m4_hourly. It shares the short-term regime of the final training points. Fine-tuning exploits that recent local regime (a recency effect), which inflates the validation gain, but the advantage does not carry over to the test windows.
- **Daily data:** the validation block covers about 90 days and represents the test period much better.

Other candidate explanations did **not** separate misled runs from good ones:
- the zero-shot test/validation error ratio;
- the test span relative to the validation span;
- the size of the validation gain.

### 9.3 Exploratory rule and the need for fresh confirmation

A simple rule follows from 9.1: apply `+guard` only to daily-or-coarser data, and use zero-shot for hourly-or-finer data. Call it `guard-LF`.

| Set | +guard | guard-LF |
|---|---|---|
| Design (12) | 12/12 no worse, −1.73% | 12/12, −1.73% (identical; the design set has no high-frequency α > 0 choice) |
| Hold-out (9) | 7/9, +0.39%, worst +17.5% | **9/9, −2.78%, worst 0.0%** |

**This is post hoc.** It is based on only 7 high-frequency runs, 3 of them informative. It must be confirmed on data from **sources never used so far**, before `guard-LF` can appear in the paper as a method. Candidates:
- High-frequency: electricity/H, electricity/15T, bitbrains_fast_storage/H, bitbrains_rnd/H.
- Low-frequency: electricity/D, electricity/W, m4_daily, m4_monthly, temperature_rain/D.

A more principled alternative to test alongside it is **time-matched validation** for high-frequency data: spread the validation windows over a longer stretch (for example at least one week, or at least the test span) instead of 3 adjacent windows.

### 9.4 Value for the paper even without a new rule

Section 9.1 is a substantive finding in its own right: *validation-based adaptation decisions are reliable for daily-or-coarser series (r = 0.86, 84% sign agreement) but unreliable for high-frequency series, where short validation blocks overstate fine-tuning gains.* It explains when fine-tuning can be trusted, which practitioners need to know.

## 10. Fresh-source confirmation of `guard-LF`: pre-registration (written 2026-11-20, before the runs)

**Rule under test, frozen now:** `guard-LF`.
- Hourly-or-finer data (base frequency H, T/min or S): use zero-shot (α = 0).
- Daily-or-coarser data: use `+guard` (`both_improve`, m = 3%, V = 3).

**Datasets.** All come from sources never used in any earlier run (design, confirm or hold-out). Short horizon, GIFT-Eval protocol.

| Group | Configurations |
|---|---|
| High frequency (4) | electricity/H, electricity/15T, bitbrains_fast_storage/H, bitbrains_rnd/H |
| Low frequency (3) | electricity/D, electricity/W, m4_daily |

We excluded m4_monthly (48,000 series) and temperature_rain (32,000 series) only because of compute cost, and decided this before seeing any results.

**Setup:** Chronos-2 LoRA full, seed 0, 4,000 steps, context 1,024. Everything else is identical to sections 6–8.

**Primary criteria:**
- (1) `guard-LF` is no worse than zero-shot (≤ +1% MWQL) on ≥ 6 of 7 datasets;
- (2) mean change vs zero-shot ≤ 0%.

**Secondary test of the mechanism from §9:**
- (3) On the 4 high-frequency datasets, the share of runs where validation says fine-tuning helps by more than 3% but it hurts on test (call these "misled" runs) is higher than on the 3 low-frequency datasets.
- (4) The original `+guard` performs worse than `guard-LF` on the high-frequency datasets.

Because criteria (1)–(2) are trivially met on high-frequency data (zero-shot is used there), the informative part is the low-frequency datasets, together with (3) and (4).

## 11. Fresh-source results (written 2026-11-20, after the runs)

Both parts ran without errors and with no deviations from §10: F1 took 11,786 s and F2 took 8,871 s on a Kaggle T4. The raw results are in `results/freshF1/` and `results/freshF2/`, and the analysis script is `kaggle/analysis/fresh_analysis.py`.

### 11.1 Change in MWQL vs zero-shot (%; negative = better)

| Dataset | Freq. | LoRA full | +wise (α) | +wise1 | +guard | **guard-LF** |
|---|---|---|---|---|---|---|
| electricity/H | high | −1.33 | −1.33 (1.0) | −1.33 | 0.00 | **0.00** |
| electricity/15T | high | −0.50 | −0.50 (1.0) | −0.50 | 0.00 | **0.00** |
| bitbrains_fast_storage/H | high | **+28.75** | 0.00 (0.0) | 0.00 | 0.00 | **0.00** |
| bitbrains_rnd/H | high | −28.99 | −28.96 (0.5) | 0.00 | 0.00 | **0.00** |
| electricity/D | low | −1.65 | −1.65 (1.0) | −1.65 | 0.00 | **0.00** |
| electricity/W | low | −9.17 | −6.79 (0.25) | −9.17 | 0.00 | **0.00** |
| m4_daily | low | −0.35 | −0.35 (1.0) | −0.35 | 0.00 | **0.00** |
| **Mean** | | −1.89 | **−5.65** | −1.86 | 0.00 | **0.00** |
| No worse (≤ +1%) | | 6/7 | **7/7** | 7/7 | 7/7 | 7/7 |

On all 7 datasets, `+guard` and `guard-LF` chose α = 0, so their forecasts are exactly the zero-shot model's; the differences are below 10⁻⁵ % and come from floating point.

### 11.2 Pre-registered criteria

| Criterion | Result | Verdict |
|---|---|---|
| (1) guard-LF no worse on ≥ 6/7 | 7/7 | **PASS** |
| (2) guard-LF mean ≤ 0% | 0.00% (identical to zero-shot) | **PASS as a tie**: met, but only because the rule did nothing |
| (3) More "misled" runs on high- than on low-frequency data | 0/4 vs 0/3. Validation never claimed a gain above 3% that then failed on test | **NOT SUPPORTED** |
| (4) +guard worse than guard-LF on high-frequency data | Identical (both α = 0 everywhere) | **NOT SUPPORTED** |

**Honest reading.** The primary criteria pass, but they pass trivially. As §10 warned, (1) and (2) say little on high-frequency data, and here the guard also stayed at zero-shot on all three low-frequency datasets. Its validation gains there (1.0–1.2%) were below the 3% margin, and on electricity/W the pooled validation curve pointed the wrong way. The mechanism proposed in §9 was not reproduced: on these new sources validation was never over-optimistic. In two cases it was too *pessimistic*: on electricity/W the α = 1 validation change was −6% while the test gain was +9%, and on bitbrains_rnd/H it was −3% while the test gain was +29%. So we cannot claim to have confirmed that high-frequency validation misleads. The §9 explanation stays a hypothesis supported by 3 hold-out cases only.

The safety claim did survive: fine-tuning would have been +28.8% worse on bitbrains_fast_storage/H, and every interpolation rule avoided that loss. The other side is opportunity cost. The guard rules left −29% (bitbrains_rnd/H) and −9% (electricity/W) of available gain unused, while plain `+wise` captured most of it.

### 11.3 All held-out data together (hold-out §8 + fresh §11: 16 datasets, seed 0, LoRA full)

| Rule | Mean change | No worse | Worst case |
|---|---|---|---|
| LoRA full (α = 1) | +5.95% | 10/16 | +53.3% |
| +wise | **−3.22%** | 13/16 | +17.5% |
| +wise1 | −0.11% | 13/16 | +23.6% |
| +guard (pre-registered in §6) | +0.22% | 14/16 | +17.5% |
| guard-LF (*post hoc*, frozen in §10) | −1.56% | **16/16** | **0.0%** |

### 11.4 Consequences for the paper

1. **Main claim (supported):** interpolating between zero-shot and fine-tuned weights, with α chosen on validation data, turns fine-tuning from a high-variance step into a bounded one.
   - Naive fine-tuning: +6% on average, worst case +53%.
   - Any interpolation rule: worst case +17.5% or better.
   - guard-LF: 0% worst case on 16 datasets.
2. **Frame the selection rules as a risk–return frontier**, not as one best rule. `+wise` gives the largest average gain (−3.2%) but has tail risk. `guard-LF` has no observed tail risk (16/16) but a smaller gain (−1.6%) and often falls back to zero-shot. Both are reported, with their provenance: `+wise` was fixed in advance, while guard-LF is post hoc and was only tested on 7 fresh datasets, where it equalled zero-shot.
3. **Drop the high-frequency claim.** Do not claim that "validation misleads on high-frequency data"; the fresh data did not reproduce it. Report it as a limitation: short validation windows are an unreliable signal *in both directions*.
4. **Next steps that would strengthen the paper:**
   - a second foundation model (for example TimesFM or Moirai) with the same protocol;
   - a less conservative guard (for example m = 1%, or pooling more validation windows) evaluated on all 28 datasets with leave-one-dataset-out tuning;
   - seeds 1–2 on the fresh sources.

## 12. Second model (TimesFM 2.5, 200M): pre-registration (written 2026-11-20, before any TimesFM run)

**Purpose.** Test whether the findings from Chronos-2 carry over to a model from a different family: TimesFM 2.5 is a Google decoder-only model, while Chronos-2 is an Amazon encoder model. The finding to test is that weight interpolation with α chosen on validation data bounds the risk of fine-tuning.

**Notebook:** `kaggle/tsfm_timesfm.ipynb`, source `tsfm_timesfm.py`.
- Data, splits, metrics and the α grid are identical to §6–§11.
- The selection rules are frozen exactly as before: `+wise`, `+wise1`, `+guard` (m = 3%, V = 3) and `guard-LF`.

**Fine-tuning.** TimesFM has no official recipe, so we fixed the following before any run:
- own LoRA implementation: r = 8, α_LoRA = 16, applied to all 20 `qkv_proj` and `out` attention projections and to the point-output layer (41 layers, 1.25M parameters);
- AdamW with learning rate 1e-4, linear decay, gradient clipping 1.0;
- batch 32, 4,000 steps, context 1,024, seed 0;
- loss: pinball loss over the 9 quantiles of the last-patch forecast, computed in the model's own RevIN-normalised space.

Inference uses fixed flags for every α: no continuous quantile head, no flip invariance, quantile-crossing fix on. The run also reports a reference row with the authors' recommended flags (`zero_shot_rec`), which is not used by any rule.

**Datasets:** all 28 configurations (design 12, hold-out 9, fresh 7). For TimesFM all of them are new data for the frozen rules.

**Single allowed adjustment.** A quick test runs first (ett1/W and us_births/D, 500 steps). Only if it shows that LoRA has essentially no effect (|change| < 0.5% at α = 1 on both datasets) or that the loss diverges may the learning rate be changed, once. Any such change will be recorded here as a deviation.

**Criteria** (`lora_full`, seed 0, 28 datasets, MWQL change vs zero-shot):
- (T1) Bounded risk: the worst case of each interpolation rule (`+wise`, `+guard`, `guard-LF`) is no larger than the worst case of plain LoRA (α = 1).
- (T2) `guard-LF` is no worse than zero-shot (≤ +1%) on ≥ 25/28 datasets, and its mean change is ≤ 0%.
- (T3) `+wise` has a lower mean change than plain LoRA, and its mean is ≤ 0%.

We will report all methods per group (design, hold-out and fresh), whether each criterion passes or fails.

**Caveat known in advance:** TimesFM 2.5's pre-training corpus may overlap with some GIFT-Eval sources. This would affect the zero-shot level, but not the comparison between selection rules on the same model.

### 12.1 Quick-test result (TimesFM, 500 steps, 2026-11-20)

The quick test ran without errors in 508 s on a T4. Each fit took about 0.36 s per step, which puts the full run at roughly 25 minutes per dataset for fitting. The raw files are in `results/tfm_quick/`.

| Dataset | Zero-shot MWQL | α = 1 (LoRA) | +wise (α) | +guard / guard-LF | Zero-shot, recommended flags |
|---|---|---|---|---|---|
| ett1/W | 0.2409 | **+17.1%** | +1.6% (0.25) | 0.0% (α = 0) | −0.1% |
| us_births/D | 0.01836 | −4.3% | −4.3% (1.0) | −4.3% (α = 1) | −2.6% |

**Adjustment rule from §12.** LoRA clearly changes the model (|change| at α = 1 is 17% and 4%, both above the 0.5% threshold), and the training loss is finite and of the expected size (0.20 and 0.08 in normalised pinball units). So **no adjustment is made**: the learning rate stays at 1e-4, and the full run uses exactly the settings registered in §12.

These two datasets already show both sides of the trade-off on TimesFM. On ett1/W fine-tuning hurts badly and every rule limits the damage. On us_births/D the guard correctly accepts a real gain.

### 12.2 Part A partial result (TimesFM, 6 of 28 datasets, 2026-11-20)

Part A ran as Kaggle notebook `tsfm-timesfm-A` with the full registered settings: 4,000 steps, lr 1e-4, seed 0. It took about 2.5 h on a T4, about 24–26 minutes of fitting per dataset. Files are in `results/tfm_A/`: `results.csv`, `alpha_curves_compact.csv` and `change_vs_zero_shot.csv`. The analysis script is `analysis/tfm_partA_analysis.py`.

Change in test MWQL vs zero-shot (%; negative = better). Selected α is in brackets.

| Dataset | LoRA α=1 | +wise | +wise1 | +guard | guard-LF | zero_shot_rec |
|---|---|---|---|---|---|---|
| us_births/D | +6.98 | −2.67 (0.1) | −2.67 (0.1) | 0.00 (0) | 0.00 (0) | −2.60 |
| saugeenday/D | +4.56 | −11.17 (0.25) | −12.50 (0.5) | −11.17 (0.25) | −11.17 (0.25) | +2.19 |
| ett1/W | +20.39 | +0.04 (0.1) | 0.00 (0) | 0.00 (0) | 0.00 (0) | −0.14 |
| ett2/W | +15.13 | 0.00 (0) | −1.17 (0.5) | 0.00 (0) | 0.00 (0) | +3.16 |
| solar/W | +29.41 | 0.00 (0) | 0.00 (0) | 0.00 (0) | 0.00 (0) | +1.06 |
| hierarchical_sales/W | +14.64 | +0.27 (0.25) | +2.67 (0.5) | 0.00 (0) | 0.00 (0) | −0.16 |
| **mean** | **+15.18** | **−2.25** | −2.28 | −1.86 | −1.86 | +0.58 |
| **worst** | +29.41 | +0.27 | +2.67 | 0.00 | 0.00 | +3.16 |
| no worse (≤ +1%) | 0/6 | 6/6 | 5/6 | 6/6 | 6/6 | 3/6 |

**Interim status of the §12 criteria (6/28 datasets; not a verdict).**
- T1, bounded risk: on track. The worst case of each rule is at most +0.27% (`+wise`) and 0.00% (`+guard`, `guard-LF`), against +29.4% for plain LoRA.
- T2, guard-LF: on track, 6/6 no worse and mean −1.86%. Up to 3 failures are still allowed over 28 datasets.
- T3, +wise: on track, mean −2.25% vs +15.18% for plain LoRA.

**Observations**
- With 4,000 steps, plain LoRA on TimesFM is worse than zero-shot on **every** Part A dataset. This holds even for us_births/D, which improved by 4.3% after 500 steps in the quick test, so longer training overfits.
- In this setting the interpolation rules matter much more than they did for Chronos-2: they turn a mean degradation of +15% into a gain of about 2%.
- Validation correctly flags the harm at α = 1 on all 6 datasets: the val gain is negative wherever the test gain is negative. So none of the rules was misled.
- The remaining gap to the oracle is small. The best test α gives −5.4% on us_births/D and −2.0% on ett2/W, where `+guard` stays at 0. That is the price of being conservative.
- On the full 4,000-step run, the recommended inference flags (`zero_shot_rec`) are not uniformly better, with a mean of +0.58%. This supports keeping the fixed flags for every α.

The remaining 22 datasets (Parts B, H1, H2, F1, F2) run after the weekly Kaggle GPU quota reset.

## 13. TimesFM 2.5 on all 28 datasets: final result for the §12 pre-registration (2026-11-22)

**Runs.** All with the settings registered in §12 (LoRA r = 8, lr 1e-4, 4,000 steps, context 1,024, seed 0), no deviations.

| Notebook | Datasets | Runtime |
|---|---|---|
| `tsfm-timesfm-A` | Part A (6 design) | ~2.5 h |
| `tsfm-timesfm-BH` | Part B (6 design) + H1 + H2 (9 hold-out) | 24,331 s (6.8 h) |
| `tsfm-timesfm-F` | F1 + F2 (7 fresh) | 13,131 s (3.6 h) |

On 2026-11-21 Kaggle's T4 queue was stuck for about 2 h (a Kaggle-wide capacity problem; an empty test notebook also stayed queued). F was cancelled and resubmitted once; the code was not changed.

**Files.**
- Results: `results/tfm_A/`, `results/tfm_BH/`, `results/tfm_F/`; combined table `results/tfm_all/change_vs_zero_shot.csv` and `summary_by_group.csv`.
- Scripts: `analysis/tfm_save_part.py` (saves a scraped part), `analysis/tfm_all_analysis.py` (this section).

**Data caveats.** Kaggle's output viewer returns at most 50 rows per file.
- **BH:** the full `results.csv` (120 rows) and `alpha_curves.csv` (90 rows) were read from the Kaggle output files and saved in `results/tfm_BH/`. They match the notebook's own `per_dataset_change_vs_zero_shot.csv` to within 4·10⁻⁷ percentage points. The α values in the run log (`selected_alpha.csv`) agree as well.
- **F:** all rows were recovered. Seven rows of `bitbrains_fast_storage/H` were rebuilt from its α-curve and the logged α values (`results/tfm_F/fill_bitbrains_fast_storage.json`). Each rebuilt row matches Kaggle's own change table exactly.

### 13.1 Change in test MWQL vs zero-shot (%; negative = better)

| Dataset | Group | LoRA α=1 | +wise | +wise1 | +guard | guard-LF |
|---|---|---|---|---|---|---|
| us_births/D | design | +6.98 | −2.67 | −2.67 | 0.00 | 0.00 |
| saugeenday/D | design | +4.56 | −11.17 | −12.50 | −11.17 | −11.17 |
| ett1/W | design | +20.39 | +0.04 | 0.00 | 0.00 | 0.00 |
| ett2/W | design | +15.13 | 0.00 | −1.17 | 0.00 | 0.00 |
| solar/W | design | +29.41 | 0.00 | 0.00 | 0.00 | 0.00 |
| hierarchical_sales/W | design | +14.64 | +0.27 | +2.67 | 0.00 | 0.00 |
| bizitobs_l2c/H | design | +24.18 | 0.00 | +4.44 | 0.00 | 0.00 |
| ett1/D | design | +34.06 | +2.86 | +6.33 | 0.00 | 0.00 |
| ett2/D | design | **+124.39** | +0.69 | 0.00 | 0.00 | 0.00 |
| M_DENSE/D | design | +1.29 | −0.99 | −0.99 | −0.99 | −0.99 |
| LOOP_SEATTLE/D | design | +10.99 | +0.32 | 0.00 | 0.00 | 0.00 |
| jena_weather/D | design | +12.69 | −0.56 | −0.56 | 0.00 | 0.00 |
| hospital | hold-out | +4.19 | +0.16 | +0.16 | 0.00 | 0.00 |
| covid_deaths | hold-out | +23.94 | −14.85 | −4.54 | −10.55 | −10.55 |
| car_parts_with_missing | hold-out | −3.48 | −3.48 | −3.48 | −3.48 | −3.48 |
| m4_weekly | hold-out | +1.10 | −1.42 | −1.42 | −1.42 | −1.42 |
| **bizitobs_service** | hold-out | +29.79 | **+29.79** | +8.24 | **+29.79** | 0.00 |
| m4_hourly | hold-out | −1.42 | +0.03 | 0.00 | 0.00 | 0.00 |
| SZ_TAXI/H | hold-out | +1.61 | +0.90 | +0.30 | +0.90 | 0.00 |
| kdd_cup_2018_with_missing/D | hold-out | +21.28 | 0.00 | +0.39 | 0.00 | 0.00 |
| bizitobs_application | hold-out | +58.42 | +4.67 | +33.79 | −1.73 | 0.00 |
| electricity/H | fresh | −7.64 | −6.68 | −6.68 | −6.68 | 0.00 |
| electricity/15T | fresh | −16.22 | −16.22 | −16.22 | −16.22 | 0.00 |
| electricity/W | fresh | −5.43 | 0.00 | −5.43 | 0.00 | 0.00 |
| bitbrains_rnd/H | fresh | −4.84 | 0.00 | 0.00 | 0.00 | 0.00 |
| electricity/D | fresh | −1.27 | −4.05 | −1.45 | 0.00 | 0.00 |
| m4_daily | fresh | −2.83 | −3.04 | −3.04 | 0.00 | 0.00 |
| bitbrains_fast_storage/H | fresh | +20.97 | 0.00 | 0.00 | 0.00 | 0.00 |

### 13.2 Summary

| Set | Rule | Mean | Worst | No worse (≤ +1%) | Better (< −1%) |
|---|---|---|---|---|---|
| **All 28** | LoRA α = 1 | +14.89% | +124.4% | 8/28 | 8 |
| | +wise | −0.91% | +29.8% | 25/28 | 9 |
| | +wise1 | −0.14% | +33.8% | 23/28 | 11 |
| | +guard | −0.77% | +29.8% | 27/28 | 7 |
| | **guard-LF** | **−0.99%** | **0.0%** | **28/28** | 4 |
| Design (12) | LoRA / +wise / +guard / guard-LF | +24.9 / −0.93 / −1.01 / −1.01 | +124.4 / +2.9 / 0 / 0 | 0 / 11 / 12 / 12 | |
| Hold-out (9) | LoRA / +wise / +guard / guard-LF | +15.1 / +1.76 / +1.50 / −1.72 | +58.4 / +29.8 / +29.8 / 0 | 2 / 7 / 8 / 9 | |
| Fresh (7) | LoRA / +wise / +guard / guard-LF | −2.46 / −4.28 / −3.27 / 0.00 | +21.0 / 0 / 0 / 0 | 6 / 7 / 7 / 7 | |
| Held-out (16) | LoRA / +wise / +guard / guard-LF | +7.39 / −0.89 / −0.59 / −0.97 | +58.4 / +29.8 / +29.8 / 0 | 8 / 14 / 15 / 16 | |

The reference row with the authors' recommended inference flags (`zero_shot_rec`) has a mean of −0.45% over the 28 datasets, ranging from −11.4% to +16.3%. It is not part of any rule.

### 13.3 Verdict against the §12 criteria: **all three PASS**

| Criterion | Result | Verdict |
|---|---|---|
| T1: the worst case of each rule is ≤ the worst case of plain LoRA (+124.4%) | +wise +29.8%, +guard +29.8%, guard-LF 0.0% | **PASS** |
| T2: guard-LF is no worse on ≥ 25/28 datasets, with mean ≤ 0 | 28/28, −0.99% | **PASS** |
| T3: +wise mean is below the LoRA mean and ≤ 0 | −0.91% vs +14.89% | **PASS** |

### 13.4 What the second model adds

1. **The main claim carries over to a second model family.** On TimesFM, plain LoRA fine-tuning is much riskier than on Chronos-2: mean +14.9%, worse than zero-shot on 20 of 28 datasets, and +124% on ett2/D. Every interpolation rule turns this into a small average gain with a much smaller tail.
2. **The same risk–return frontier appears.** `+wise` and `+guard` capture real gains: −16% on electricity/15T, −15%/−11% on covid_deaths, −11% on saugeenday/D. But they keep one tail case. `guard-LF` has no tail (28/28, worst 0%), but on fresh data it gives up gains (fresh mean 0% vs −3.3% to −4.3%), exactly as on Chronos-2 (§11).
3. **bizitobs_service misled validation again, on a different model.** Chronos-2 gave +17.5%; TimesFM gives +29.8%. In both cases every validation rule moved to α = 1, and only the frequency rule avoided the loss. On TimesFM, bizitobs_application (also 10-second data) shows the same tendency: +wise1 +33.8%, +wise +4.7%. Across the high-frequency datasets, the mean for `+guard` is +0.67%, while low-frequency datasets average −1.45%.
   - This partly revives the §9 mechanism, which §11 had not reproduced on Chronos-2.
   - It is still a small set: 2–3 informative datasets, all from the bizitobs source. So we report it as an observation, not as a confirmed mechanism.
4. **guard-LF is now confirmed on fresh data for TimesFM**, where it was pre-registered (§12, T2). But as on Chronos-2, its fresh-data result is mostly "stay at zero-shot".

### 13.5 Consequences and next steps

- The paper can now claim, with two model families and pre-registration: *validation-selected weight interpolation bounds the risk of fine-tuning time-series foundation models.*
  - Worst case falls from +53% (Chronos-2) and +124% (TimesFM) to at most +30%, or 0% for guard-LF.
  - Average change goes from fine-tuning damage to a small gain (−0.6% to −1%).
- Before the paper:
  - try a less conservative guard tuned with leave-one-dataset-out across both models, and evaluate it on a fresh hold-out;
  - add seeds 1–2 for statistical tests.

### 13.6 Validation vs test on TimesFM (§9-style; `analysis/tfm_val_vs_test.py`, `results/tfm_all/val_vs_test.csv`)

We compare the gain from fine-tuning (α = 1 vs α = 0) on validation with the gain on test. A run is "misled" when validation reports a gain above 3% but the test result is worse.

| Data frequency | Datasets | Pearson r | Spearman ρ | Sign agreement | Misled |
|---|---|---|---|---|---|
| Daily or coarser | 19 | +0.14 | +0.64 | **89%** | 1 of 2 (m4_weekly, test −1.1%) |
| Hourly or finer | 9 | +0.45 | +0.42 | **56%** | **2 of 4** (bizitobs_service −29.8%, SZ_TAXI/H −1.6%) |

The low-frequency Pearson r is driven by one outlier: covid_deaths, with a validation gain of −225%. The rank correlation is the better summary.

- **The direction matches §9:** validation is a more reliable signal for daily-or-coarser data than for high-frequency data (sign agreement 89% vs 56%).
- **The only large misled case is again bizitobs_service.** The other misled cases are small (−1% to −2%). As in §11, validation also errs in the pessimistic direction: on bitbrains_rnd/H it reports −36%, but the test gain is +4.8%.
- **The rules are conservative.** The oracle (best α chosen on test) would give −3.42% on average over the 28 datasets. `+wise` (−0.91%) and `guard-LF` (−0.99%) capture about 27–29% of that. On TimesFM, validation mostly says "do not fine-tune", and it is usually right: plain LoRA is harmful on 20 of 28 datasets. The gains it misses are mainly intermediate α values (0.25–0.75) that look worse than α = 0 on validation but help on test (saugeenday, covid_deaths, ett2/W, electricity/W).

## 14. Less conservative guards: offline study on 56 runs (2026-11-22, no GPU)

**Question.** Over 28 datasets, the oracle (best α on test) is −3.4% to −4.0%. The frozen rules capture only about 20–35% of that. Can a less conservative setting recover more gain without bringing back tail risk?

**Data and method** (`analysis/rules_v2.py`, outputs in `results/rules_v2/`):
- 56 runs: Chronos-2 and TimesFM 2.5, 28 datasets each, LoRA full, seed 0, the full α-curves.
- One generalised guard: among α ≤ cap, take the argmin of pooled validation MWQL among the α values that
  - improve pooled validation by more than m (m_HF for hourly-or-finer data),
  - and, optionally, do not worsen the most recent window.
  - Optionally, use α = 0 for high-frequency data.
- Hyper-parameters are chosen by leave-one-dataset-out over the other 27 datasets (both models), with two objectives:
  - mean change;
  - mean change + 0.5 × CVaR10 (the mean of the worst 10% of runs), labelled "mean+risk".

### 14.1 Results (56 runs; change in test MWQL vs zero-shot)

| Rule | Tuned | Mean | Worst | CVaR10 | No worse | Better (< −1%) | Share of oracle |
|---|---|---|---|---|---|---|---|
| LoRA α = 1 | – | +15.87% | +185.6% | +101.6% | 22/56 | 19 | – |
| +wise (argmin) | fixed | −1.21% | +29.8% | +17.1% | 47/56 | 20 | 32% |
| +guard (m = 3%) | fixed | −0.69% | +29.8% | +11.8% | 53/56 | 13 | 19% |
| guard-LF (m = 3%) | fixed | −1.31% | 0.0% | 0.0% | 56/56 | 10 | 35% |
| guard, m tuned | LODO mean | −0.78% | +29.8% | +13.1% | 50/56 | 17 | 21% |
| guard, m + cap tuned | LODO mean | −0.66% | +29.8% | +11.9% | 50/56 | 17 | 18% |
| full family (m, cap, last, HF→0) | LODO mean | −0.34% | +29.8% | +15.9% | 47/56 | 17 | 9% |
| **guard-LF, m tuned** | LODO mean+risk | **−1.53%** | **0.0%** | 0.0% | **56/56** | 13 | **41%** |
| **guard, m_LF/m_HF + cap tuned** | LODO mean+risk | **−1.50%** | **0.0%** | 0.0% | **56/56** | 13 | 40% |
| oracle (best α on test) | – | −3.71% | 0.0% | 0.0% | 56/56 | 30 | 100% |

The LODO choice was the same in all 28 folds for both of the best families:
- guard-LF with **m = 2%**;
- the soft-frequency guard with **m_LF = 2%, m_HF = 10%, cap = 0.5**.

### 14.2 What this shows

1. **Tuning the margin or cap is not enough.**
   - Tuned only for mean gain, every family gets worse than the fixed rules, because the heavy tail (bizitobs_service, ett2/W on Chronos-2) dominates the objective.
   - Tuned with a risk term, the best honest rules reach about 40% of the oracle gain: −1.5% vs −1.3% for frozen guard-LF.
   - The remaining gap is a limit of the **validation signal itself** (§9, §13.6), not of the thresholds.
2. **Two candidates are worth a fresh test:**
   - **(A) guard-LF with m = 2%.** It differs from the frozen m = 3% on only 3 of 56 runs, all gains: Chronos hospital −7.6%, TimesFM m4_daily −3.0% and electricity/D −1.5%.
   - **(B) a soft frequency rule:** m = 2% for daily-or-coarser data; for high-frequency data, m = 10% and α ≤ 0.5.
     - Unlike guard-LF, it can still fine-tune high-frequency data when validation is very clear. Example: TimesFM electricity/15T −12.1%.
     - It has the same tail (0%, 56/56), and its mean is −1.50%.
     - It is more balanced across groups (fresh −1.18% vs −0.32% for A) and across models (Chronos −1.44%, TimesFM −1.56%).
3. Both were chosen after seeing these 56 runs. Neither can be claimed until it is confirmed on configurations that no rule has seen.

### 14.3 Confirmation of the §14 rules: pre-registration (frozen 2026-11-22, before any run)

- **Rules under test:** (A) guard-LF, m = 2%. (B) soft-frequency guard, m_LF = 2%, m_HF = 10%, cap 0.5. Frozen guard-LF (m = 3%) and +wise are reported for reference.
- **Datasets:** 8 GIFT-Eval short-term configurations not used by any run so far (4 low-frequency, 4 high-frequency).
  - Low frequency: restaurant (new source), hierarchical_sales/D, solar/D, us_births/W.
  - High frequency: ett1/H, M_DENSE/H, kdd_cup_2018_with_missing/H, jena_weather/H.
  - Limitation: except restaurant, these share a source with configurations used earlier, so they are new configurations rather than fully new sources.
- **Models and settings:** Chronos-2 and TimesFM 2.5, LoRA full, seed 0, everything else as in §6/§12. For every α, the notebooks also record validation and test coverage and the relative width of the central 80% (q0.1–q0.9) and 60% (q0.2–q0.8) intervals, for RQ4. No rule uses these values. 90% is not possible because TimesFM outputs only the quantiles 0.1–0.9.
- **Criteria (both models pooled, 16 runs):**
  - (C1) each of A and B is no worse than zero-shot (≤ +1%) on ≥ 15/16 runs;
  - (C2) the mean change of each is ≤ the frozen guard-LF (m = 3%);
  - (C3) the worst case of each is ≤ +1%.
- **How the rules are computed:** offline from the α-curves, exactly as in `analysis/rules_v2.py`:
  - (A) `pick(m=0.02, hf_zero=True)`;
  - (B) `pick(m=0.02, m_hf=0.10, cap=0.5)`;
  - high frequency means base frequency H, T or S.
- **Secondary analyses (reported, not criteria):**
  - results per model;
  - +wise and oracle;
  - validation-vs-test sign agreement;
  - coverage at α = 0 vs α = 1 and at the α each rule selects.
- **Runs:** one Kaggle notebook per model, part `N` (`tsfm-chronos-N` and `tsfm-timesfm-N`), about 5–7 GPU-hours each.

## 15. Confirmation of the §14 rules on 8 new configurations × 2 models (2026-11-22)

Runs: Kaggle notebooks `tsfm-chronos-N` (session 372377306) and `tsfm-timesfm-N` (session 372377642), LoRA full, seed 0. Raw files are in `results/chr_N/` and `results/tfm_N/`. Analysis: `analysis/confirm_N.py`; outputs in `results/confirm_N/` (per_run.csv, summary.csv, coverage.csv). The rules were computed offline from the α-curves exactly as frozen in §14.3; the notebooks' own +guard and guard-LF columns give the same α.

### 15.1 Primary result (16 runs; change in test MWQL vs zero-shot, %)

| rule | mean | median | worst | no worse (≤ +1%) | better (< −1%) |
|---|---|---|---|---|---|
| LoRA (α = 1) | +6.28 | +2.02 | +38.04 (Chronos-2, solar/D) | 7/16 | 5/16 |
| **A: guard-LF, m = 2%** | **−0.84** | 0.00 | **0.00** | **16/16** | 1/16 |
| **B: m_LF 2% / m_HF 10% / cap 0.5** | **−0.84** | 0.00 | **0.00** | **16/16** | 1/16 |
| ref: frozen guard-LF, m = 3% | −0.84 | 0.00 | 0.00 | 16/16 | 1/16 |
| ref: +guard, m = 3% | −0.84 | 0.00 | 0.00 | 16/16 | 1/16 |
| ref: +wise | −1.23 | 0.00 | +0.05 | 16/16 | 4/16 |
| oracle (best α on test) | −2.14 | −0.19 | 0.00 | 16/16 | 6/16 |

**Pre-registered criteria:**
- C1: A 16/16, B 16/16 (needed ≥ 15). **PASS.**
- C2: A −0.839%, B −0.839%, frozen guard-LF −0.839%. **PASS, but only as a tie.**
- C3: worst case 0.00% for both. **PASS.**

**The honest reading.** The confirmation passes, but it cannot tell A and B apart from the frozen rule. On all 16 runs A, B, guard-LF (m = 3%) and +guard pick the same α. The only non-zero choice is Chronos-2 on us_births/W (α = 0.5, −13.4%). On every other run validation either got worse with fine-tuning or improved by less than 2%, so every guard returned zero-shot. The extra return that A and B showed offline (§14: −1.53% / −1.50% vs −1.31%) therefore did not show up here. It was not refuted, but these 8 configurations had no runs in the margin band (2–3%) where the rules differ.

### 15.2 Per model and per frequency (mean / worst, %)

| rule | Chronos-2 | TimesFM 2.5 | low freq. | high freq. |
|---|---|---|---|---|
| LoRA (α = 1) | +3.09 / +38.04 | +9.48 / +23.33 | +9.05 / +38.04 | +3.51 / +22.10 |
| A = B = guard-LF | −1.68 / 0.00 | 0.00 / 0.00 | −1.68 / 0.00 | 0.00 / 0.00 |
| +wise | −1.91 / 0.00 | −0.55 / +0.05 | −2.34 / +0.05 | −0.12 / 0.00 |
| oracle | −2.74 | −1.55 | −2.69 | −1.59 |

- Plain LoRA is risky again: it is worse than zero-shot on 9/16 runs, with +38% on Chronos-2 solar/D, +23% on TimesFM us_births/W and +22% on TimesFM ett1/H.
- TimesFM's validation loss at α = 1 rose on all 8 datasets (+0.8% to +64%), so every guard stayed at zero-shot. That was correct on 6/8 runs. The two misses (M_DENSE/H −6.9%, restaurant −1.7%) cost return but not risk.
- +wise was again the best on return and almost risk-free here (worst +0.05%). Over all 72 runs, though, its worst case is still +29.8% (bizitobs_service, §13).

### 15.3 Validation vs test (secondary)
Sign agreement between the validation and test change at α = 1 was 13/16 (low frequency 7/8, high frequency 6/8). The disagreements were:
- Chronos-2 M_DENSE/H: validation worse, test −4.6%;
- TimesFM M_DENSE/H: validation +64%, test −6.9%;
- TimesFM restaurant: validation +0.8%, test −1.7%.

All three are conservative misses. No run had validation that misled towards a harmful α, unlike bizitobs_service and m4_hourly earlier.

### 15.4 Interval coverage (RQ4; mean over 16 runs, test set)

| at | cov 80% (nominal 0.80) | rel. width 80% | cov 60% (nominal 0.60) | rel. width 60% |
|---|---|---|---|---|
| α = 0 (zero-shot) | 0.779 | 0.854 | 0.590 | 0.554 |
| α = 1 (LoRA) | 0.656 | 0.748 | 0.484 | 0.478 |
| α chosen by A / B | 0.776 | 0.854 | 0.585 | 0.553 |

| model | cov80 α=0 | cov80 α=1 | cov60 α=0 | cov60 α=1 |
|---|---|---|---|---|
| Chronos-2 | 0.758 | 0.708 | 0.566 | 0.529 |
| TimesFM 2.5 | 0.801 | 0.604 | 0.613 | 0.440 |

- Fine-tuning makes the intervals narrower (−12% width) and **under-covers**. TimesFM's 80% interval drops from 0.80 to 0.60, and on us_births/W it falls to 0.03.
- Zero-shot is almost perfectly calibrated (0.78 / 0.59).
- The guards keep the zero-shot calibration because they mostly keep α = 0.
- This is new evidence for RQ4: the harm of naive fine-tuning is larger for uncertainty than for point/quantile accuracy, and α-selection protects both.
- Caveat: this is only 16 runs, and coverage was not recorded for the earlier 56.

### 15.5 Pooled over all 72 runs (56 from §14 + these 16; secondary)

| rule | mean | median | worst | no worse | better |
|---|---|---|---|---|---|
| LoRA (α = 1) | +13.74 | – | +185.6 | 29/72 | – |
| A: guard-LF m = 2% | −1.37 | 0.00 | 0.00 | 72/72 | 14/72 |
| B: m_LF 2% / m_HF 10% / cap 0.5 | −1.35 | 0.00 | 0.00 | 72/72 | 14/72 |
| frozen guard-LF m = 3% | −1.20 | 0.00 | 0.00 | 72/72 | 11/72 |
| +wise | −1.21 | 0.00 | +29.8 | 63/72 | 24/72 |
| oracle | −3.36 | – | 0.00 | – | – |

The 56 runs include the data the rules were tuned on (LODO in §14), so only the 16 new runs count as confirmation.

### 15.6 Consequences for the paper
1. **Main claim (bounded risk), now confirmed on 72 runs, 2 models and 36 configurations.** Validation-selected interpolation with a frequency-aware guard is never worse than zero-shot, while plain LoRA is worse on 60% of runs, with a worst case of +186%.
2. **The return claim stays modest.** Guards capture ~40% of the oracle gain (16 new runs: 0.84 of 2.14; all 72: 1.37 of 3.36). The less conservative A/B rules are confirmed safe, but their extra return over m = 3% is not confirmed: they tied here.
   - Primary rule for the paper: A (guard-LF, m = 2%). It is the simplest, it was pre-registered, and it is no worse than any alternative on any data so far. B is reported as a variant.
3. **Calibration (RQ4) is a strong secondary result.** Fine-tuning hurts coverage more than accuracy, and the guard keeps zero-shot calibration.
4. **Next:**
   - statistical tests (Wilcoxon / bootstrap CIs on the 72 runs);
   - more seeds on a subset;
   - an ablation of fine-tuning length;
   - coverage for the earlier runs if GPU time allows;
   - then writing.

## 16. Statistical tests on all 72 runs (2026-11-22, no GPU)

Script: `analysis/stats_tests.py`; outputs in `results/stats/` (per_run_72.csv, bootstrap_ci.csv, wilcoxon.csv).
- **Unit:** one run = (model, dataset), 36 configurations × 2 models, LoRA full, seed 0.
- **Effect:** change in test MWQL vs zero-shot (%), from the α-curves.
- **Bootstrap:** cluster bootstrap over datasets (both models of a configuration resampled together), 10,000 draws.
- **Wilcoxon:** signed-rank, Pratt zero handling, two-sided. Holm correction over the six primary tests.

### 16.1 Bootstrap 95% CIs (all 72 runs)

| rule | mean change [95% CI] | geometric-mean change [95% CI] | runs worse than zero-shot (> +1%) | worst |
|---|---|---|---|---|
| LoRA (α = 1) | +13.74 [+5.94, +22.59] | +10.81 [+4.55, +17.95] | 60% [46, 72] | +185.6 |
| **A: guard-LF m = 2%** | **−1.37 [−2.43, −0.54]** | −1.44 [−2.54, −0.56] | **0% [0, 0]** | **0.0** |
| B: m_LF 2% / m_HF 10% / cap 0.5 | −1.35 [−2.42, −0.50] | −1.42 [−2.55, −0.52] | 0% [0, 0] | 0.0 |
| frozen guard-LF m = 3% | −1.20 [−2.28, −0.37] | −1.27 [−2.39, −0.38] | 0% [0, 0] | 0.0 |
| +wise | −1.21 [−3.39, +1.02] | −1.53 [−3.76, +0.67] | 12% [6, 21] | +29.8 |
| oracle | −3.36 [−5.01, −1.96] | −3.55 [−5.31, −2.04] | 0% | 0.0 |

By subset (mean change of A [95% CI]):
- Chronos-2: −1.85 [−3.30, −0.66].
- TimesFM 2.5: −0.89 [−1.81, −0.19].
- Low frequency (46 runs): −2.15 [−3.69, −0.90].
- High frequency (26 runs): 0 by construction. B gains −0.46 [−1.39, 0] there.

LoRA at α = 1:
- Chronos-2: +13.79 [+3.65, +27.00].
- TimesFM 2.5: +13.69 [+6.92, +22.44].

### 16.2 Wilcoxon signed-rank (all 72 runs)

| test | non-zero pairs | mean diff. | p | p (Holm) |
|---|---|---|---|---|
| A vs zero-shot | 15 | −1.37 | 1.1e-4 | 5.6e-4 |
| B vs zero-shot | 15 | −1.35 | 1.1e-4 | 5.6e-4 |
| +wise vs zero-shot | 53 | −1.21 | 0.023 | 0.047 |
| LoRA vs zero-shot | 72 | +13.74 | 2.8e-4 | 8.5e-4 |
| A vs LoRA | 67 | −15.11 | < 1e-5 | 2e-5 |
| A vs +wise | 41 | −0.16 | 0.71 | 0.71 |
| A vs guard-LF m = 3% (secondary) | 3 | −0.17 | 0.083 | – |

### 16.3 Reading
1. **Accuracy.** All three guards improve on zero-shot with a CI that excludes 0, while plain LoRA is significantly worse. The A − LoRA difference is large and significant for each model separately (Chronos-2 p = 0.009, TimesFM p = 1.3e-4).
2. **Risk.** Guards have 0 runs worse than zero-shot. For +wise the CI is 6–21% of runs, and for LoRA 46–72%. +wise has the same average gain as A (p = 0.71) but not its risk bound. This is the risk–return statement the paper makes.
3. **A vs the m = 3% guard.** They differ on only 3 of 72 runs (p = 0.08). The case for m = 2% over m = 3% is therefore weak, so the paper should present m as a robust setting rather than claim a better margin.
4. **Caveat.** The 56 earlier runs were used to design the rules. The confirmation-only subset (16 runs) gives A −0.84 [−2.52, 0.00], with 1 non-zero pair, so it has no power on its own; its evidence is the 0/16 worse.

## 17. Stability run: extra seeds and fine-tuning length (registered 2026-11-22, before the run)

- **Datasets (part S):**
  - three low-frequency datasets with seed-0 results for both models: us_births/D, ett1/W, solar/W;
  - one high-frequency dataset where validation misled the guard: bizitobs_service.
- **Notebooks:** `tsfm-chronos-S` (STAB = True) and `tsfm-timesfm-S`.
- **Seeds:** LoRA full, 4000 steps, seeds 1 and 2.
  - TimesFM: all 4 datasets.
  - Chronos-2: bizitobs_service only; the other three already have seeds 0–2 (§6).
- **Fine-tuning length:** LoRA full with 500 and 1000 steps (`lora_s500`, `lora_s1000`), seed 0, both models, all 4 datasets. 4000 steps = the existing seed-0 runs.
- **Questions (descriptive; no pass/fail criteria):**
  - (S1) Is the α chosen by A, B and guard-LF the same across seeds, and is the guard's sign (no worse than zero-shot) stable?
  - (S2) Does shorter fine-tuning reduce the harm of α = 1, and does validation-selected α still add value at 500 / 1000 steps?
  - (S3) Is the bizitobs_service failure of validation (§8, §13) seed-specific?

## 18. Stability run results (2026-11-22)

**Runs:**
- `tsfm-chronos-S` (session 372450635, 2.0 h) and `tsfm-timesfm-S` (session 372450212, ~4.3 h). Both ran without errors.
- Raw files are in `results/chr_S/` and `results/tfm_S/`. Code and logs are archived in `runs/`.
- Analysis: `analysis/stability_S.py`, with outputs in `results/stability_S/per_run.csv`.
- Seed 0 at 4000 steps comes from the original runs (partA, holdoutH1, tfm_A, tfm_BH). Chronos-2 seeds 1–2 on the three low-frequency datasets come from confirmA (§6).
- There are 40 (model, dataset, length, seed) runs in total.

### 18.1 S1: stability across seeds (4000 steps, seeds 0/1/2)

| Model | Dataset | LoRA α = 1 (%) | α chosen by A | A (%) | α chosen by B | B (%) | +guard m = 3% (%) | +wise (%) |
|---|---|---|---|---|---|---|---|---|
| Chronos-2 | us_births/D | −8.5 / −8.4 / −8.9 | 0.75 / 0.75 / 0.75 | −7.2 / −7.2 / −7.5 | 0.5 / 0.25 / 0.5 | −4.6 / −2.5 / −4.9 | = A | = A |
| Chronos-2 | ett1/W | +32.2 / +32.5 / +32.5 | 0 / 0 / 0 | 0 | 0 | 0 | 0 | +0.2 / −0.1 / +0.1 |
| Chronos-2 | solar/W | +186 / +190 / +167 | 0 / 0 / 0 | 0 | 0 | 0 | 0 | +2.9 / +2.7 / +7.4 |
| Chronos-2 | bizitobs_service | +17.5 / +15.9 / +17.3 | 0 / 0 / 0 | 0 | 0 | 0 | **+17.5 / +15.9 / +17.3** | +17.5 / +15.9 / +17.3 |
| TimesFM 2.5 | us_births/D | +7.0 / −1.0 / +7.3 | 0 / 0 / 0 | 0 | 0 | 0 | 0 | −2.7 / −1.0 / −2.4 |
| TimesFM 2.5 | ett1/W | +20.4 / +47.1 / +47.1 | 0 / 0 / 0 | 0 | 0 | 0 | 0 | 0.0 / +0.3 / +1.4 |
| TimesFM 2.5 | solar/W | +29.4 / +23.6 / +34.4 | 0 / 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| TimesFM 2.5 | bizitobs_service | +29.8 / +21.3 / +15.3 | 0 / 0 / 0 | 0 | 0 / 0.5 / 0.5 | 0 / **−7.1 / −8.1** | **+29.8 / +4.0 / +15.3** | +29.8 / +4.0 / +15.3 |

- **Rule A chose the same α for every seed in all 8 (model, dataset) pairs, and it was never worse than zero-shot.**
- Rule B was also never worse. Its α varied with the seed in two pairs, and on TimesFM bizitobs_service its 10% high-frequency margin with cap 0.5 recovered a gain of 7–8% for seeds 1 and 2.
- The sign of plain LoRA's harm is stable across seeds, except TimesFM on us_births/D (−1% to +7%).

### 18.2 S2: fine-tuning length (seed 0; change in test MWQL vs zero-shot, %)

| Model | Dataset | LoRA 500 / 1000 / 4000 steps | Rule A 500 / 1000 / 4000 | +wise 500 / 1000 / 4000 |
|---|---|---|---|---|
| Chronos-2 | us_births/D | −2.8 / −4.7 / −8.5 | 0 / 0 / −7.2 | −2.8 / −4.7 / −7.2 |
| Chronos-2 | ett1/W | +3.5 / +11.9 / +32.2 | 0 / 0 / 0 | 0 / +0.6 / +0.2 |
| Chronos-2 | solar/W | **+81.1 / +74.7** / +185.6 | 0 / 0 / 0 | +2.7 / +2.1 / +2.9 |
| Chronos-2 | bizitobs_service | +1.4 / +4.7 / +17.5 | 0 / 0 / 0 | +0.3 / +4.7 / +17.5 |
| TimesFM 2.5 | us_births/D | −4.3 / +1.9 / +7.0 | −4.3 / 0 / 0 | −4.3 / −2.5 / −2.7 |
| TimesFM 2.5 | ett1/W | +17.1 / +19.8 / +20.4 | 0 / 0 / 0 | +1.6 / +0.9 / 0.0 |
| TimesFM 2.5 | solar/W | **+48.8 / +46.4** / +29.4 | 0 / 0 / 0 | 0 / 0 / 0 |
| TimesFM 2.5 | bizitobs_service | +3.4 / +10.9 / +29.8 | 0 / 0 / 0 | −3.1 / +10.9 / +29.8 |

- Shorter fine-tuning reduces the harm in some cases (Chronos ett1/W +32% → +3.5%, both bizitobs_service runs). It does not remove it: at 500 steps plain LoRA is still worse than zero-shot in 6 of 8 runs, by up to +81% (Chronos-2 solar/W).
- For TimesFM on solar/W the harm is larger at 500 steps than at 4000. Choosing the fine-tuning length is therefore not a substitute for the guard.
- Rule A was never worse than zero-shot at any length.

### 18.3 S3: is the bizitobs_service failure seed-specific? No.

For both models and all three seeds, validation reports a large gain from fine-tuning (−13% to −23% MWQL at α = 1), but the test loss rises by 15% to 30%.
- The plain guard (m = 3%) was misled in 5 of the 6 runs. The exception is TimesFM seed 1, where it chose α = 0.75 and still lost 4%.
- +wise behaves the same way.
- The frequency switch avoided the loss in all six runs.

So the failure of validation on this 10-second dataset is a property of the data and the validation design, not chance. This is the strongest single argument for the frequency switch.

### 18.4 Over all 40 stability runs

| Rule | Worst change vs zero-shot |
|---|---|
| A | 0.0% |
| B | 0.0% |
| +guard m = 3% | +29.8% |
| +wise | +29.8% |

Plain LoRA was worse than zero-shot in 33 of 40 runs.

Consequences for the paper:
- The bounded-risk result holds across seeds and fine-tuning lengths.
- Rule B's high-frequency margin can recover gains on high-frequency data without loss. This was seen on 2 of 2 TimesFM bizitobs seeds, but the number is small.

## 19. Coverage re-run with conformal recalibration (registered 2026-11-22, before the runs)

- **Purpose (RQ4):** in §15, coverage was measured only on the 16 confirmation runs. This run re-runs the 28 earlier configurations for both models (LoRA full, seed 0, settings unchanged). For every α it records:
  - coverage and relative width of the 80% and 60% central intervals (validation and test);
  - **split-conformal (CQR) recalibration**, fitted on the validation windows and evaluated on test (`test_cqr_cov*`, `test_cqr_width*`).
  - Conformal scores are pooled across series after scaling by the mean |y| of each window's context.
- **Notebooks:**
  - `tsfm_chronos_C1` (A+B+H1, 17 configurations) and `tsfm_chronos_C2` (H2+F1+F2, 11);
  - `tsfm_timesfm_C1` (A+B, 12) and `tsfm_timesfm_C2` (H1+H2+F1+F2, 16).
- **Runs:** the notebooks are run by collaborators on their own Kaggle accounts.
- **Questions (descriptive):**
  - (R1) Does the drop in coverage from α = 0 to α = 1 seen in §15 hold across 56 runs?
  - (R2) Does CQR on the validation windows bring coverage back to nominal at α = 1 and at the selected α, and at what width?
  - (R3) Reproducibility: is the seed-0 MWQL change at α = 1 and at the selected α close to the original runs? Fine-tuning on GPU is not bit-exact.

### 19.1 Results for Chronos-2 (2026-11-23; TimesFM in §19.2)

**Runs:**
- `tsfm-chronos-C1` (17 configurations, session 372569977) and `tsfm-chronos-C2` (11, session 372569109), run by a collaborator.
- Both ran without errors. Files are in `results/chr_C1`, `results/chr_C2`, with code and logs in `runs/`.
- Analysis: `analysis/cover_C.py`, with outputs in `results/cover_C/`.

**Bug found and fixed.**
- In 4 configurations (bitbrains_rnd/H, bitbrains_fast_storage/H, electricity/W, electricity/D) some windows have an all-NaN context. The per-window scale then became NaN, because `max(nan, 1e-8)` returns nan, so the CQR quantile was NaN and the CQR coverage was reported as 0.
- These 4 are excluded from the CQR rows below. MWQL and raw coverage are not affected.
- The code is fixed in all three scripts (all-NaN or zero scale → 1, and non-finite scores are dropped). The not-yet-run TimesFM and Chronos-Bolt notebooks were rebuilt before they ran.

**R3. Reproducibility: exact.**
- Zero-shot MWQL matches the original runs to within 0.0003%.
- The change at α = 1 and the α chosen by rule A are identical on all 28 configurations.
- Chronos-2 fine-tuning with a fixed seed is deterministic on the Kaggle T4.

**R1. Coverage drops with fine-tuning (28 configurations, test windows, mean):**

| at | cov 80% | rel. width 80% | cov 60% | rel. width 60% |
|---|---|---|---|---|
| α = 0 (zero-shot) | 0.775 | 1.199 | 0.569 | 0.537 |
| α = 1 (LoRA) | 0.676 | 0.667 | 0.482 | 0.392 |
| α chosen by rule A | 0.771 | 1.152 | 0.559 | 0.506 |

- The 80% coverage fell by more than 0.05 from α = 0 to α = 1 in 13 of 28 configurations and rose in none.
- The drop is larger for low-frequency data (0.790 → 0.663) than for high-frequency data (0.744 → 0.705).
- Rule A keeps the zero-shot calibration.
- With §15 this is now shown on 44 Chronos-2 runs (28 here + 16 in §15).

**R2. Split-conformal recalibration on the validation windows (24 configurations with valid CQR):**

| at | cov 80% raw → CQR | width 80% raw → CQR | cov 60% raw → CQR | mean \|cov80 − 0.80\| raw → CQR |
|---|---|---|---|---|
| α = 0 | 0.777 → 0.792 | 0.659 → 0.673 | 0.569 → 0.587 | 0.062 → 0.067 |
| α = 1 | **0.662 → 0.779** | 0.548 → 0.699 | 0.469 → 0.579 | **0.139 → 0.069** |
| rule A | 0.772 → 0.796 | 0.604 → 0.628 | 0.558 → 0.587 | 0.062 → 0.064 |

- CQR fitted on the V validation windows **repairs the under-coverage of the fine-tuned model**: the 80% coverage goes from 0.66 to 0.78 (nominal 0.80), at the cost of 28% wider intervals.
- For zero-shot and rule A, which are already close to nominal, CQR changes little.
- The validation windows that choose α can therefore also calibrate the intervals of the chosen model, with no extra data. This answers the second half of RQ4 for Chronos-2.
- Caveat: average coverage hides per-configuration errors. Mean absolute deviation from nominal stays about 0.06–0.07, because three validation windows are few for some datasets.

### 19.2 Results for TimesFM 2.5 (2026-11-24) and both models together

**Runs:**
- `tsfm-timesfm-C1` (12 configurations, session 373128035, 5 h 8 min) and `tsfm-timesfm-C2` (16, session 373065735, 7 h 23 min), run by a collaborator.
- No error cells, no failed fits, and no NaN CQR rows: the fix of §19.1 works.
- Files are in `results/tfm_C1`, `results/tfm_C2`, with code and logs in `runs/`. Analysis: `analysis/cover_C.py`.

**R3. Reproducibility: exact for TimesFM too.**
- Zero-shot MWQL matches to within 0.0003%.
- The change at α = 1 and the α chosen by rule A are identical on 28/28.
- Over both models, 56/56 reproduce exactly.

**R1/R2 (28 configurations, test windows, mean; nominal 0.80):**

| at | cov 80% raw → CQR | mean \|cov80 − 0.80\| raw → CQR | configs within ±0.05 raw → CQR | median rel. width 80% raw → CQR |
|---|---|---|---|---|
| α = 0 (zero-shot) | 0.804 → 0.808 | 0.034 → 0.054 | 23 → 14 | 0.275 → 0.334 |
| α = 1 (LoRA) | **0.607 → 0.810** | **0.195 → 0.050** | 8 → 20 | 0.198 → 0.428 |
| rule A | 0.800 → 0.801 | 0.030 → 0.049 | 24 → 17 | 0.264 → 0.289 |

- **Fine-tuning hurts TimesFM calibration even more than Chronos-2.**
  - 80% coverage drops from 0.80 to 0.61, more on low-frequency data (0.81 → 0.58).
  - Over both models, the 80% coverage fell by more than 0.05 in 31/56 configurations and rose in none.
- **Rule A keeps the zero-shot calibration:** 0.800, with 24/28 configurations within ±0.05.
- **CQR repairs the fine-tuned model** (0.607 → 0.810; per-configuration error 0.195 → 0.050). The intervals become wider (median 0.20 → 0.43): the narrow fine-tuned intervals were over-confident.
- **New observation (exploratory, not pre-registered).** For models that are already calibrated (zero-shot, rule A), CQR on three validation windows makes the per-configuration calibration *worse*: 0.030 → 0.049, and configurations within ±0.05 fall from 24 to 17.
  - The same holds for Chronos-2 (0.062 → 0.064–0.067).
  - Mean coverage is unchanged, so this is noise from estimating a quantile from few windows, not bias.
  - Practical reading: recalibrate only a fine-tuned model (α > 0); keep zero-shot intervals as they are. To be checked on Bolt and the §22 runs before it is claimed.
- **Reporting note.** Mean relative width is dominated by a few series (Chronos-2 mean 1.20 vs median 0.24; Bolt mean 11.7 at α = 1). The paper reports **median** widths.

**Code check after these runs: no change needed.**
- The scripts ran cleanly on both models and reproduced earlier results exactly.
- The only log warning is the harmless "unauthenticated requests to the HF Hub".
- The §22 notebooks already contain the same (fixed) code.

## 20. Third model: Chronos-Bolt (base) — pre-registration (written 2026-11-22, before any run)

- **Model:** `amazon/chronos-bolt-base` (205M). It is a T5 encoder-decoder over patches with a direct multi-step quantile head. Both the architecture and the training objective differ from Chronos-2 (encoder with group attention) and TimesFM 2.5 (decoder).
- **Fine-tuning, fixed now:**
  - own LoRA, r = 8, scaling 16;
  - adapters on all q/k/v/o projections of encoder self-attention, decoder self-attention and cross-attention, and on the output layer of the output patch embedding;
  - AdamW, lr 1e-4, linear decay, clipping 1.0, batch 32, 4000 steps, fine-tuning context 1024;
  - loss: the model's own quantile loss;
  - prediction uses up to 2048 context points.
  - A local CPU test with `chronos-bolt-tiny` checked the code path: shapes, LoRA merge, α = 0 equal to zero-shot, coverage and CQR.
  - There is no GPU quick test and **no adjustment is allowed**: the results are reported as they come.
- **Data:** all 36 configurations (28 + 8 confirmation). Notebooks `tsfm_bolt_B1` (A+B+H1) and `tsfm_bolt_B2` (H2+F1+F2+N).
- **Rules (all frozen):** rule A (guard-LF, m = 2%) is primary; B, guard-LF m = 3%, +guard and +wise are reported.
- **Criteria (36 runs, change in test MWQL vs zero-shot):**
  - (K1) the worst case of each interpolation rule is ≤ the worst case of plain LoRA;
  - (K2) rule A is no worse than zero-shot (≤ +1%) on ≥ 34/36 configurations, with mean ≤ 0%;
  - (K3) the +wise mean is below the plain-LoRA mean.
- **Secondary:** coverage and CQR as in §19, and validation-vs-test sign agreement by frequency.

## 21. Third model: Chronos-Bolt (base) — results (2026-11-24)

**Runs:**
- `tsfm-bolt-B1` (A+B+H1, 17 configurations, session 372813455, 7 h 10 min) and `tsfm-bolt-B2` (H2+F1+F2+N, 19 configurations, session 372814034, 8 h 46 min), both on a T4 x2, run by a collaborator.
- Both ran without error cells. Files are in `results/bolt_B1`, `results/bolt_B2`, with code and logs in `runs/`.
- Analysis: `analysis/bolt_B.py`, with outputs in `results/bolt_B/`.
- The fine-tuning recipe was fixed in §20 and not adjusted. High frequency = the §14 HF set, plus base frequency H/T/S for the 8 N configurations.

### 21.1 Primary result (36 runs; change in test MWQL vs zero-shot, %)

| rule | mean | median | worst | no worse (≤ +1%) | better (< −1%) |
|---|---|---|---|---|---|
| LoRA (α = 1) | +42.81 | +2.63 | +952.1 | 15/36 | 13 |
| **A: guard-LF m = 2%** | **−0.85** | 0.00 | **+6.87** | **34/36** | 4 |
| B: m_LF 2% / m_HF 10% / cap 0.5 | −4.59 | 0.00 | +6.87 | 34/36 | 6 |
| guard-LF m = 3% | −0.85 | 0.00 | +6.87 | 34/36 | 4 |
| +guard m = 3% | −4.39 | 0.00 | +12.26 | 33/36 | 6 |
| +wise | −4.69 | 0.00 | +12.26 | 31/36 | 12 |
| oracle (best α on test) | −6.72 | −1.66 | 0.00 | 36/36 | 21 |

### 21.2 Verdict against the §20 criteria: **all three PASS**

- **K1 PASS.** The worst case of every interpolation rule (+6.9% to +12.3%) is far below plain LoRA's (+952%).
- **K2 PASS, at the threshold.** Rule A is no worse than zero-shot on exactly 34/36, with mean −0.85%.
- **K3 PASS.** The +wise mean is −4.69%, against +42.81% for plain LoRA.

### 21.3 What Bolt adds

1. **Plain fine-tuning is much more harmful on Bolt than on the other two models.**
   - Mean +42.8%, worse on 21/36.
   - Extreme cases: car_parts_with_missing +952%, solar/W +203%, covid_deaths +176%, bitbrains_fast_storage/H +69%.
   - For comparison, on 72 Chronos-2/TimesFM runs the mean was +13.7% and the worst +186%.
   - The fixed recipe (lr 1e-4, 4000 steps), set without a GPU quick test, is clearly too aggressive for Bolt on several datasets. Interpolation therefore matters most exactly where fine-tuning is least reliable.
2. **The guard does not fully hold its 0% worst case on Bolt.** Rule A is worse than zero-shot on two low-frequency configurations:
   - covid_deaths: +4.85% at α = 0.25;
   - kdd_cup_2018_with_missing/D: +6.87% at α = 0.5.
   - In both, pooled validation improved by more than 2% at the chosen α, but the test windows did not. This is the validation-signal limit of §9/§13.6, not a threshold problem.
   - On Chronos-2 and TimesFM the worst case of rule A was 0.00% on all 72 runs. The paper must say that the bound is empirical and **not guaranteed**: on a third model it held on 34/36 runs, with a worst case of +6.9% instead of +952%.
3. **Rule A captures little of the oracle on Bolt** (−0.85 of −6.72, 13%), because guard-LF sets α = 0 for all 13 high-frequency configurations.
   - On Bolt the large gains are on high-frequency data: bizitobs_service −73%, bizitobs_application −67% (HF oracle −11.8%).
   - Rule B, which still allows α ≤ 0.5 for HF data under a strong validation signal, catches most of this: HF −10.5%, overall −4.59%, with the same worst case as A (+6.87%).
   - This is the first data where B is clearly better than A, while being equally safe. It supports reporting B as the variant for users who accept a weaker HF prior. A stays primary, as pre-registered.
4. **Validation vs test sign agreement at α = 1:** 26/36 (LF 18/23, HF 8/13). This is similar to the other models; high-frequency validation is the less reliable signal.
5. **The 8 new configurations (N) behave like the 28.** Rule A mean −0.72%, worst 0.00%; plain LoRA mean +6.2%, worst +46%.

### 21.4 Coverage and CQR (secondary, 36 configurations, no CQR NaN rows)

| at | cov 80% raw → CQR | rel. width 80% raw → CQR | cov 60% raw → CQR |
|---|---|---|---|
| α = 0 (zero-shot) | 0.793 → 0.795 | 0.797 → 0.797 | 0.598 → 0.586 |
| α = 1 (LoRA) | **0.675 → 0.774** | 11.71 → 11.89 | 0.482 → 0.583 |
| α chosen by rule A | 0.780 → 0.787 | 0.791 → 0.792 | 0.577 → 0.580 |

- The same pattern as Chronos-2 (§19.1):
  - fine-tuning lowers coverage, more on low-frequency data (LF 0.805 → 0.652, HF 0.772 → 0.716);
  - rule A keeps zero-shot calibration;
  - CQR on the validation windows brings the fine-tuned model back near nominal (0.675 → 0.774).
- The mean α = 1 width (11.7) is dominated by the configurations where fine-tuning diverged, such as car_parts_with_missing.

### 21.5 Consequences for the paper

1. **Main claim now holds on three models with different architectures.**
   - Validation-selected interpolation removes the heavy tail of fine-tuning: worst case +6.9% vs +952% on Bolt, and 0% vs +186% on Chronos-2/TimesFM.
   - Over 108 runs, rule A is no worse than zero-shot on 106.
2. **Honest wording of the bound.** "Never worse" is replaced by "no worse on 106/108 runs; worst +6.9%". Both exceptions are low-frequency configurations where validation and test disagree.
3. **Rule B gets a stronger place as a variant.** On Bolt it recovers most of the high-frequency gain at the same worst case as A.
4. **Pending:** (TimesFM C1/C2 are now done, §19.2.) Then:
   - pooled statistics over all three models;
   - the paper tables.

## 22. Interpolation vs early stopping — pre-registration (written 2026-11-24, before any run)

**Question.** Is the gain from selecting α on validation windows due to *interpolation*, or would the same validation windows used for **early stopping** (choosing the fine-tuning step) do as well? Can the two be combined? This is the most likely reviewer objection. It matters most for Chronos-Bolt, where the fixed recipe diverged on several datasets (§21).

**Design (frozen now):**
- **Models and data:** Chronos-2, TimesFM 2.5 and Chronos-Bolt base; all 36 configurations; LoRA full, seed 0; every setting as in §6/§12/§20 (4000 steps, linear decay; no setting changes).
- **Checkpoints:** one fine-tuning run per configuration. During it, the LoRA update is recorded at steps s ∈ {500, 1000, 2000} and at the end (4000).
  - For each s, θ_s = θ0 + ΔLoRA(s).
  - At the final step the run also records the usual α-curve (α ∈ {0, .1, .25, .5, .75, 1}), which doubles as a reproducibility check against §15/§19/§21.
  - For s ∈ {500, 1000, 2000}, the notebooks evaluate θ(s, α) = θ0 + α·ΔLoRA(s) for α ∈ {.25, .5, .75, 1} on the V validation windows and the test windows, with the same metrics, coverage and CQR columns as before.
- **Sanity check, built into the code:** θ0 + ΔLoRA(4000) must equal the merged fine-tuned weights (max abs. difference < 1e-4).
- **Parts:** three notebooks per model (E1, E2, E3: 12 configurations each, balanced by fine-tuning time), 9 notebooks in total.

**Methods.** All are chosen on validation only (pooled MWQL over the V windows; "last" = most recent window). HF = the §21 definition. Grid S = {500, 1000, 2000, 4000}.

| label | candidates | rule |
|---|---|---|
| **ES** (plain early stopping) | θ_s, s ∈ S, α = 1 | argmin pooled validation |
| **ES+guard** | zero-shot ∪ {θ_s} | rule-A guard on the step axis: m = 2%, last window not worse, HF → zero-shot |
| **A** (primary, as before) | θ(4000, α), α ∈ {0, .1, .25, .5, .75, 1} | rule A |
| **Joint** | zero-shot ∪ {θ(s, α): s ∈ S, α ∈ {.25, .5, .75, 1}} | rule-A guard over the joint grid |
| oracle-step, oracle-joint (reference) | the same grids | best on test |

Secondary variants (reported, not claimed): ES+guard and Joint without HF → zero-shot, and with the rule-B setting (m_HF = 10%, α ≤ 0.5 for HF).

**Pre-registered comparisons (108 runs pooled; change in test MWQL vs zero-shot; paired Wilcoxon signed-rank, Holm over P2–P3):**
- **(P1) Plain early stopping does not bound risk.** It is supported if ES has a worst case > +10% **and** is no worse than zero-shot (≤ +1%) on fewer runs than A.
- **(P2) α-axis vs step-axis under the same guard (A vs ES+guard).**
  - "Interpolation is better": mean(A) < mean(ES+guard) and one-sided p < 0.05.
  - "Equivalent": |mean difference| < 0.5 pp and neither one-sided test is significant.
  - Otherwise "early stopping is better", reported as such.
- **(P3) Combining helps (Joint vs A).** It is supported if mean(Joint) ≤ mean(A) − 0.5 pp, one-sided p < 0.05, **and** worst(Joint) ≤ worst(A) + 1 pp.
- **No-change rule.** Whatever P2/P3 show, rule A stays the primary method of the paper; this section can only add or qualify claims.

**Secondary (reported):**
- per model and LF/HF;
- whether Joint or ES+guard avoids the two Bolt failures of rule A (covid_deaths, kdd_cup_2018_with_missing/D);
- coverage and CQR at the selected point;
- the compute cost of each method, from logged evaluation time per (s, α);
- reproducibility of the 4000-step α-curve.

**Runs:** 9 Kaggle notebooks (`tsfm-{chronos,timesfm,bolt}-E{1,2,3}`), about 6–7 GPU-hours each. Analysis: `analysis/es_E.py`, written and tested on CPU smoke outputs on 2026-11-24, before any E result exists.
- **Code check (CPU smoke test, all three models):** θ0 + ΔLoRA(final) reproduces the merged weights exactly (max abs. difference 0), and the intermediate checkpoints and α rows are written as designed.

### 22.1 Interim check (2026-11-24; 3 of 9 notebooks — **not the pre-registered test**, which needs all 108 runs)

**Runs so far:**
- `tsfm-chronos-E1` (session 373283801, 10.0 h) and `tsfm-chronos-E2` (373283641, 7.7 h).
- `tsfm-bolt-E3` (373314636, version 2, 5.7 h; version 3 was a duplicate start, cancelled).

**Code check:**
- No error cells, no failed fits.
- The built-in checkpoint check gave max |θ0 + ΔLoRA(4000) − merged| = 0 on all 36 runs.
- The 4000-step α-curve reproduces the earlier runs exactly (median |diff| 0.0 pp).
- **No code change is needed.**
- Runtime note: Chronos-2 evaluation is slow on long series (electricity/15T, electricity/H and m4_daily take about 2.5 h each for 18 evaluations), which is why chronos-E1 took 10.0 h. The estimate for chronos-E3 (long series: kdd_cup_2018_with_missing/H, M_DENSE/H, ett1/H) is about 4.2 h of fine-tuning plus 3–4 h of evaluation, below the 12 h limit.

**Interim numbers (36 runs: Chronos-2 × 24, Bolt × 12; change in test MWQL vs zero-shot):**

| method | mean | worst | no worse |
|---|---|---|---|
| LoRA (4000, α = 1) | +41.7% | +952% | 17/36 |
| ES | +5.5% | +69.8% | 22/36 |
| ES+guard | −1.32% | +2.2% | 35/36 |
| A | −1.86% | 0.0% | 36/36 |
| Joint | −1.85% | +2.2% | 35/36 |

These are recorded only to show that the pipeline works. No claim is made until all 9 notebooks are in.

### 22.2 Results (2026-11-26; all 9 notebooks, 108 runs — the pre-registered test)

**Runs:**
- All nine `tsfm-*-E*` notebooks finished. The last was `tsfm-timesfm-E2` (version 8, session 374196418, 5.7 h).
- An earlier attempt of timesfm-E2 (version 2) failed because Internet was off in that session; versions 3–7 were cancelled starts.
- No error cells, no failed fits. The checkpoint check gave 0 on all 108 runs.
- The 4000-step α-curves reproduce the earlier runs exactly on all three models (median |diff| 0.0 pp).
- Files are in `results/{chr,tfm,bolt}_E{1,2,3}`, with code and logs in `runs/`.
- Analysis: `analysis/es_E.py`, output in `results/es_E/` (`report.txt`, `per_run.csv`, `selected.csv`, `summary.csv`).

**Primary table (108 runs; change in test MWQL vs zero-shot, %):**

| method | mean | median | worst | no worse (≤ +1%) | better (< −1%) |
|---|---|---|---|---|---|
| LoRA (4000 steps, α = 1) | +23.43 | +3.09 | +952.1 | 44/108 | 37 |
| ES (plain early stopping) | +5.35 | +0.32 | +176.2 | 64/108 | 47 |
| ES+guard | −0.79 | 0.00 | +10.25 | 105/108 | 15 |
| **A (α interpolation, guard)** | **−1.20** | 0.00 | **+6.87** | **106/108** | 18 |
| Joint (step × α, guard) | −1.26 | 0.00 | +9.13 | 104/108 | 22 |
| oracle-step / oracle-joint | −4.60 / −5.36 | – | 0.00 | 108/108 | 52 / 63 |

**Per model (mean / worst):**
- A: Chronos-2 −1.85 / 0.00, TimesFM −0.89 / 0.00, Bolt −0.85 / +6.87.
- ES+guard: −1.03 / +10.25, −0.36 / 0.00, −0.99 / +9.13.

**Verdict against the §22 criteria:**
- **(P1) SUPPORTED.** Plain early stopping does not bound risk: worst case +176% (Bolt covid_deaths), no worse on only 64/108, against 106/108 for A.
- **(P2) EQUIVALENT,** as defined in advance.
  - Mean difference A − ES+guard = −0.41 pp, which is below the 0.5 pp equivalence margin.
  - Holm-adjusted one-sided p = 0.42 (A better) and 0.85 (ES+guard better).
  - Paired: A better on 12 runs, worse on 9, tied (±0.1 pp) on 87.
- **(P3) NOT SUPPORTED.** Joint vs A: −0.06 pp, p = 0.42, and a worse tail (+9.13 vs +6.87). Searching over steps as well as α adds nothing.

**What this means:**
1. **The validation guard, not the α axis by itself, is what removes the tail.** Given the same guard on the same validation windows, choosing how far to move along the fine-tuning path by α (interpolation) or by step (early stopping) gives statistically equivalent average results. The paper must say this plainly.
2. **Where they differ is the tail, not the mean.** Runs worse than zero-shot by more than 1%:
   - A: 2 (both on Bolt);
   - ES+guard: 3, including Chronos-2 ett2/W at +10.3%, where A keeps 0%;
   - Joint: 4.
   Interpolation also needs only the final weights, not stored checkpoints. These are the honest arguments for A, but they are secondary, not a significance claim.
3. **Neither guard fixes every failure.**
   - On Bolt covid_deaths, ES+guard stays at 0% while A gives +4.85%.
   - On Bolt kdd_cup_2018_with_missing/D, all guarded methods are worse than zero-shot (A +6.87%, the others +9.13%). Validation and test disagree there whichever axis is searched.
4. **Calibration:** at the selected point, test 80% coverage is 0.782 (A), 0.777 (ES+guard), 0.776 (Joint) and 0.713 (plain ES).
5. **Compute:** the extra checkpoint evaluations cost 11.3 GPU-h across all runs, against 5.7 h for the final α-curve.

**Consequences for the paper:**
- Reframe the contribution as **validation-guarded selection along the fine-tuning path**, with weight interpolation as the recommended instance. It matches guarded early stopping on average, has the best tail of the methods tested, and needs no checkpoints.
- Report P2 as equivalence, not as a win.
- No change to the primary method (rule A), as pre-registered.

