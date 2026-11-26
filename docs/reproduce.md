# Reproducing the study

## 1. Analysis only (no GPU)

```bash
pip install -r requirements.txt
```

Every analysis script reads the CSV files in `results/` and writes its outputs to a sub-folder of `results/` (or to `figures/`). Run them from the repository root:

| Script | Purpose | Log section |
|---|---|---|
| `analysis/selection_rules.py` | first offline study of selection rules on the design runs | §4b |
| `analysis/confirm_analysis.py` | seeds 1–2 confirmation of the guard | §6 |
| `analysis/failure_diagnosis.py` | why the hold-out stage failed | §9 |
| `analysis/fresh_analysis.py` | fresh-source confirmation of guard-LF | §11 |
| `analysis/tfm_all_analysis.py`, `tfm_partA_analysis.py`, `tfm_val_vs_test.py` | TimesFM 2.5 stage | §13 |
| `analysis/rules_v2.py` | leave-one-dataset-out study of less conservative guards (rules A and B) | §14 |
| `analysis/confirm_N.py` | confirmation of rules A and B on 8 unseen configurations | §15 |
| `analysis/stats_tests.py` | statistical tests on the first 72 runs | §16 |
| `analysis/stability_S.py` | seeds and fine-tuning length | §18 |
| `analysis/cover_C.py` | interval coverage and conformal recalibration (Chronos-2, TimesFM) | §19 |
| `analysis/bolt_B.py` | third model (Chronos-Bolt), pre-registered criteria K1–K3 | §21 |
| `analysis/es_E.py` | interpolation vs early stopping, pre-registered comparisons P1–P3 | §22 |
| `analysis/stats108.py` | final statistics on all 108 runs (paper tables) | – |
| `analysis/paper_figures.py`, `method_figure.py`, `appendix_table.py` | paper figures and appendix table | – |

`analysis/tfm_save_part.py` is a helper that was used once to store early TimesFM results read from the Kaggle output viewer; it needs command-line arguments and is not part of the analysis pipeline.

## 2. Re-running the experiments

The experiments ran on Kaggle (one NVIDIA T4 or T4 × 2, internet on). Each notebook in `notebooks/` is the script in `scripts/` with the stage flags and `PART` already set, for example `tsfm_bolt_E2.ipynb` = `scripts/tsfm_bolt.py` with `ES = True` and `PART = "E2"`.

1. Create a Kaggle notebook, import the `.ipynb` file, set *Accelerator = GPU T4* and *Internet = On*.
2. Use *Save Version → Save & Run All (Commit)*. The notebook installs its dependencies, downloads only the GIFT-Eval configurations it needs and writes its CSV files to `/kaggle/working/results_*`.
3. Copy the CSV files to the matching `results/` folder (see `runs/README.md`) and re-run the analysis.

Each notebook takes about 2–10 GPU-hours. Fine-tuning uses seed 0 unless stated otherwise; re-running a configuration with the same seed reproduced the test loss and the selected α exactly in every check we made (`results/cover_C/reproducibility.csv`, `results/es_E/reproducibility.csv`).

### Local smoke test (CPU)

All three scripts have a smoke mode with tiny settings that checks the code path without a GPU:

```bash
TSFM_SMOKE=1 TSFM_WORK=./smoke python scripts/tsfm_bolt.py
TSFM_SMOKE=1 TSFM_ES=1 TSFM_WORK=./smoke python scripts/tsfm_timesfm.py   # also exercises the checkpoint code
```

The smoke mode needs `chronos-forecasting`, `timesfm`, `gluonts`, `datasets` and `huggingface_hub`; TimesFM runs with random weights, Chronos-Bolt uses the `tiny` checkpoint.

## 3. Software versions

The Kaggle runs used the Kaggle Python image of November 2026 (Python 3.13, PyTorch 2.x) with `chronos-forecasting` ≥ 2.1 and `timesfm[torch]` ≥ 2.5, installed by the first cell of each notebook. The analysis was run with Python 3.14, NumPy, pandas, SciPy 1.18 and Matplotlib.
