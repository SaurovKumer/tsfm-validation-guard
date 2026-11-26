# Executed Kaggle notebooks

Each folder holds one executed notebook (`<name>.ipynb`, with the exact code that ran and its printed log) and the CSV files it wrote. The GIFT-Eval data are not included; every notebook downloads only the configurations it needs from the Hugging Face hub. Session ids identify the Kaggle runs.

| Notebook | Session id | Output folder | Copy used by the analysis |
|---|---|---|---|
| tsfm-phase1-parta | 371310344 | results_A | results/partA |
| tsfm-phase1-partb | 371310737 | results_B | results/partB |
| tsfm-confirm-parta | 371426472 | results_confirm_A | results/confirmA |
| tsfm-confirm-partb | 371426887 | results_confirm_B | results/confirmB |
| tsfm-holdout-h1 | 371608404 | results_holdout_H1 | results/holdoutH1 |
| tsfm-holdout-h2 | 371608646 | results_holdout_H2 | results/holdoutH2 |
| notebook4d1058bb0a (fresh F1) | 371639766 | results_fresh_F1 | results/freshF1 |
| tsfm-fresh-f2 | 371640165 | results_fresh_F2 | results/freshF2 |
| tsfm-timesfm-quick | 371693607 | results_tfm_quick | results/tfm_quick |
| tsfm-timesfm-a | 371723253 | results_tfm_A | results/tfm_A |
| tsfm-timesfm-bh | 372213649 | results_tfm_BH1H2 | results/tfm_BH |
| tsfm-timesfm-f | 372214923 | results_tfm_F1F2 | results/tfm_F |
| tsfm-chronos-n | 372377306 | results_newcfg_N | results/chr_N |
| tsfm-timesfm-n | 372377642 | results_tfm_N | results/tfm_N |
| tsfm-chronos-s | 372450635 | results_stab_S | results/chr_S |
| tsfm-timesfm-s | 372450212 | results_tfm_S | results/tfm_S |
| tsfm-chronos-c1 | 372569977 | results_cover_C1 | results/chr_C1 |
| tsfm-chronos-c2 | 372569109 | results_cover_C2 | results/chr_C2 |
| tsfm-bolt-quick | 372807389 | results_bolt_quick | results/bolt_quick |
| tsfm-bolt-b1 | 372813455 | results_bolt_B1 | results/bolt_B1 |
| tsfm-bolt-b2 | 372814034 | results_bolt_B2 | results/bolt_B2 |
| tsfm-timesfm-c1 | 373128035 | results_tfm_C1 | results/tfm_C1 |
| tsfm-timesfm-c2 | 373065735 | results_tfm_C2 | results/tfm_C2 |
| tsfm-chronos-e1 | 373283801 | results_es_E1 | results/chr_E1 |
| tsfm-chronos-e2 | 373283641 | results_es_E2 | results/chr_E2 |
| tsfm-bolt-e3 (version 2) | 373314636 | results_bolt_E3 | results/bolt_E3 |
| tsfm-bolt-e1 | 373536067 | results_bolt_E1 | results/bolt_E1 |
| tsfm-bolt-e2 | 373362162 | results_bolt_E2 | results/bolt_E2 |
| tsfm-chronos-e3 | 373543257 | results_es_E3 | results/chr_E3 |
| tsfm-timesfm-e1 | 373542345 | results_tfm_E1 | results/tfm_E1 |
| tsfm-timesfm-e3 (version 2) | 373524809 | results_tfm_E3 | results/tfm_E3 |
| tsfm-timesfm-e2 (version 8) | 374196418 | results_tfm_E2 | results/tfm_E2 |

Notes:
- `notebook4d1058bb0a` is the fresh-source part F1 (the notebook kept its default Kaggle name).
- `tsfm-bolt-e3` and `tsfm-timesfm-e3` are version 2 of their notebook, `tsfm-timesfm-e2` is version 8; the other versions of these notebooks were cancelled starts or, for timesfm-E2 version 2, a session without internet access that failed before running any code of the study.
