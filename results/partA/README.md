Part A of the Phase-1 run (Kaggle notebook tsfm-phase1-partA, version 1, finished 2026-11-19, 18633 s).
Chronos-2, 4000 steps, seed 0, FT context 1024, V=3 validation windows (+wise) and 1 window (+wise1).
Datasets: us_births/D, saugeenday/D, ett1/W, ett2/W, solar/W, hierarchical_sales/W. (solar/W used V=2 because the series are short.)

summary_normalised.csv: mean over the 6 datasets of each metric divided by Seasonal Naive (rounded to 4 d.p.).
per_dataset_change_vs_zero_shot.csv: % change of normalised MWQL vs zero-shot (negative = better than zero-shot).
Both were read from the Kaggle output viewer. The full results.csv (66 rows) and alpha_curves.csv (108 rows) are only
partly visible in the viewer (first 50 rows), so they are not copied here yet.
