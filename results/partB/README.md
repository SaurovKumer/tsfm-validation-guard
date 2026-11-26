Part B of the Phase-1 run (Kaggle notebook tsfm-phase1-partB, version 1, finished 2026-11-19, 16793 s).
Chronos-2, 4000 steps, seed 0, FT context 1024, V=3 validation windows (+wise) and 1 window (+wise1).
Datasets: bizitobs_l2c/H, ett1/D, ett2/D, M_DENSE/D, LOOP_SEATTLE/D, jena_weather/D.

summary_normalised.csv: mean over the 6 datasets of each metric divided by Seasonal Naive (rounded to 4 d.p.).
per_dataset_change_vs_zero_shot.csv: % change of normalised MWQL vs zero-shot (negative = better than zero-shot).
Both were read from the Kaggle output viewer. The full results.csv (66 rows) and alpha_curves.csv (108 rows) are only
partly visible in the viewer (first 50 rows), so they are not copied here yet.
