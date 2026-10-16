# MNGMOD

**Multi-scale Noisy Granular-Ball Manifold Outlier Detection**

## Running the Code

Run `main_2_K_copy_runs.py` from the project root directory:

```bash
python main_2_K_copy_runs.py
```

The script runs MNGMOD on the `.mat` datasets in the `mydata_4` directory. Each dataset should contain a variable named `trandata`, with features in the first columns and the ground-truth label in the last column (`0` for normal samples and `1` for outliers).

## Parameter Settings

The following parameter ranges are used:

| Parameter | Search range | Step |
| --------- | ------------ | ---- |
| `k1`      | 0.05–1.00    | 0.05 |
| `k2`      | 0.05–1.00    | 0.05 |

A two-stage parameter search is performed. First, `k2` is fixed at `1.0` to select `k1`. Then, the selected `k1` is fixed to select `k2`. Parameters are selected based on the mean AUC across repeated runs.

Each evaluated parameter combination is run **10 times in parallel**, using random seeds **42–51**. Metrics are calculated separately for each run and then summarized by their mean and standard deviation.

Results are saved as Excel and PKL files in `myresult/MNGMOD`. The PKL files retain the anomaly score sequence for every parameter combination and random seed.
