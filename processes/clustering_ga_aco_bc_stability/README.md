# GA-ACO-BC subject subsampling

Repeat [GA-ACO-BC clustering](../clustering_ga_aco_bc/README.md) on random subject subsamples. Each run selects subjects without replacement and optimizes a partition and a connected subnetwork using the same GA and ACS utilities as the full-sample process. The normative graph and individual connectivity inputs remain fixed across repetitions.

## Requirements

Use the [Python environment](../../README.md#python), including the local `antco-css` dependency. Inputs are:

- A SUVR Parquet file under `data/processed/suvr`.
- A numeric adjacency Parquet file under `data/processed/graphical_lasso`, with rows and columns in connectivity tensor order.
- A dense `(subjects, regions, regions)` NumPy tensor under `data/processed/individual_conn_matrix` and its matching `_header.txt` file.

## Execution

Run from the repository root with the Python environment activated. Set `MLFLOW_URI` to use a tracking server; 
```bash
python processes/clustering_ga_aco_bc_stability/main.py \
    --suvr_data 20251130_suvr.parquet \
    --conn_matrix connectivity_matrix_lambda_0.06.parquet \
    --r_matrix 20251130_R_matrix.npy \
    --mlflow_experiment_name ga-stability-c5-n25-v2 \
    --mlflow_run_name seed2000 \
    --n_clusters 5 --n_nodes 25 --sel_prop 0.85 \
    --population_size 80 --generations 150 --n_jobs 20 \
    --seed 2000
```

| Option | Description |
| --- | --- |
| `--suvr_data`, `--conn_matrix`, `--r_matrix` | Required filenames within the input folders. |
| `--mlflow_experiment_name`, `--mlflow_run_name` | Required tracking names. |
| `--n_clusters`, `--n_nodes` | Required cluster count and one or more subnetwork sizes. |
| `--sel_prop` | Required sample proportion, greater than 0 and at most 1. |
| `--population_init` | `kmeans` (default) or `random`. |
| `--population_size`, `--generations` | Both default to 100 for individual runs. |
| `--n_jobs` | Evaluation workers; default 1. |
| `--min_samples_per_cluster` | Minimum cluster size; default 5. |
| `--seed` | Subject-selection and K-Means-informed initialization seed; default 2000. |



## Experiment batch

Edit the configuration block in [`v2_c5_n25_lambda_0.06.sh`](v2_c5_n25_lambda_0.06.sh), then run:

```bash
bash processes/clustering_ga_aco_bc_stability/v2_c5_n25_lambda_0.06.sh
```
