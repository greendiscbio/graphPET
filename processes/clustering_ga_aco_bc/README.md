# GA-ACO-BC clustering

## Requirements

Use the [Python environment](../../README.md#python), including the local `antco-css` dependency. Inputs are:

- A SUVR Parquet file under `data/processed/suvr`.
- A numeric adjacency Parquet file under `data/processed/graphical_lasso`, with rows and columns in connectivity tensor order.
- A dense `(subjects, regions, regions)` NumPy tensor under `data/processed/individual_conn_matrix` and its matching `_header.txt` file.

The SUVR table and connectivity tensor must have the same subject order. Region indices must also match the SUVR columns used for ROI projections. 

## Execution

Run from the repository root with the Python environment activated. Set `MLFLOW_URI` to use a tracking server; otherwise the Python entrypoint uses MLflow's default location. 

```bash
python processes/clustering_ga_aco_bc/main.py \
    --suvr_data 20251130_suvr.parquet \
    --conn_matrix connectivity_matrix_lambda_0.06.parquet \
    --r_matrix 20251130_R_matrix.npy \
    --mlflow_experiment_name ga-aco-bc-lambda-0.06-global \
    --mlflow_run_name c5-n25 \
    --n_clusters 5 --n_nodes 25 \
    --population_size 80 --generations 150 --n_jobs 25
```

| Option | Description |
| --- | --- |
| `--suvr_data`, `--conn_matrix`, `--r_matrix` | Required filenames within the input folders. |
| `--mlflow_experiment_name`, `--mlflow_run_name` | Required tracking names. |
| `--n_clusters`, `--n_nodes` | Required cluster count and one or more subnetwork sizes. |
| `--groups` | Optional cohort labels; omitted to include all subjects. |
| `--population_init` | `kmeans` (default) or `random`. |
| `--population_size`, `--generations` | Both default to 100 for individual runs. |
| `--n_jobs` | Evaluation workers; default 1. |
| `--min_samples_per_cluster` | Minimum cluster size; default 5. |
| `--seed` | K-Means-informed initialization seed; default 2000. |

For variant-specific clustering, add `--groups controls controls-adni nf_ppa`, replacing `nf_ppa` with `lv_ppa` or `se_ppa` as needed. The control labels are combined after selection. K-Means-informed initialization uses PCA with 25 components.

## Experiment batch

Edit the configuration block in [`experiments.sh`](experiments.sh), then run:

```bash
bash processes/clustering_ga_aco_bc/experiments.sh
```
