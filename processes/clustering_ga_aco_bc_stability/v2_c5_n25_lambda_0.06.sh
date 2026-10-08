#!/usr/bin/env bash
set -euo pipefail

# Experiment configuration
PYTHON="${PYTHON:-python}"
export MLFLOW_URI="${MLFLOW_URI-http://127.0.0.1:8081}"
SUVR_DATA="20251130_suvr.parquet"
CONN_MATRIX="connectivity_matrix_lambda_0.06.parquet"
R_MATRIX="20251130_R_matrix.npy"
SEL_PROP=0.85
NUM_CLUSTERS=5
NUM_NODES=25
NUM_JOBS=20
GENERATIONS=150
POPULATION_SIZE=80
POPULATION_INIT="kmeans"
MIN_SAMPLES_PER_CLUSTER=5
FIRST_SEED=2000
LAST_SEED=2099

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd -- "$REPO_ROOT"

for ((seed = FIRST_SEED; seed <= LAST_SEED; seed++)); do
    command=(
        "$PYTHON" "$SCRIPT_DIR/main.py"
        --suvr_data "$SUVR_DATA"
        --conn_matrix "$CONN_MATRIX"
        --r_matrix "$R_MATRIX"
        --mlflow_experiment_name "ga-stability-c$NUM_CLUSTERS-n$NUM_NODES-v2"
        --mlflow_run_name "seed$seed"
        --n_clusters "$NUM_CLUSTERS"
        --sel_prop "$SEL_PROP"
        --population_init "$POPULATION_INIT"
        --population_size "$POPULATION_SIZE"
        --generations "$GENERATIONS"
        --n_jobs "$NUM_JOBS"
        --min_samples_per_cluster "$MIN_SAMPLES_PER_CLUSTER"
        --n_nodes "$NUM_NODES"
        --seed "$seed"
    )
    "${command[@]}"
done
