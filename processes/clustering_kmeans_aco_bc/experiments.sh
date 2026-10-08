#!/usr/bin/env bash
set -euo pipefail

# Experiment configuration
PYTHON="${PYTHON:-python}"
export MLFLOW_URI="${MLFLOW_URI-http://127.0.0.1:8081}"
SUVR_DATA="20251130_suvr.parquet"
CONN_MATRIX="connectivity_matrix_lambda_0.06.parquet"
R_MATRIX="20251130_R_matrix.npy"
EXPERIMENT_PREFIX="kmeans-aco-bc-lambda-0.06"
COHORTS=(global lv_ppa nf_ppa se_ppa)
GLOBAL_CLUSTERS=(2 3 4 5 6)
VARIANT_CLUSTERS=(2 3 4)
N_NODES=(10 15 20 25)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd -- "$REPO_ROOT"

for cohort in "${COHORTS[@]}"; do
    if [[ "$cohort" == global ]]; then
        clusters=("${GLOBAL_CLUSTERS[@]}")
        groups=()
    else
        clusters=("${VARIANT_CLUSTERS[@]}")
        groups=(--groups controls controls-adni "$cohort")
    fi

    for n_clusters in "${clusters[@]}"; do
        for n_nodes in "${N_NODES[@]}"; do
            command=(
                "$PYTHON" "$SCRIPT_DIR/main.py"
                --suvr_data "$SUVR_DATA"
                --conn_matrix "$CONN_MATRIX"
                --r_matrix "$R_MATRIX"
                --mlflow_experiment_name "$EXPERIMENT_PREFIX-$cohort"
                --mlflow_run_name "c$n_clusters-n$n_nodes"
                --n_clusters "$n_clusters"
                --n_nodes "$n_nodes"
            )
            "${command[@]}" "${groups[@]}"
        done
    done
done
