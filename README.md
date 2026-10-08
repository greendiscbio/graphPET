# Graph-constrained unsupervised clustering of individual brain metabolic connectivity in primary progressive aphasia


Repository with the code associated with research article: 

Graph-constrained unsupervised clustering of individual brain metabolic connectivity in primary progressive aphasia 
<mark>TBD. Add link</mark>.

## Repository notes

We're still working on improving the organization and documentation of this repository. As the project evolved, so did the analyses and the scripts behind them, and some parts of the code are not as clean or well documented as we would have liked. We're gradually working through these details to make everything easier to follow and reuse.

We're doing our best to improve things, and we really appreciate your patience while we continue polishing everything! 🙏

Nevertheless, the repository includes all the scripts needed to reproduce the main analyses carried out using the proposed framework, provided that users have access to the required data and software.

The neuroimaging, clinical, and neuropsychological data used in this study were obtained from a private cohort at the Hospital Clinico San Carlos and the 🔗 [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/). The original datasets from ADNI are not included in this repository, nor do we provide access to them.

Researchers wishing to work with ADNI data must request access directly through ADNI and comply with its data-sharing and publication policies. Further information on access procedures, usage conditions, and citation requirements can be found on the [ADNI data access page](https://adni.loni.usc.edu/data-samples/access-data/).

This repository contains only the scripts developed for data processing and analysis. Any steps involving ADNI data assume that the user has obtained the necessary permissions, and compliance with ADNI's data-use agreements remains the user's responsibility.

Finally, some parts of the processing pipeline, particularly those involving neuroimaging, depend on third-party software that is proprietary or requires a separate license. These tools are not distributed with the repository, and users should ensure they have the appropriate licenses before running the corresponding scripts.

## Processes

| Step | Process | Description |
| --- | --- | --- |
| 1 | [SUVR extraction](processes/extract_suvrs/README.md) | Normalize PET uptake by the median cerebellar uptake and calculate mean SUVRs for AAL regions. |
| 2 | [Normative connectivity](processes/graphical_lasso/README.md) | Estimate binary connectivity graphs from healthy controls using graphical lasso and evaluate regularization parameters. |
| 3 | [Individual connectivity](processes/individual_conn_matrix/README.md) | Calculate dense matrices of control-referenced metabolic co-abnormality for each subject. |
| 4a | [GA-ACO-BC clustering](processes/clustering_ga_aco_bc/README.md) | Optimize subject partitions with a genetic algorithm and ACS selection of connected subnetworks. |
| 4b | [KMeans-ACO-BC clustering](processes/clustering_kmeans_aco_bc/README.md) | Select connected subnetworks with ACS to maximize K-Means silhouette scores. |

Steps 4a and 4b are alternative clustering strategies using the outputs of the preparation steps.

## Environment setup

Run all commands from the repository root.

### Python

Use Python 3.11. The Pipfile requires a local checkout of [`Ant-Colony-Optimization-CSS`](https://github.com/FernandoGaGu/Ant-Colony-Optimization-CSS) at `../Ant-Colony-Optimization-CSS`. Install the dependencies and local utilities:

```bash
mkdir -p .venv
pipenv install
pipenv shell   # activate the environment
```

### MLflow

Clustering runs store metrics and artifacts in MLflow. To use a local tracking server, start it in a separate terminal with the Python environment activated:

```bash
mlflow server --host 127.0.0.1 --port 8081
```

In the execution terminal, set the tracking URI:

```bash
export MLFLOW_URI=http://127.0.0.1:8081
```

The batch launchers default to this address and respect an existing `MLFLOW_URI`. Set `MLFLOW_URI=""` to use MLflow's default local store without a server.

### R

Install the packages required by normative connectivity estimation:

```bash
Rscript -e 'install.packages(c("glasso", "igraph", "Matrix", "corrplot", "ggplot2", "dplyr", "reshape2", "arrow"), repos = "https://cloud.r-project.org")'
```

## Execution

Edit the preparation scripts' configuration blocks and update the input filenames after each process completes.

```bash
python processes/extract_suvrs/main.py
Rscript processes/graphical_lasso/main.R
mkdir -p data/processed/individual_conn_matrix
python processes/individual_conn_matrix/main.py
```

Choose a clustering strategy and edit its `experiments.sh` configuration block before running its batch:

```bash
bash processes/clustering_ga_aco_bc/experiments.sh
# Or:
bash processes/clustering_kmeans_aco_bc/experiments.sh
```

Each batch covers global and variant-specific analyses across cluster counts and subnetwork sizes. The process READMEs also describe individual runs.


