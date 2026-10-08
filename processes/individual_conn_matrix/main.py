"""Compute dense individual metabolic co-abnormality matrices."""

from datetime import datetime

import numpy as np

import utils

# Configuration
suvr_file = "20251130_suvr.parquet"
output_directory = utils.variables.INDV_CONN_MATRIX_FOLDER


def main():
    """Export control-referenced pairwise weights and their region labels."""
    data, suvr, groups = utils.io.load_suvr_data(suvr_file, standarize=False)
    control_mask = (groups == "controls") | (groups == "controls-adni")

    suvr_cols = [c for c in data.columns if c.endswith("suvr")]
    region_mask = np.array([not ("cereb" in c or "vermis" in c) for c in suvr_cols])
    suvr_cols = np.array(suvr_cols)[region_mask]
    suvr = suvr[:, region_mask]

    # Use control population standard deviations (ddof=0).
    suvr_norm_zscores = (
        suvr - suvr[control_mask].mean(axis=0, keepdims=True)
    ) / suvr[control_mask].std(axis=0, keepdims=True)

    n_subjects = suvr_norm_zscores.shape[0]
    n_rois = suvr_norm_zscores.shape[1]
    r_matrix = np.empty(shape=(n_subjects, n_rois, n_rois), dtype=float)

    for i in range(n_subjects):
        subject_zscores = suvr_norm_zscores[i].reshape(-1, 1)
        r_matrix[i, ...] = np.abs(subject_zscores + subject_zscores.T)

    curr_date = datetime.now().strftime("%Y%m%d")
    np.save(output_directory / f"{curr_date}_R_matrix.npy", r_matrix)

    with open(output_directory / f"{curr_date}_R_matrix_header.txt", "w") as f:
        f.write("\t".join(suvr_cols.tolist()))

    print(f"Data saved in {output_directory}")


if __name__ == "__main__":
    main()
