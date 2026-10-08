# SUVR extraction

Extract regional standardized uptake value ratios (SUVRs) from spatially normalized FDG-PET scans. Each scan is divided by the median uptake within the cerebellar reference mask, then averaged within each SPM12 AAL region. NIfTI files (`.nii` and `.nii.gz`) are read recursively.

## Configuration and execution

Edit the configuration block in [`main.py`](main.py):

| Parameter | Description |
| --- | --- |
| `reference_cerebellum_mask` | Cerebellar reference mask. |
| `aal_coding` | ROI labels, defaulting to `data/metadata/aal_roi_coding.json`. |
| `input_directory_hcsc`, `input_directory_adni` | Input scan directories. |
| `hcsc_fwhm`, `adni_fwhm` | Gaussian smoothing FWHM in mm; `None` and `8`, respectively. |
| `output_directory` | Output folder, defaulting to `data/processed/suvr`. |


From the repository root, with the Python environment activated:

```bash
python processes/extract_suvrs/main.py
```

## Output

Writes `YYYYMMDD_suvr.parquet` to the output folder, which is created automatically. The date is the execution date.

The table has an `id`/`file` index, one `<region>_mean_suvr` column per AAL region, and a `group` column containing `se_ppa`, `nf_ppa`, `lv_ppa`, `controls`, or `controls-adni`. Configure this filename as the input to the normative and individual connectivity processes.
