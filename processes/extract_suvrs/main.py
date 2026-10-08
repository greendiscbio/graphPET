"""Extract cerebellum-referenced mean SUVRs for AAL regions."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from tqdm import tqdm

import utils

# Configuration
aal_coding = utils.variables.DATA_FOLDER / "metadata" / "aal_roi_coding.json"
output_directory = utils.variables.SUVR_FOLDER
reference_cerebellum_mask = utils.variables.DATA_FOLDER / 'processed' / 'brain_templates' / '20240615_cerebellum_FSLfnirt_MNI152_voxel1mm_x181_y217_z181_th25.nii'
input_directory_hcsc = utils.variables.DATA_FOLDER / "raw" / "nifti" / "hcsc"
input_directory_adni = utils.variables.DATA_FOLDER / "raw" / "nifti" / "adni" / "controls"
hcsc_fwhm = None
adni_fwhm = 8


def extract_suvrs_from_path(
    path: Path,
    reference_roi_file: Path,
    aal_coding: Path,
    fwhm: int = None,
) -> pd.DataFrame:
    """Read NIfTI scans recursively and aggregate normalized uptake by AAL region."""

    def __worker__(
        file: Path,
        ref_roi_nii: nib.nifti1.Nifti1Image,
        aal_nii: nib.nifti1.Nifti1Image,
        aal_coding_hash: dict,
        unique_aal_indices: np.ndarray,
        fwhm: int,
    ) -> dict:
        try:
            ref_roi_nii = deepcopy(ref_roi_nii)
            aal_nii = deepcopy(aal_nii)
            nii_obj = nib.load(file)

            if fwhm is not None:
                nii_obj = image.smooth_img(nii_obj, fwhm=fwhm)

            # Preserve discrete mask and atlas labels when resampling.
            r_ref_roi_nii = image.resample_to_img(
                ref_roi_nii,
                nii_obj,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
            r_aal_nii = image.resample_to_img(
                aal_nii,
                nii_obj,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            pet_data = nii_obj.get_fdata()
            ref_data = r_ref_roi_nii.get_fdata().astype(int)
            aal_data = r_aal_nii.get_fdata().astype(int)

            assert not np.any(np.isnan(pet_data)), f"PET data contains NaNs ({file})"
            assert not np.any(np.isnan(ref_data)), "Reference ROI contains NaNs"
            assert not np.any(np.isnan(aal_data)), "AAL contains NaNs"

            reference_suvr = np.median(pet_data[ref_data == 1])
            suvr_data = pet_data / reference_suvr

            extracted_data = {}
            for index in unique_aal_indices:
                roi_label = f"{aal_coding_hash.get(str(index), index)}_mean_suvr"
                if roi_label in extracted_data:
                    raise ValueError(f"Overwriting information for {roi_label}")

                region_mask = aal_data == index
                extracted_data[roi_label] = float(suvr_data[region_mask].mean())

            extracted_data["id"] = file.name.replace(".nii.gz", "").replace(".nii", "")
            extracted_data["file"] = str(file.resolve())
            return extracted_data
        except Exception as ex:
            print(f"Exception in file: {file.resolve()}. {type(ex)}: {ex}")
            return None

    nii_files = [
        f for f in path.rglob("*") if f.name.endswith(".nii") or f.name.endswith(".nii.gz")
    ]
    ref_roi_nii = nib.load(reference_roi_file)
    aal_nii = utils.neuroimaging.load_aal_nii()
    aal_coding_hash = utils.io.load_json(aal_coding)

    unique_aal_indices = np.unique(aal_nii.get_fdata())
    unique_aal_indices = unique_aal_indices[~np.isclose(unique_aal_indices, 0.0)].astype(int)

    suvr_df = [
        __worker__(
            file=file,
            ref_roi_nii=ref_roi_nii,
            aal_nii=aal_nii,
            aal_coding_hash=aal_coding_hash,
            unique_aal_indices=unique_aal_indices,
            fwhm=fwhm,
        )
        for file in tqdm(nii_files, desc="Extracting SUVR data...")
    ]
    suvr_df = pd.DataFrame([v for v in suvr_df if v is not None]).set_index(["id", "file"])

    print(f"Generated dataframe shape: {suvr_df.shape}")
    print(suvr_df.head(5))
    print("\n\n")
    return suvr_df


if __name__ == "__main__":
    se_ppa_df = extract_suvrs_from_path(
        input_directory_hcsc / "se_ppa", reference_cerebellum_mask, aal_coding, hcsc_fwhm
    )
    nf_ppa_df = extract_suvrs_from_path(
        input_directory_hcsc / "nf_ppa", reference_cerebellum_mask, aal_coding, hcsc_fwhm
    )
    lv_ppa_df = extract_suvrs_from_path(
        input_directory_hcsc / "lv_ppa", reference_cerebellum_mask, aal_coding, hcsc_fwhm
    )
    controls_df = extract_suvrs_from_path(
        input_directory_hcsc / "controls", reference_cerebellum_mask, aal_coding, hcsc_fwhm
    )
    controls_adni_df = extract_suvrs_from_path(
        input_directory_adni, reference_cerebellum_mask, aal_coding, adni_fwhm
    )

    se_ppa_df["group"] = "se_ppa"
    nf_ppa_df["group"] = "nf_ppa"
    lv_ppa_df["group"] = "lv_ppa"
    controls_df["group"] = "controls"
    controls_adni_df["group"] = "controls-adni"
    merged_df = pd.concat(
        [se_ppa_df, nf_ppa_df, lv_ppa_df, controls_df, controls_adni_df], axis=0
    )

    curr_date = datetime.now().strftime("%Y%m%d")
    output_directory.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_directory / f"{curr_date}_suvr.parquet")
