"""AAL atlas loading and regional image mapping."""

import nibabel as nib
import numpy as np
from nilearn import image
from nilearn.datasets import fetch_atlas_aal


def load_aal_nii() -> tuple:
    """Load the SPM12 AAL atlas as a three-dimensional NIfTI image."""
    atlas = fetch_atlas_aal(version="SPM12")
    atlas_nii = nib.load(atlas["maps"])

    if len(atlas_nii.shape) > 3:
        if len(atlas_nii.shape) == 4 and atlas_nii.shape[-1] == 1:
            atlas_nii = image.index_img(atlas_nii, 0)
        else:
            raise ValueError(f"Unhandled image dimensions: {list(atlas_nii.shape)!r}")

    return atlas_nii


def create_aal_template(roi_data: dict, default_val: float = 0.0):
    """Map values keyed by lowercase AAL region names to a NIfTI image."""
    assert isinstance(roi_data, dict)

    aal_atlas = fetch_atlas_aal()
    aal = image.load_img(aal_atlas.maps)
    aal_data = np.array(aal.get_fdata())
    aal_labels = [label.lower() for label in aal_atlas["labels"]]
    aal_coords = [float(v) for v in aal_atlas["indices"]]
    region_id = dict(zip(aal_labels, aal_coords))
    region_coords = {
        region: np.where(aal_data == index) for region, index in region_id.items()
    }
    aal_template = np.zeros(shape=aal_data.shape)

    for roi, coords in region_coords.items():
        if roi in roi_data:
            aal_template[coords] = roi_data[roi]
        else:
            aal_template[coords] = default_val

    nii_obj = image.new_img_like(
        data=aal_template,
        ref_niimg=image.load_img(aal_atlas.maps),
    )
    return nii_obj
