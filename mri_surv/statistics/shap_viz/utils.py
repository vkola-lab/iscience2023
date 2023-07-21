from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import nibabel as nib
import os

from statistics.mlp_output_wrappers import name_to_lobe_map, load_roiname_to_roiid_map

"""
The following colormap copied from
https://matplotlib.org/2.0.2/examples/pylab_examples/custom_cmap.html
"""

__all__ = [
    "cmap_transparent",
    "rescale",
    "BASEDIR",
    "sig",
    "load_input",
    "load_rid_and_cluster",
    "dump_series",
]

BASEDIR = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test"


def dump_series() -> None:
    """
    Generates 3 files to use for decoding:
    1. index_to_lobe_map is a map of ID #s for each lobe
    2. id_to_region_map is a map of ID #s to regions
    3. name_to_lobe_map is a map of region names to lobes

    """
    region_to_id = load_roiname_to_roiid_map()  # name to ID #
    name_to_lobe = name_to_lobe_map()  # name to lobe
    c = {
        z: name_to_lobe[name]  # map ID # to lobe
        for name, id_ in region_to_id.items()
        for z in id_
        if name in name_to_lobe.keys()
    }
    se = pd.Series(c, name="Lobe")
    se.to_csv("metadata/data_processed/index_to_lobe_map.csv", index_label="ID")
    id_to_region = {z: x for x, y in region_to_id.items() for z in y}
    se = pd.Series(id_to_region, name="RegionName")
    se.to_csv("metadata/data_processed/id_to_region_map.csv", index_label="ID")
    se = pd.Series(name_to_lobe, name="Lobe")
    se.to_csv("metadata/data_processed/name_to_lobe_map.csv", index_label="RegionName")


def load_input(rid: str) -> np.ndarray:
    data = (
        nib.load(os.path.join(BASEDIR, f"masked_brain_mri_{rid}.nii"))
        .get_fdata()
        .astype(np.float32)
    )
    data[data != data] = 0
    data = rescale(data, (0, 2.5))
    return data


def load_rid_and_cluster() -> pd.DataFrame:
    """

    Returns:
        pd.DataFrame: labels from RID > risk subgroup
    """
    return pd.read_csv("./metadata/data_processed/nacc_cluster_labels.csv").set_index(
        "RID", drop=True
    )


cdict3 = {
    "red": (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.8, 1.0),
        (0.75, 1.0, 1.0),
        (1.0, 0.4, 1.0),
    ),
    "green": (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.9, 0.9),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    "blue": (
        (0.0, 0.0, 0.4),
        (0.25, 1.0, 1.0),
        (0.5, 1.0, 0.8),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
}

cdict3["alpha"] = ((0.0, 1.0, 1.0), (0.5, 0.6, 0.6), (1.0, 1.0, 1.0))

cmap_transparent = LinearSegmentedColormap("transp_map", cdict3)


def sig(x: float) -> str:
    """
    Returns significance for each p value range

    Args:
        x (float): p-value

    Returns:
        str: significance stars
    """
    if x < 0.001:
        return "***"
    elif x < 0.01:
        return "**"
    elif x < 0.05:
        return "*"
    else:
        return ""


def rescale(array: np.ndarray, tup: tuple) -> np.ndarray:
    """
    rescales an array to range tup[0] to tup[1], per Xiao's rescaling feature

    Args:
        array (np.ndarray): nib ndarray
        tup (tuple): len(2) tuple with min and max to scale to

    Returns:
        np.ndarray: rescaled array
    """
    a = np.max(array) - np.min(array)
    t = abs(tup[0] - tup[1])
    return array * t / a
