import os

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting
from typing import Dict
from tqdm import tqdm

from statistics.mlp_output_wrappers import name_to_lobe_map
from statistics.shap_viz.utils import load_input, load_rid_and_cluster, cmap_transparent


__all__ = [
    "ShapBrainAvg",
    "ShapBrainClusterAvg",
    "ShapBrainAvgAbs",
    "ShapBrainClusterAvgAbs",
    "average_over_folds_and_time",
    "average_abs_over_folds_and_time",
    "load_masked_shap_rid_mn",
    "mask_brain_all_lobe_mn",
    "mask_brain_all_lobe_mn_abs",
]

plt.register_cmap(cmap=cmap_transparent)


class ShapBrainAvg:

    _path = "/data2/MRI_PET_DATA/shap"  # make path here
    _new_path = os.path.join(_path, "..", "shap_averaged")

    def __init__(self, rid: str):
        """
        __init__

        load 1 brain for 1 person

        take average

        Parameters
        ----------
        rid : str, rid for shap person
        """
        self.rid = rid
        self._load_and_average_brain()
        self.mri = load_input(rid)

    def _load_and_average_brain(self) -> None:
        self.affine = []
        self.original_brain = []
        for fold in range(5):
            for bin_ in (24, 48, 108):
                path = os.path.join(self._path, f"{fold}_{bin_}_{self.rid}.nii")
                img = nib.load(path)
                data = img.get_fdata()
                self.original_brain.append(
                    np.expand_dims(data, 3)
                )  # this gets the numeric data from the array
                self.affine.append(np.asarray(img.affine))
        assert all(
            [np.array_equal(x, y) for x, y in zip(self.affine[1:], self.affine[:-1])]
        )
        self.affine = self.affine[0]
        self.original_brain = np.mean(
            np.concatenate(self.original_brain, axis=3), axis=3, keepdims=False
        )
        self.img = nib.Nifti1Image(self.original_brain, self.affine)

    def save_brain(self) -> None:
        os.makedirs(self._new_path, exist_ok=True)
        img = self.img
        file_path = os.path.join(self._new_path, f"mean_shap_{self.rid}.nii")
        if os.path.exists(file_path):
            img_exist = nib.load(file_path)
            assert np.array_equal(img_exist.get_fdata(), self.original_brain)
            return
        nib.save(img, os.path.join(self._new_path, f"mean_shap_{self.rid}.nii"))

    @classmethod
    def load_brain(cls, rid: str) -> nib.Nifti1Image:
        return nib.load(os.path.join(cls._new_path, f"mean_shap_{rid}.nii"))

class ShapBrainAvgAbs:

    _path = "/data2/MRI_PET_DATA/shap"  # make path here
    _new_path = os.path.join(_path, "..", "shap_averaged_abs")

    def __init__(self, rid: str):
        """
        __init__

        load 1 brain for 1 person

        take average

        Parameters
        ----------
        rid : str, rid for shap person
        """
        self.rid = rid
        self._load_and_average_brain()
        self.mri = load_input(rid)

    def _load_and_average_brain(self) -> None:
        self.affine = []
        self.original_brain = []
        for fold in range(5):
            for bin_ in (24, 48, 108):
                path = os.path.join(self._path, f"{fold}_{bin_}_{self.rid}.nii")
                img = nib.load(path)
                data = img.get_fdata()
                self.original_brain.append(
                    np.expand_dims(data, 3)
                )  # this gets the numeric data from the array
                self.affine.append(np.asarray(img.affine))
        assert all(
            [np.array_equal(x, y) for x, y in zip(self.affine[1:], self.affine[:-1])]
        )
        self.affine = self.affine[0]
        self.original_brain = np.mean(
            np.abs(
                np.concatenate(self.original_brain, axis=3)), axis=3, keepdims=False
        )
        self.img = nib.Nifti1Image(self.original_brain, self.affine)

    def save_brain(self) -> None:
        os.makedirs(self._new_path, exist_ok=True)
        img = self.img
        file_path = os.path.join(self._new_path, f"mean_abs_shap_{self.rid}.nii")
        if os.path.exists(file_path):
            img_exist = nib.load(file_path)
            assert np.array_equal(img_exist.get_fdata(), self.original_brain)
            return
        nib.save(img, os.path.join(self._new_path, f"mean_abs_shap_{self.rid}.nii"))

    @classmethod
    def load_brain(cls, rid: str) -> nib.Nifti1Image:
        return nib.load(os.path.join(cls._new_path, f"mean_abs_shap_{rid}.nii"))


class ShapBrainClusterAvg:
    path = "/data2/MRI_PET_DATA/shap_averaged"

    def __init__(self, cluster: int):
        self.cluster = cluster
        rids = load_rid_and_cluster().reset_index().set_index("Cluster Idx")
        self.rids = rids.loc[cluster, "RID"].to_numpy()
        self.affine = None
        self.brain = None
        self.img = None

    def generate_cluster_avg(self) -> None:
        brains = []
        self.affine = []
        for rid in tqdm(self.rids):
            try:
                bs = ShapBrainAvg.load_brain(rid)
            except:
                print(f"Couldn't load brain {rid}, will try to generate")
                sba = ShapBrainAvg(rid)
                bs = sba.img
            dat = np.expand_dims(np.asarray(bs.get_fdata()), 3)
            brains.append(dat)
            self.affine.append(np.asarray(bs.affine))
        assert all(
            np.array_equal(x, y) for x, y in zip(self.affine[1:], self.affine[:-1])
        )
        self.affine = self.affine[0]
        self.brain = np.mean(np.concatenate(brains, axis=3), axis=3, keepdims=False)
        self.img = nib.Nifti1Image(self.brain, self.affine)

    def save_brain(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        img = self.img
        file_path = os.path.join(self.path, f"cluster_shap_{self.cluster}.nii")
        if os.path.exists(file_path):
            img_existing = nib.load(file_path)
            assert np.array_equal(img_existing.get_fdata(), self.brain)
            return
        nib.save(img, file_path)

    @classmethod
    def load_brain(cls, cluster: int):
        sbca = ShapBrainClusterAvg(cluster)
        sbca.img = nib.load(os.path.join(cls.path, f"cluster_shap_{cluster}.nii"))
        sbca.affine = np.asarray(sbca.img.affine)
        sbca.brain = np.asarray(sbca.img.get_fdata())
        return sbca

    def plot_brain_diff(self, title, sbca, cut_coords=None):
        img_path = "./figures"
        if cut_coords is None:
            cut_coords = []
            for coord in ("x", "y", "z"):
                cut_coords.append(
                    plotting.find_cut_slices(
                        self.img, direction=coord, n_cuts=1, spacing="auto"
                    )
                )
        f_name = os.path.join(img_path, f"shap_cnn_brain_{title}_diff.png")
        os.makedirs(img_path, exist_ok=True)
        f_data = np.asarray(self.img.get_fdata()) - np.asarray(sbca.img.get_fdata())
        f_data = (f_data - np.mean(f_data)) / np.std(f_data)
        img = nib.Nifti1Image(f_data, self.affine)
        bg_img = nib.load("./metadata/data_raw/NACC/masked_brain_mri_NACC356689.nii")
        plotting.plot_stat_map(
            img,  # could replace with self.mask_img
            cut_coords=cut_coords,
            colorbar=True,
            cmap=plt.cm.RdBu_r,
            bg_img=bg_img,
            annotate=False,
            title=title,
            draw_cross=False,
            black_bg=False,
            threshold=2,
            vmax=2,
        )
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.savefig(f_name[:-4] + ".svg", dpi=300, transparent=True)
        plt.close()
        return cut_coords

class ShapBrainClusterAvgAbs:
    path = "/data2/MRI_PET_DATA/shap_averaged_abs"

    def __init__(self, cluster: int):
        self.cluster = cluster
        rids = load_rid_and_cluster().reset_index().set_index("Cluster Idx")
        self.rids = rids.loc[cluster, "RID"].to_numpy()
        self.affine = None
        self.brain = None
        self.img = None

    def generate_cluster_avg(self) -> None:
        brains = []
        self.affine = []
        for rid in tqdm(self.rids):
            try:
                bs = ShapBrainAvgAbs.load_brain(rid)
            except:
                print(f"Couldn't load brain {rid}, will try to generate")
                sba = ShapBrainAvgAbs(rid)
                bs = sba.img
            dat = np.expand_dims(np.asarray(bs.get_fdata()), 3)
            brains.append(dat)
            self.affine.append(np.asarray(bs.affine))
        assert all(
            np.array_equal(x, y) for x, y in zip(self.affine[1:], self.affine[:-1])
        )
        self.affine = self.affine[0]
        self.brain = np.mean(np.abs(np.concatenate(brains, axis=3)), axis=3, keepdims=False)
        self.img = nib.Nifti1Image(self.brain, self.affine)

    def save_brain(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        img = self.img
        file_path = os.path.join(self.path, f"cluster_shap_{self.cluster}.nii")
        if os.path.exists(file_path):
            img_existing = nib.load(file_path)
            assert np.array_equal(img_existing.get_fdata(), self.brain)
            return
        nib.save(img, file_path)

    @classmethod
    def load_brain(cls, cluster: int):
        sbca = ShapBrainClusterAvgAbs(cluster)
        sbca.img = nib.load(os.path.join(cls.path, f"cluster_shap_{cluster}.nii"))
        sbca.affine = np.asarray(sbca.img.affine)
        sbca.brain = np.asarray(sbca.img.get_fdata())
        return sbca

    def plot_brain_diff(self, title, sbca, cut_coords=None):
        img_path = "./figures"
        if cut_coords is None:
            cut_coords = []
            for coord in ("x", "y", "z"):
                cut_coords.append(
                    plotting.find_cut_slices(
                        self.img, direction=coord, n_cuts=1, spacing="auto"
                    )
                )
        f_name = os.path.join(img_path, f"shap_cnn_brain_{title}_abs_diff.png")
        os.makedirs(img_path, exist_ok=True)
        f_data = np.asarray(self.img.get_fdata()) - np.asarray(sbca.img.get_fdata())
        f_data = (f_data - np.mean(f_data)) / np.std(f_data)
        img = nib.Nifti1Image(f_data, self.affine)
        bg_img = nib.load("./metadata/data_raw/NACC/masked_brain_mri_NACC356689.nii")
        plotting.plot_stat_map(
            img,  # could replace with self.mask_img
            cut_coords=cut_coords,
            colorbar=True,
            cmap=plt.cm.RdBu_r,
            bg_img=bg_img,
            annotate=False,
            title=title,
            draw_cross=False,
            black_bg=False,
            threshold=2,
            vmax=5,
        )
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.savefig(f_name[:-4] + ".svg", dpi=300, transparent=True)
        plt.close()
        return cut_coords

def average_over_folds_and_time() -> None:
    rids = load_rid_and_cluster()
    for rid in tqdm(rids.index):
        brain = ShapBrainAvg(rid)
        brain.save_brain()
    for clust in tqdm(range(4), desc="Cluster"):
        sbc = ShapBrainClusterAvg(clust)
        sbc.generate_cluster_avg()
        sbc.save_brain()

def average_abs_over_folds_and_time() -> None:
    rids = load_rid_and_cluster()
    for rid in tqdm(rids.index):
        brain = ShapBrainAvgAbs(rid)
        brain.save_brain()
    for clust in tqdm(range(4), desc="Cluster"):
        sbc = ShapBrainClusterAvgAbs(clust)
        sbc.generate_cluster_avg()
        sbc.save_brain()

def mask_brain_rid_lobe_mn(rid: int) -> pd.DataFrame:
    ventricles = [
        "CSF",
        "3rd Ventricle",
        "4th Ventricle",
        "Inferior Lateral Ventricle",
        "Lateral Ventricle",
        "Background",
    ]
    rid_to_cluster = load_rid_and_cluster().to_dict()["Cluster Idx"]
    index_to_lobe = (
        pd.read_csv("metadata/data_processed/index_to_lobe_map.csv")
        .set_index("ID")
        .to_dict()["Lobe"]
    )
    index_to_lobe = _map_old_to_new_index(index_to_lobe)
    neuromorph = nib.load(
        "./metadata/data_raw/spm12_tpm/labels_Neuromorphometrics.nii"
    )  # small bug here, fixed
    parcellation = np.asarray(neuromorph.get_fdata())
    rid_brain = ShapBrainAvg.load_brain(rid)
    data_brain = np.asarray(rid_brain.get_fdata())
    lobes = list(set([value for value in index_to_lobe.values() if value not in ventricles]))
    lobe_to_index = {
        lobe: [x for x, y in index_to_lobe.items() if y == lobe] for lobe in lobes
    }
    summed_vals = {x: 0 for x in lobes}
    for lobe, idx in lobe_to_index.items():
        mask = np.full(parcellation.shape, False)
        for id_ in idx:
            mask = mask | (parcellation == id_)
        if np.sum(np.where(mask, 1.0, 0.0)) == 0:
            continue
        masked_brain = np.ma.masked_array(data_brain, mask=~mask)
        summed_vals[lobe] = masked_brain.mean()
    df = pd.Series(summed_vals).to_frame("Shap")
    df["Lobe"] = lobes
    df["RID"] = rid
    df["Cluster"] = rid_to_cluster[rid]
    return df.reset_index()


def mask_brain_rid_lobe_mn_abs(rid: int) -> pd.DataFrame:
    ventricles = [
        "CSF",
        "3rd Ventricle",
        "4th Ventricle",
        "Inferior Lateral Ventricle",
        "Lateral Ventricle",
        "Background",
    ]
    rid_to_cluster = load_rid_and_cluster().to_dict()["Cluster Idx"]
    index_to_lobe = (
        pd.read_csv("metadata/data_processed/index_to_lobe_map.csv")
        .set_index("ID")
        .to_dict()["Lobe"]
    )
    index_to_lobe = _map_old_to_new_index(index_to_lobe)
    neuromorph = nib.load(
        "./metadata/data_raw/spm12_tpm/labels_Neuromorphometrics.nii"
    )  # small bug here, fixed
    parcellation = np.asarray(neuromorph.get_fdata())
    rid_brain = ShapBrainAvgAbs.load_brain(rid)
    data_brain = np.asarray(rid_brain.get_fdata())
    lobes = list(set([value for value in index_to_lobe.values() if value not in ventricles]))
    lobe_to_index = {
        lobe: [x for x, y in index_to_lobe.items() if y == lobe] for lobe in lobes
    }
    summed_vals = {x: 0 for x in lobes}
    for lobe, idx in lobe_to_index.items():
        mask = np.full(parcellation.shape, False)
        for id_ in idx:
            mask = mask | (parcellation == id_)
        if np.sum(np.where(mask, 1.0, 0.0)) == 0:
            continue
        masked_brain = np.ma.masked_array(data_brain, mask=~mask)
        summed_vals[lobe] = masked_brain.__abs__().mean()
    df = pd.Series(summed_vals).to_frame("Shap")
    df["Lobe"] = lobes
    df["RID"] = rid
    df["Cluster"] = rid_to_cluster[rid]
    return df.reset_index()


def _assign_id_to_side(l: list, side: str) -> int:
    if len(l) == 1:
        return int(l[0])
    assert len(l) == 2
    if side == "l":
        return int(l[1])
    elif side == "r":
        return int(l[0])
    raise ValueError


def _construct_old_to_new_index_map(new_index_to_lobe: pd.DataFrame) -> Dict[int, int]:
    df = pd.read_csv("./metadata/data_raw/neuromorphometrics/neuromorphometrics.csv", sep=";")
    df = df.set_index("ROIid")
    df = df.assign(
            ROIid_new=df["ROIbaseid"].apply(lambda x: x.strip("[]").strip().split(" ")),
            Side=df["ROIabbr"].apply(lambda x: x[0] if x != "BG" else x)
        )
    df = df.assign(
            ROIid_new=df[["ROIid_new", "Side"]].agg(lambda x: _assign_id_to_side(x[0],x[1]), axis=1)
        )
    roi_dict = df["ROIid_new"].to_dict()
    return roi_dict

def _map_old_to_new_index(new_index_to_lobe: pd.DataFrame) -> Dict[int, int]:
    roi_dict = _construct_old_to_new_index_map(new_index_to_lobe)
    new_dict = {roi_dict[x]: y for x,y in new_index_to_lobe.items()}
    return new_dict

def load_masked_shap_rid_mn() -> pd.DataFrame:
    return pd.read_csv("metadata/data_processed/masked_shap_brains_rid_mn.csv")

def mask_brain_all_lobe_mn(debug=False) -> pd.DataFrame:
    df = []
    rid_and_cluster_map = load_rid_and_cluster()
    for rid in tqdm(rid_and_cluster_map.index, leave=True, position=0):
        df.append(mask_brain_rid_lobe_mn(rid))
    df = pd.concat(df, axis=0, ignore_index=True)
    df.rename(columns={"index": "Region"}, inplace=True)
    df["Cluster"] = df["Cluster"].replace({0: "H", 1: "IH", 2: "IL", 3: "L"})
    if not debug:
        df.to_csv("metadata/data_processed/masked_shap_brains_rid_lobe_mn.csv", index=False)
    return df

def mask_brain_all_lobe_mn_abs(debug=False) -> pd.DataFrame:
    df = []
    rid_and_cluster_map = load_rid_and_cluster()
    for rid in tqdm(rid_and_cluster_map.index, leave=True, position=0):
        df.append(mask_brain_rid_lobe_mn_abs(rid))
    df = pd.concat(df, axis=0, ignore_index=True)
    df.rename(columns={"index": "Region"}, inplace=True)
    df["Cluster"] = df["Cluster"].replace({0: "H", 1: "IH", 2: "IL", 3: "L"})
    if not debug:
        df.to_csv("metadata/data_processed/masked_shap_brains_rid_lobe_mn_abs.csv", index=False)
    return df

def map_brain_ids_orig_to_new() -> pd.DataFrame:
    index_to_region = (
        pd.read_csv("metadata/data_processed/id_to_region_map.csv")
        .set_index("ID")
        .to_dict()["RegionName"]
    )

    new_to_old = _construct_old_to_new_index_map(index_to_region)

    img = nib.load("./metadata/data_raw/neuromorphometrics/neuromorphometrics.nii")

    dat = img.get_fdata()

    datnew = np.zeros_like(dat)

    for key, val in new_to_old.items():
        datnew[np.where(dat == key)] = val

    img_new = nib.Nifti1Image(datnew, img.affine)

    nib.save(img_new, "metadata/data_processed/neuromorphometrics_cat12_reindexed.nii")

