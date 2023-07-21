import pandas as pd
import re
import os
import numpy as np
import nibabel as nib
import abc
import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import subprocess
from nilearn import plotting
from typing import Dict, Tuple
from statistics.utilities \
    import CONFIG
from statistics.clustered_mlp_output_wrappers import load_any_long, \
    load_adni_clusters, load_adni_ad_clusters
from statistics.mlp_output_wrappers import load_roiname_to_roiid_map
from statistics.aggregate_utilities import cluster_average, \
    group_cortical_regions_by_subtype
from statistics.dataframe_validation import DataFrame, \
    ParcellationClusteredLongSchema, pa
from statistics.shap_viz.shap_plots import get_region_order

__all__ = [
    'load_brain_for_groups',
    'load_brain_for_groups_xyz_rank'
    ]

class Brain(object):
    def __init__(self, props=None, rid='', train_or_test='ADNI'):
        if props is None:
            props = CONFIG
        self.props = props
        for key, value in self.props.items():
            setattr(self, key, value)
        """MICA -- you will want to plot the mask"""
        self.mask_path = os.path.join(
                self.image_directories[train_or_test]['atlas'],
                f'wneuromorphometrics_{rid}_mri.nii'
        )
        self.original_brain_path = os.path.join(
                self.image_directories[train_or_test]['basedir'],
                f'masked_brain_mri_{rid}.nii'
        )
        self.rid = rid
        self._load_mask() # loading mask here
        self._load_brain()
        self._recoded = False

    def _load_mask(self) -> None:
        try:
            data = nib.load(self.mask_path)
            self.mask = data.get_fdata()
            self.hdr = data.header
            self.affine = data.affine
            self.mask_img = data
        except FileNotFoundError as e:
            print(e)
            self.mask = np.nan
            self.hdr = np.nan
            self.affine = np.nan

    def _load_brain(self) -> None:
        try:
            data = nib.load(self.original_brain_path)
            self.original_brain = data.get_fdata()
            self.brain_img = data
        except FileNotFoundError as e:
            print(e)
            self.original_brain = np.array([])

    @abc.abstractmethod
    # def plot_brain(self) -> None:
    #     raise NotImplementedError
    
    def recode_background(self):
        if not all([np.isnan(x) for x in self.mask.reshape(-1,1)]):
            mask = self.mask.copy()
            mask[mask == 1] = np.nan
            self.mask_img = nib.Nifti1Image(np.ma.masked_invalid(mask), self.affine)
            self._recoded = True

    def plot_brain(self, title='', cmap_nm='glasbey_dark', cut_coords=(range(
            -51,-1,20), range(-41,1,20), range(-21,21,20))):
        mask_img_path = './figures/figure_4'
        f_name = os.path.join(mask_img_path,
                              f'mri_parcellated_brain_{self.rid}.svg')
        os.makedirs(mask_img_path, exist_ok=True)
        if not self._recoded:
            self.recode_background()
        cmap = copy.copy(cc.cm[cmap_nm])
        cmap.set_bad('w', alpha=1)
        for idx, dim in enumerate(('x','y','z')):
            if idx == 0:
                title = f'RID {self.rid}'
            else:
                title = ''
            # please make background white or black, make regions
            # as distinct as possible, and can use 3x3 grid
            plotting.plot_img(self.mask_img,  # could replace with self.mask_img
                                display_mode=dim,
                                cut_coords=cut_coords[idx],
                                cmap = cmap,
                                colorbar=False,
                                annotate=False,
                                title=title,
                                axes=plt.subplot(3,1,idx+1),
                                )
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.close()

class BrainSample(object):
    def __init__(self, props=None, brain_path='', title=''):
        if props is None:
            props = CONFIG
        self.props = props
        for key, value in self.props.items():
            setattr(self, key, value)
        self.title = title
        self.plot_img_path = './figures/test_figs'
        self.plot_img_prefix = os.path.join(self.plot_img_path,
                                f'mri_slice_{self.title}')
        self._brain_path = brain_path
        self._load_brain()

    def _load_brain(self) -> None:
        try:
            data = nib.load(self._brain_path)
            self.original_brain = data.get_fdata()
            self.brain_img = data
            self.affine = data.affine
            self.shape = data.header.get_data_shape()
            self.center = np.asarray(self.shape) // 2
            self.top_corner = nib.affines.apply_affine(
                    self.affine, self.shape[:-1])
            self.bottom_corner = nib.affines.apply_affine(
                    self.affine, [0, 0, 0])
            self.range = [[int(x),int(y)] for x,y in zip(self.bottom_corner,
                                               self.top_corner)]
        except FileNotFoundError as e:
            print(e)
            self.original_brain = np.array([])

    def plot_slices(self):
        os.makedirs(self.plot_img_path, exist_ok=True)
        for idx, dim in enumerate(('x','y','z')):
            for _slice in range(self.range[idx][0],self.range[idx][1],5):
                ax = plt.subplot(111)
                plotting.plot_anat(self.brain_img,
                                   axes=ax,
                                   display_mode=dim,
                                   cut_coords=[_slice])
                f_name = f'{self.plot_img_prefix}_{dim}_' \
                         f'{str(_slice).zfill(3)}.pdf'
                plt.savefig(f_name, dpi=300, transparent=True)
                plt.close()

    def concatenate_slices(self):
        fi = os.listdir(self.plot_img_path)
        fi = [os.path.join(self.plot_img_path, x) for x in fi]
        for idx, dim in enumerate(('x','y','z')):
            matching_fi = list(filter(lambda x: re.match(
                    f'^{self.plot_img_prefix}_{dim}_[0-9]+\.pdf',x), fi))
            matching_fi = ' '.join(matching_fi)
            cmd = f'pdftk {matching_fi} cat output ' \
                  f'{self.plot_img_prefix}_{dim}'
            subprocess.run(cmd, shell=True, capture_output=True)
            rm_cmd = f'rm {matching_fi}'
            subprocess.run(rm_cmd, shell=True)

# from https://nipy.org/nibabel/coordinate_systems.html

class ShapBrain(Brain):
    def __init__(self, props=None, rid=None, shap_map=None, _bin=None,
                 threshold=0.001, name=''):
        if props is None:
            props = CONFIG
        if rid is not None:
            super().__init__(props=props, rid=rid, train_or_test='ADNI')
            self.path_config = None
        else:
            self.rid = name
            for prop, item in props.items():
                setattr(self, prop, item)
            self.mask_img = nib.load(
                    self.path_config['neuromorphometrics_mask']
            )
            self.mask = self.mask_img.get_fdata()
            self.hdr = self.mask_img.header
            self.affine = self.mask_img.affine
            self.original_brain = None
            self.brain_img = None
        self.shap_mask_img_path = os.path.join('.', 'figures',
                                                    'shap_glass_brain')
        self.shap_mask_path = os.path.join('.', 'metadata','data_processed'
                                                    'shap_masks')
        self.shap_map = shap_map
        self.bin = _bin
        self.threshold = threshold
        self._make_shap_mask()

    def _make_shap_mask(self) -> None:
        shap_map = self.shap_map
        mask = self.mask
        shap_mask = np.zeros_like(mask)
        for idx in shap_map.keys():
            if abs(shap_map[idx]) >= self.threshold:
                shap_mask[np.where(mask == idx)] = shap_map[idx]
        shap_mask[np.where(mask < 1)] = np.nan
        self.shap_mask = shap_mask
        self.shap_img = nib.Nifti1Image(shap_mask, self.affine,
                                             header=self.hdr)

    def make_shap_mask_select(self, region_idx) -> None:
        shap_map = self.shap_map
        mask = self.mask
        shap_mask = np.zeros_like(mask)
        for idx in region_idx:
            shap_mask[np.where(mask == idx)] = shap_map[idx]
        shap_mask[np.where(mask < 1)] = np.nan
        self.shap_mask = shap_mask
        self.shap_img = nib.Nifti1Image(shap_mask, self.affine,
                                             header=self.hdr)

    def save_brain(self) -> None:
        f_name = os.path.join(self.shap_mask_path,
                              f'shap_mask_{self.rid}_bin{self.bin}')
        os.makedirs(self.shap_mask_path, exist_ok=True)
        nib.save(self.shap_img, f_name + '.nii')

    def plot_brain(self, vmax=0.015, title=''):
        f_name = {}
        f_name['svg'] = os.path.join(self.shap_mask_img_path,
                              f'stat_map_{self.rid}_bin'
                              f'{self.bin}.svg')
        f_name['png'] = os.path.join(self.shap_mask_img_path,
                              f'stat_map_{self.rid}_bin'
                              f'{self.bin}.png')
        os.makedirs(self.shap_mask_img_path, exist_ok=True)
        for _, value in f_name.items():
            if self.brain_img is not None:
                brain_plt = plotting.plot_stat_map(self.shap_img,
                                display_mode='z',
                                cut_coords=range(-20,21,20),
                                bg_img=self.brain_img,
                                cmap = cc.cm.bmy,
                                colorbar=True,
                                annotate=False,
                                title=title
                                )
            else:
                brain_plt = plotting.plot_stat_map(self.shap_img,
                                display_mode='z',
                                cut_coords=range(-20,21,20),
                                cmap = cc.cm.bmy,
                                colorbar=True, vmax=vmax,
                                symmetric_cbar=True,
                                annotate=False,
                                title=title
                                )
            plt.savefig(value, dpi=300)
            plt.close()
    
    def plot_brain_new(self, vmax=0.015, title='', cut_coords=None):
        img_path = './figures'
        if cut_coords is None:
            cut_coords = []
            for c in ('x','y','z'):
                cut_coords.append(
                    plotting.find_cut_slices(
                        self.img,
                        direction=c,
                        n_cuts=1,
                        spacing='auto')
                    )
        f_name = os.path.join(img_path,
                              f'shap_cnn_brain_{self.rid}.png')
        os.makedirs(img_path, exist_ok=True)
        plotting.plot_stat_map(self.img,  # could replace with self.mask_img
                                bg_img=None,
                                cut_coords=cut_coords,
                                colorbar=True,
                                cmap=plt.cm.bwr,
                                annotate=False,
                                title=title,
                                draw_cross=False,
                                black_bg=True,
                                vmax=0.00025
                                )
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.savefig(f_name[:-4] + '.svg', dpi=300, transparent=True)
        plt.close()
        return cut_coords

    def plot_brain_xyz(self, vmax=0.015, title='',
                       cut_coords=(range(-51,-1,20),
                                   range(-41,1,20),
                                   range(-21,21,20))):
        f_name = os.path.join(self.shap_mask_img_path,
                              f'xyz_{self.rid}.svg')
        os.makedirs(self.shap_mask_img_path, exist_ok=True)
        for idx, dim in enumerate(('x','y','z')):
            if idx == 1:
                colorbar = True
            else:
                colorbar = False
            plotting.plot_stat_map(self.shap_img,
                                display_mode=dim,
                                bg_img=None,
                                cut_coords=cut_coords[idx],
                                cmap = cc.cm.bmy,
                                colorbar=colorbar, vmax=vmax,
                                symmetric_cbar=True,
                                annotate=False,
                                title=title,
                                axes=plt.subplot(3,1,idx+1)
                                )
        plt.savefig(f_name, dpi=300)
        plt.close()

class MriBrain(Brain):
    def __init__(self, rid, cluster_idx=-1, dataset='ADNI'):
        super().__init__(props=CONFIG, rid=rid, train_or_test=dataset)
        self.mask_img_path = os.path.join('.','figures','mri_brain')
        self.cluster_idx = cluster_idx

    def plot_brain(self, vmax=0.015, title='',
                   cut_coords=(
                           range(-51,-1,20),
                           range(-41,1,20),
                           range(-21,21,20))):
        f_name = os.path.join(self.mask_img_path,
                              f'mri_cluster_{self.cluster_idx}_{self.rid}.pdf')
        os.makedirs(self.mask_img_path, exist_ok=True)
        for idx, dim in enumerate(('x','y','z')):
            if idx == 0:
                title = f'Cluster {self.cluster_idx}, RID {self.rid}'
            else:
                title = ''

            plotting.plot_img(self.brain_img,
                                display_mode=dim,
                                cut_coords=cut_coords[idx],
                                cmap = cc.cm.gray,
                                colorbar=False,
                                annotate=False,
                                title=title,
                                axes=plt.subplot(3,1,idx+1)
                                )
        plt.savefig(f_name, dpi=300)
        plt.close()

class ParcellatedBrain(Brain):
    def __init__(self, rid, dataset='ADNI'):
        super().__init__(props=CONFIG, rid=rid, train_or_test=dataset)
        self.mask_img_path = os.path.join('.','figures','parcellated_brain')

    def plot_brain(self, vmax=np.nan, title='', cut_coords=(-31, -31, -31)):  #[
        f_name = os.path.join(self.mask_img_path,
                              f'mri_parcellated_brain_{self.rid}.svg')
        os.makedirs(self.mask_img_path, exist_ok=True)
        for idx, dim in enumerate(('x','y','z')):
            if idx == 0:
                title = f'RID {self.rid}'
            else:
                title = ''
            # please make background white or black, make regions
            # as distinct as possible, and can use 3x3 grid
            plotting.plot_img(self.mask_img,  # could replace with self.mask_img
                                display_mode=dim,
                                cut_coords=cut_coords[idx],
                                cmap = cc.cm.bmy,
                                colorbar=False,
                                annotate=False,
                                title=title,
                                axes=plt.subplot(3,1,idx+1)
                                )
        plt.savefig(f_name, dpi=300)
        plt.close()

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    return axes

def format_pathology_region_map(region_map_df: pd.DataFrame):
    region_map_df = region_map_df.copy()
    region_map_df.rename(columns={'Corresponding neuromorphometrics': 'idx'},
                         inplace=True)
    region_map_df.loc[:, 'idx'] = region_map_df.loc[:, 'idx'].apply(
            lambda x: list(map(int, x.split(','))))
    return region_map_df

def _generate_index_to_value_map(
        df: DataFrame[ParcellationClusteredLongSchema],
        value_set: str='Shap Value') -> pd.Series:
    region_average_over_group = cluster_average(df=df, value_set=value_set)
    name_to_id_map = load_roiname_to_roiid_map()
    region_average_over_group = region_average_over_group[['Region', 'Cluster Idx', value_set]]
    idx_to_value_map = region_average_over_group.groupby('Cluster Idx').apply(
        lambda x: _generate_neuromorph_dictionary(x, name_to_id_map, value_set)
    )
    return idx_to_value_map

def _generate_neuromorph_dictionary(region_average_over_group: pd.DataFrame,
                                    name_to_id_map: Dict[str,list], value_set:
        str):
    schema = pa.DataFrameSchema({
        'Region' : pa.Column(str),
        'Cluster Idx': pa.Column(str),
        'Shap Value': pa.Column(float, required=False),
        'Gray Matter Vol': pa.Column(float, required=False)
    })
    region_average_over_group = schema.validate(region_average_over_group)
    region_values = region_average_over_group.set_index('Region', drop=True)[value_set]
    region_values = region_values.squeeze().to_dict()
    idx_to_value_dict = {}   # for each region label integer, assign value for that region in group
    for region, region_indices in name_to_id_map.items():  # iterate over all regions and mask_idx for the neuromorphometric atlas
        for region_idx in region_indices:
            if region in region_values.keys():  # over all region labels in the list of mask labels   
                idx_to_value_dict[region_idx] = region_values[region]
            else:
                idx_to_value_dict[region_idx] = np.nan
    return idx_to_value_dict

def load_brain_for_groups(dataset: str='NACC',
                            value_set: str='Shap Value',
                            threshold: float=2.5,
                            _plt: bool=True,
                            vmax: float=10.0,
                            xyz: bool=False
                          ):
    df = load_any_long(dataset).copy()
    idx_to_value_map = _generate_index_to_value_map(df, value_set)
    shap_brains = []
    value_set = value_set.replace(' ', '').lower()
    for group, shap_map in idx_to_value_map.items():
        if group == -1:
            group = 'unclustered'
        fname=f'{value_set}_for_cluster' \
                f'_{group}_{dataset}'
        shap_brains.append(ShapBrain(
                props=CONFIG,
                rid=None,
                shap_map=shap_map,
                _bin='all',
                threshold=threshold,
                name=fname,
                )
        )
    if _plt:
        for brain in shap_brains:
            if xyz:
                brain.plot_brain_xyz(vmax=vmax)
            else:
                brain.plot_brain(vmax=vmax)
    return shap_brains

def load_brain_for_groups_xyz_rank(dataset='NACC',
                            value_set='Shap Value',
                            threshold=2.5,
                            _plt=True,
                            vmax=10, top_or_bottom='top'):
    if top_or_bottom not in ('top','bottom'):
        raise NotImplementedError
    df = load_any_long(dataset).copy()
    idx_to_value_map = _generate_index_to_value_map(df, value_set)
    top_and_bottom_regions = {}
    for cluster_idx, cluster_df in df.groupby('Cluster Idx'):
        top_and_bottom_regions[cluster_idx] = \
            get_region_order(cluster_df, by='corr')
    name_to_id_map = load_roiname_to_roiid_map()
    shap_brains = []
    value_set = value_set.replace(' ', '').lower()
    for cluster, shap_map in idx_to_value_map.items():
        regions = top_and_bottom_regions[cluster][top_or_bottom]
        idx = []
        [idx.append(_id) for region in regions for _id in name_to_id_map[region]]
        if cluster == -1:
            group = 'unclustered'
        fname=f'{value_set}_for_cluster' \
                f'_{cluster}_{dataset}_{top_or_bottom}'
        sb = ShapBrain(
                        props=CONFIG,
                        rid=None,
                        shap_map=shap_map,
                        _bin='all',
                        threshold=threshold,
                        name=fname,
                    )
        sb.make_shap_mask_select(idx)
        shap_brains.append(sb)
        if _plt:
            sb.plot_brain_xyz(vmax=vmax)
    return shap_brains

def plot_all_cortical_regions_by_subtype():
    df_adni = plot_cortical_regions_by_subtype('ADNI')
    df_adni['Time'] = 'MCI'
    df_adniad = plot_cortical_regions_by_subtype('ADNI_AD')
    df_adniad['Time'] = 'AD'
    df_combined = pd.concat([df_adni, df_adniad], axis=0)
    df_combined.to_csv('./results/cortical_regions_by_subtype_partial.csv', index=False)
    return df_combined

def plot_cortical_regions_by_subtype(dataset: str='ADNI'):
    plt.style.use('./statistics/styles/style.mplstyle')
    tbl = group_cortical_regions_by_subtype(dataset=dataset)
    tbl_all = tbl.copy()
    tbl = tbl.groupby(['Cortical Region','Subtype']).agg(np.mean)
    tbl.reset_index(inplace=True)
    tbl['Subtype'] = tbl['Subtype'].replace({'0': 'H', '1': 'IH', '2': 'IL', '3': 'L'})
    tbl = tbl.pivot('Cortical Region', 'Subtype', 'ZS Gray Matter Volume')
    sns.lineplot(
        data=tbl,
        ci=None,
        dashes=True,
        palette='flare',
        lw=5
    )
    plt.xticks(rotation=45, horizontalalignment='right')
    ax = plt.gca()
    ax.set_ylim([-1.5, 1.5])
    plt.savefig(f'figures/cortical_regions_by_subtype_{dataset}.svg')
    plt.savefig(f'figures/cortical_regions_by_subtype_{dataset}.png', bbox_inches='tight')
    plt.close()
    return tbl_all

def generate_shap_brains() -> None:
    load_brain_for_groups(_plt=True,
                             dataset='NACC', value_set='Shap Value',
                             threshold=0.001, vmax=0.005)
    load_brain_for_groups(_plt=True,
                             dataset='NACC', value_set='Gray Matter Vol',
                             threshold=0.3, vmax=1.5)


def generate_mri_brains_by_cluster() -> None:
    adni_idx = load_adni_clusters()
    for rid in adni_idx.index:
        brain = MriBrain(rid, cluster_idx=adni_idx.loc[rid, 'Cluster Idx'], dataset='ADNI')
        brain.plot_brain()
    adni_ad_idx = load_adni_ad_clusters()
    for rid in adni_ad_idx.index:
        brain = MriBrain(rid, cluster_idx=adni_idx.loc[rid, 'Cluster Idx'], dataset='ADNI_AD')
        brain.plot_brain()

def generate_gmv_xyz() -> None:
    # load_brain_for_groups(_plt=True,
    #                          dataset='NACC', value_set='Gray Matter Vol',
    #                          threshold=0.3, vmax=1.5, xyz=True)
    # load_brain_for_groups(_plt=True,
    #                          dataset='ADNI', value_set='Gray Matter Vol',
    #                          threshold=0.3, vmax=1.5, xyz=True)
    # load_brain_for_groups(_plt=True,
    #                         dataset='ADNI_AD', value_set='Gray Matter Vol',
    #                         threshold=0.3, vmax=1.5, xyz=True)

    load_brain_for_groups(_plt=True,
                             dataset='NACC', value_set='Gray Matter Vol',
                             threshold=0.0, vmax=1.5, xyz=True)
    load_brain_for_groups(_plt=True,
                             dataset='ADNI', value_set='Gray Matter Vol',
                             threshold=0.0, vmax=1.5, xyz=True)

def generate_shap_values_xyz() -> None:
    load_brain_for_groups(_plt=True,
                            dataset='ADNI_AD', value_set='Gray Matter Vol',
                            threshold=0.3, vmax=1.5,xyz=True)
    load_brain_for_groups(_plt=True,
                            dataset='ADNI', value_set='Gray Matter Vol',
                            threshold=0.3, vmax=1.5, xyz=True)

def generate_shap_values_by_cluster_ranked() -> None:
    load_brain_for_groups_xyz_rank(_plt=True,
                             dataset='NACC', value_set='Shap Value',
                             threshold=0.001, vmax=0.005, top_or_bottom='top')
    load_brain_for_groups_xyz_rank(_plt=True,
                             dataset='NACC', value_set='Shap Value',
                             threshold=0.001, vmax=0.005, top_or_bottom='bottom')

def main():
    # figs = os.listdir('./figures/shuffled_mris_nii')
    # for fig in figs:
    #     brain = BrainSample(brain_path='./figures/shuffled_mris_nii/' + fig,
    #                         title=fig[:-4])
    #     brain.plot_slices()
    #     brain.concatenate_slices(
    # Brain(rid='0746').plot_brain()
    # generate_shap_brains()
    generate_gmv_xyz()
    # generate_shap_values_xyz()
    # generate_shap_values_by_cluster_ranked()
    # plot_all_cortical_regions_by_subtype()