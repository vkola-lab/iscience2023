import plotly.graph_objects as go
import os
import colorcet as cc
import pandas as pd
import numpy as np
from statistics.utilities import CONFIG
from statistics.clustered_mlp_output_wrappers import load_adni_clusters, load_adni_ad_clusters

def sankey_plot(cluster_to_cluster_map, label):
    fig = go.Figure(data=[go.Sankey(
        # Define nodes
        node = dict(
        pad = 15,
        thickness = 15,
        line = dict(color = "black", width = 0.5),
        label =  label,
        color =  'black'
        ),
        # Add links
        link = dict(
        source =  cluster_to_cluster_map['MCI Cluster'],
        target =  cluster_to_cluster_map['AD Cluster'],
        value =  cluster_to_cluster_map['Weight'],
        label =  cluster_to_cluster_map['label'],
        color =  cluster_to_cluster_map['color']
    ))])
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=24,
            # color="RebeccaPurple"
        )
    )
    os.makedirs(os.path.join(os.path.split(CONFIG['sankey_diagram'])[0],'..'), exist_ok=True)
    fig.write_image(CONFIG['sankey_diagram'])
    return fig
    
def sankey_diagram():
    adni_idx = load_adni_clusters()
    adni_ad_idx = load_adni_ad_clusters()
    adni_idx = adni_idx.loc[adni_ad_idx.index]
    adni_idx.rename(columns={'Cluster Idx': 'MCI Cluster'}, inplace=True)
    adni_ad_idx.rename(columns={'Cluster Idx': 'AD Cluster'}, inplace=True)
    adni_idx = adni_idx.merge(adni_ad_idx, left_index=True, right_index=True)
    adni_idx = _sankey_weights(adni_idx)
    adni_idx = _sankey_prep(adni_idx)
    fig = sankey_plot(adni_idx, label=['Subtype 0','Subtype 1','Subtype 2','Subtype 3',' Subtype 0','Subtype 1','Subtype 2','Subtype 3'])

def _sankey_prep(adni_idx):
    colorset = cc.b_linear_wyor_100_45_c55[50:]
    step = len(colorset) // len(adni_idx)
    adni_idx.sort_values(by=['MCI Cluster','AD Cluster'], inplace=True)
    adni_idx.reset_index(drop=True, inplace=True)
    adni_idx['label'] = adni_idx.apply(lambda x: '{} to {}'.format(x['MCI Cluster'], x['AD Cluster']), axis=1)
    colors_tmp = np.asarray(colorset)[np.arange(0,255-50-step-1,step)]
    adni_idx['color'] = colors_tmp
    print(adni_idx)
    raise NotImplementedError
    return adni_idx

def _sankey_weights(adni_idx):
    weights = pd.crosstab(
        adni_idx['MCI Cluster'], adni_idx['AD Cluster']
    ).reset_index().melt(id_vars='MCI Cluster', value_name='Weight')
    weights['AD Cluster'] = weights['AD Cluster'].astype(int) + 4
    weights['AD Cluster'] = weights['AD Cluster'].astype(str)
    return weights 

def main():
    sankey_diagram()