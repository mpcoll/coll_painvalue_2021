from nilearn.image import load_img, new_img_like, resample_to_img
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from nilearn.plotting import view_img, plot_stat_map
from nilearn import plotting
from nilearn.regions import connected_regions
from nilearn.reporting import get_clusters_table
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ptitprince as pt
from nilearn.image import load_img
from nilearn.masking import apply_mask

###################################################################
# Paths
###################################################################

basepath = '/data'
outpath =  opj(basepath, 'derivatives/figures')
if not os.path.exists(outpath):
    os.makedirs(outpath)

###################################################################
# Plot options
###################################################################

# colors
current_palette = sns.color_palette('colorblind', 6)
colp = current_palette[0]
colm = current_palette[2]

cold = current_palette[1]
colc = current_palette[3]

# Label size
labelfontsize = 7
titlefontsize = np.round(labelfontsize*1.5)
ticksfontsize = np.round(labelfontsize*0.8)
legendfontsize = np.round(labelfontsize*0.8)

# Font
plt.rcParams['font.family'] = 'Helvetica'

# Despine
plt.rc("axes.spines", top=False, right=False)

# Background T1
bgimg = opj(basepath, 'external/tpl-MNI152NLin2009cAsym_space-MNI_res-01_T1w_brain.nii')

# Brain cmap
cmap = plotting.cm.cold_hot

group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))

################################################################
# Slices plot
###################################################################

# Load corrected map
painvsmoney = load_img(opj(basepath, 'derivatives/mvpa/searchlights/sl_crosspainmoney',
                           'sl_crosspm_scores_fwe05.nii'))
view_img(painvsmoney)


# # Remove significant voxels from mask
# nsynth_mask_pm = new_img_like(nsynth_mask, np.where(painvsmoney.get_data() != 0,
#                                                  0, nsynth_mask.get_data()))


# PLot slices
to_plot = {'x': [-6, 2, 12, 0, 6, 36, -22],
           'y': [8, 10, 12, 14],
           'z': [-4, 66, 30]}

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1, 1))
        disp = plot_stat_map(painvsmoney, cmap=cmap, colorbar=False,
                        bg_img=bgimg,
                        dim=-0.3,
                        black_bg=False,
                        display_mode=axis,
                        axes=ax,
                        vmax=0.3,
                        cut_coords=(c,),
                        alpha=1,
                        annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        # disp.add_overlay(nsynth_mask_pm, cmap='binary_r', **{'alpha': 0.7})
        fig.savefig(opj(outpath, 'pvsm_fwe05_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)
# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1, 1))
thr = np.min(np.abs(painvsmoney.get_data()[painvsmoney.get_data() != 0]))
disp = plot_stat_map(painvsmoney,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=False,
                display_mode='x',
                axes=ax,
                threshold=thr,
                vmax=0.3,
                cut_coords=(0,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Pearson r', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
disp._colorbar_ax.set_yticks([-0.29, 0.29])
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels([lab[0], '', '', '', lab[-1]], fontsize=ticksfontsize)
disp._colorbar_ax.yaxis.set_tick_params(pad=-0.5)

fig.savefig(opj(outpath, 'pvm05_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)






# Load corrected map
painvsshock = load_img(opj(basepath, 'derivatives/mvpa/searchlights/searchlight_crosspainshock',
                           'sl_crossps_scores_fwe05.nii'))
view_img(painvsshock, bg_img=bgimg)


# # Remove significant voxels from mask
# nsynth_mask_pm = new_img_like(nsynth_mask, np.where(painvsmoney.get_data() != 0,
#                                                  0, nsynth_mask.get_data()))


# PLot slices
to_plot = {'x': [-30, 34],
           'y': [12, 36],
           'z': [-7, -10, -14]}

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1, 1))
        disp = plot_stat_map(painvsshock, cmap=cmap, colorbar=False,
                        bg_img=bgimg,
                        dim=-0.3,
                        black_bg=False,
                        display_mode=axis,
                        symmetric_cbar=True,
                        axes=ax,
                        vmax=0.35,
                        cut_coords=(c,),
                        alpha=1,
                        annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        # disp.add_overlay(nsynth_mask_pm, cmap='binary_r', **{'alpha': 0.7})
        fig.savefig(opj(outpath, 'pvsp_fwe05_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)
# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1, 1))
thr = np.min(np.abs(painvsshock.get_data()[painvsshock.get_data() != 0]))
disp = plot_stat_map(painvsshock,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=True,
                display_mode='x',
                axes=ax,
                threshold=thr,
                vmax=0.35,
                cut_coords=(-30,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Pearson r', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
disp._colorbar_ax.set_yticks([-0.25, 0.25])
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels([lab[0], '', '', '', lab[-1]], fontsize=ticksfontsize)
disp._colorbar_ax.yaxis.set_tick_params(pad=-0.5)

fig.savefig(opj(outpath, 'pvp05_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)


# Load corrected map
painvsemo= load_img(opj(basepath, 'derivatives/mvpa/searchlights/searchlight_crosspainemo',
                           'sl_crosspemo_scores_fwe05.nii'))
view_img(painvsemo, bg_img=bgimg)



# PLot slices
to_plot = {'x': [2, -2, -28, 38],
           'y': [],
           'z': [-6, 36, 2, -8]}

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1, 1))
        disp = plot_stat_map(painvsemo, cmap=cmap, colorbar=False,
                        bg_img=bgimg,
                        dim=-0.3,
                        black_bg=False,
                        display_mode=axis,
                        symmetric_cbar=True,
                        axes=ax,
                        vmax=0.3,
                        cut_coords=(c,),
                        alpha=1,
                        annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        # disp.add_overlay(nsynth_mask_pm, cmap='binary_r', **{'alpha': 0.7})
        fig.savefig(opj(outpath, 'pvse_fwe05_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)
# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1, 1))
thr = np.min(np.abs(painvsemo.get_data()[painvsemo.get_data() != 0]))
disp = plot_stat_map(painvsemo,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=True,
                display_mode='x',
                axes=ax,
                threshold=thr,
                vmax=0.3,
                cut_coords=(2,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Pearson r', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
disp._colorbar_ax.set_yticks([-0.25, 0.25])
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels([lab[0], '', '', '', lab[-1]], fontsize=ticksfontsize)
disp._colorbar_ax.yaxis.set_tick_params(pad=-0.5)

fig.savefig(opj(outpath, 'pve05_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)



