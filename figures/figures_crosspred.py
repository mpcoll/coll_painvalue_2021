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
labelfontsize = 9
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
nsynth_mask = load_img(opj(basepath, 'derivatives/meta_mask.nii.gz'))

################################################################
# Slices plot
###################################################################

# Load corrected map
painvsmoney = load_img(opj(basepath, 'derivatives/mvpa/searchlights/searchlight_crosspainmoney',
                           'sl_crosspm_scores_fwe05.nii'))
view_img(painvsmoney)


# PLot slices
to_plot = {'x': [-6, 2, 12, 0, 6, 36, -22],
           'y': [12, 14],
           'z': [-4, 66, 30]}

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
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
fig, ax = plt.subplots(figsize=(1.5, 1.5))
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
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)

fig.savefig(opj(outpath, 'pvm05_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)





################################################################
# Slices plot in striatum ROI
###################################################################

# Load corrected map
painvsmoney = load_img(opj(basepath, 'derivatives/mvpa/searchlights/sl_crosspainmoney_wb_new',
                           'sl_crosspm_strROI_scores.nii'))

# PLot slices
to_plot = {
            # 'x': [],
           'y': [8, 10, 12, 14],
        #    'z': []
           }

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
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
        fig.savefig(opj(outpath, 'pvsm_uncSTRROI_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)

# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1.5, 1.5))
# thr = np.min(np.abs(painvsmoney.get_data()[painvsmoney.get_data() != 0]))
disp = plot_stat_map(painvsmoney,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=False,
                display_mode='y',
                axes=ax,
                # threshold=thr,
                vmax=0.3,
                cut_coords=(12,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Pearson r', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)

fig.savefig(opj(outpath, 'pvm05_unc_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)





# Load maps for sl in STR only
painvsmoney = load_img(opj(basepath, 'derivatives/mvpa/searchlights/sl_crosspainmoney_strROI',
                           'sl_crosspm_roi_scores_fwe05.nii'))

# PLot slices
to_plot = {
            # 'x': [],
           'y': [8, 10, 12, 14],
        #    'z': []
           }

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
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
        fig.savefig(opj(outpath, 'pvsm_few05_SLstrROI_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)

# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1.5, 1.5))
thr = np.min(np.abs(painvsmoney.get_data()[painvsmoney.get_data() != 0]))
disp = plot_stat_map(painvsmoney,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=False,
                display_mode='y',
                axes=ax,
                threshold=thr,
                vmax=0.3,
                cut_coords=(12,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Pearson r', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)

fig.savefig(opj(outpath, 'pvm05_few05_SLstrROI_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)



# Load maps for sl in STR only
painvsmoney = load_img(opj(basepath, 'derivatives/mvpa/searchlights/sl_crosspainmoney_strROI',
                           'sl_crosspm_roi_slscores.nii'))

# PLot slices
to_plot = {
           'y': [8, 10, 12, 14],
           }

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
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
        fig.savefig(opj(outpath, 'pvm05_unthres_SLstrROI_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)

# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1.5, 1.5))
thr = np.min(np.abs(painvsmoney.get_data()[painvsmoney.get_data() != 0]))
disp = plot_stat_map(painvsmoney,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=False,
                display_mode='y',
                axes=ax,
                threshold=thr,
                vmax=0.3,
                cut_coords=(12,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Pearson r', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)

fig.savefig(opj(outpath, 'pvm05_unthresh_SLstrROI_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)
