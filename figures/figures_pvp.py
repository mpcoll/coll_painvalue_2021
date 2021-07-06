from nilearn.image import load_img, new_img_like, concat_imgs
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from nilearn.plotting import view_img, plot_stat_map
from nilearn import plotting
from nilearn.regions import connected_regions

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ptitprince as pt
from nilearn.image import load_img
from nilearn.masking import apply_mask, unmask
from scipy.stats import zscore
from nltools.analysis import Roc
from scipy.stats import pearsonr, zscore, spearmanr
from matplotlib.patches import Patch
from scipy.stats import binom_test
import matplotlib.lines as mlines
from matplotlib import colors

###################################################################
# Paths
###################################################################

basepath = '/data'
outpath =  opj(basepath, 'derivatives/figures')
if not os.path.exists(outpath):
    os.makedirs(outpath)
group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))

###################################################################
# Plot options
###################################################################

# colors
current_palette = sns.color_palette('colorblind', 10)
colp = current_palette[0]
colm = current_palette[2]

cold = current_palette[1]
colc = current_palette[4]
cole = current_palette[8]

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

###################################################################
# Slices plot
###################################################################

# Load corrected map
painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                           'painlevel_bootz_fdr05.nii'))
view_img(painmapfdr, bg_img=bgimg)
# PLot slices
to_plot = {'x': [-38, -8, -6, 6, 10, 12],
           'y': [10],
           'z': [-10, 6]}

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        disp = plot_stat_map(painmapfdr, cmap=cmap, colorbar=False,
                        bg_img=bgimg,
                        dim=-0.3,
                        black_bg=False,
                        display_mode=axis,
                        axes=ax,
                        vmax=6,
                        cut_coords=(c,),
                        alpha=1,
                        annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        fig.savefig(opj(outpath, 'pmap_fdr05_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)


# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1.5, 1.5))
thr = np.min(np.abs(painmapfdr.get_data()[painmapfdr.get_data() != 0]))
disp = plot_stat_map(painmapfdr,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=True,
                display_mode='x',
                threshold=thr,
                axes=ax,
                vmax=6,
                cut_coords=(0,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Z score', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)

fig.savefig(opj(outpath, 'pvp_slices_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)


###################################################################
# y_pred plot
###################################################################

corr_pain = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                            'painlevel_cvstats.csv'))

fig, ax = plt.subplots(figsize=(1.6, 2.2))

corr_pain['pvp_cv_cosine_z'] = zscore(corr_pain['pvp_cv_cosine'])
corr_pain['Y_true'] = corr_pain['Y_true'].astype(int)

ax = sns.pointplot(y="pvp_cv_cosine_z", x='Y_true',
                   data=corr_pain,
                   scale=0.4, ci=68, errwidth=1, color=colp)  # 68% CI are SEM
# Add labels
ax.set_title("",  {'fontsize': titlefontsize})
ax.set_xlabel("Pain offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})

ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_pain.svg'), transparent=True)


###################################################################
# y_pred plot with money
###################################################################

corr_money = pd.read_csv(opj(basepath, 'derivatives/mvpa/money_offer_level', 
                             'moneylevel_cvstats.csv'))
corr_money['Y_true'] = corr_money['Y_true'].astype(int)
corr_money['type'] = 'money'
corr_pain['type'] = "pain"

# PVP and MVP on pain in same graph
corr_pain['pvp_cv_cosine_z'] = zscore(corr_pain['pvp_cv_cosine'])
corr_pain['mvp_cv_cosine_z'] = zscore(corr_pain['mvp_cv_cosine'])

corr_both = pd.concat([corr_pain]).melt(id_vars=['Y_true', 'type'], value_vars=['pvp_cv_cosine_z', 'mvp_cv_cosine_z'])
corr_both['cv_cosine_z'] = zscore(corr_both['value'])

corr_both['Y_true'] = corr_both['Y_true'].astype(int)
fig, ax = plt.subplots(figsize=(1.6, 2.2))


sns.pointplot('Y_true', 'value', hue='variable', data=corr_both,
                   palette=[colp, "#003399"], markers=['o', 's'], linestyles=['-', '--'],
                   scale=0.4, ax=ax, ci=68, errwidth=1, label='Pain offer')

# Add labels
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Pain offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False, handletextpad=0.05, loc=(0, 0.6))
for t, l in zip(legend.texts, ("Pain\npattern", "Money\npattern")):
    t.set_text(l)

ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()
fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_painwmoney.svg'), transparent=True)


# ###################################################################
# # y_pred plot with money and shock
# ###################################################################

corr_money = pd.read_csv(opj(basepath, 'derivatives/mvpa/money_offer_level',
                             'moneylevel_cvstats.csv'))

corr_shock = pd.read_csv(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                             'shocklevel_cvstats.csv'))

corr_ant = pd.read_csv(opj(basepath, 'derivatives/mvpa/anticipation_level',
                             'anticipation_cvstats.csv'))


fig, ax = plt.subplots(figsize=(1.6, 2.2))

corr_money['Y_true'] = corr_money['Y_true'].astype(int)


ax = sns.pointplot(corr_money['Y_true'], zscore(corr_money['pvp_cv_cosine']),
                   color=colp, label='Pain pattern',
                   scale=0.4, ci=68, errwidth=1)

# Add labels
ax.set_ylim(-0.7, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Money offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(labels=['Pain\npattern'],
                   fontsize=legendfontsize, frameon=False, loc=(0, 0.8))
# for t, l in zip(legend.texts, ("Money offer")):
#     t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_pvp_money.svg'), transparent=True)


fig, ax = plt.subplots(figsize=(1.6, 1.6))

corr_shock['Y_true'] = corr_shock['Y_true'].astype(int)


ax = sns.pointplot(corr_shock['Y_true'], zscore(corr_shock['pvp_cv_cosine']),
                   color=cold,
                   scale=0.4, ci=68, errwidth=1)

# Add labels
ax.set_ylim(-0.6, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Shock intensity", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False)
for t, l in zip(legend.texts, ("Money offer")):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_shock.svg'), transparent=True)


fig, ax = plt.subplots(figsize=(1.6, 1.6))

corr_ant['Y_true'] = corr_ant['Y_true'].astype(int)
corr_ant['pvp_cv_cosine_z'] = zscore(corr_ant['pvp_cv_cosine'])
corr_ant['mvp_cv_cosine_z'] = zscore(corr_ant['mvp_cv_cosine'])

corr_plot = corr_ant.melt(id_vars=['Y_true', 'type'], value_vars=['pvp_cv_cosine_z', 'mvp_cv_cosine_z'])

ax = sns.pointplot(corr_plot['Y_true'], corr_plot['value'], hue=corr_plot['variable'],
                   scale=0.4, ci=68, errwidth=1, palette=[colp, colm], legend=False)

# Add labels
ax.set_ylim(-0.6, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Anticipated intensity", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})

ax.legend([],[], frameon=False)
# # Set legend
# legend = ax.legend(fontsize=legendfontsize, frameon=False)
# for t, l in zip(legend.texts, ['', ""]):
#     t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)
# ax._legend.remove()
fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_anticipation.svg'), transparent=True)



fig, ax = plt.subplots(figsize=(1.6, 1.6))

corr_ant['Y_true'] = corr_ant['Y_true'].astype(int)
corr_ant['pvp_cv_cosine_z'] = zscore(corr_ant['pvp_cv_cosine'])
corr_ant['mvp_cv_cosine_z'] = zscore(corr_ant['mvp_cv_cosine'])

corr_plot = corr_ant.melt(id_vars=['Y_true', 'type'], value_vars=['pvp_cv_cosine_z', 'mvp_cv_cosine_z'])

ax = sns.pointplot(corr_plot['Y_true'], corr_plot['value'], hue=corr_plot['variable'],
                   scale=0.4, ci=68, errwidth=1, palette=[colp, colm], legend=False)

# Add labels
ax.set_ylim(-0.6, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Anticipated intensity", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})

ax.legend([],[], frameon=False)
# # Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False, handletextpad=0.1)
for t, l in zip(legend.texts, ['Pain pattern', "Money pattern"]):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)
# ax._legend.remove()
fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_legend.svg'), transparent=True)



fig, ax = plt.subplots(figsize=(1.6, 1.6))

corr_shock['Y_true'] = corr_shock['Y_true'].astype(int)
corr_shock['pvp_cv_cosine_z'] = zscore(corr_shock['pvp_cv_cosine'])
corr_shock['mvp_cv_cosine_z'] = zscore(corr_shock['mvp_cv_cosine'])

corr_plot = corr_shock.melt(id_vars=['Y_true', 'type'], value_vars=['pvp_cv_cosine_z', 'mvp_cv_cosine_z'])

ax = sns.pointplot(corr_plot['Y_true'], corr_plot['value'], hue=corr_plot['variable'],
                   scale=0.4, ci=68, errwidth=1, palette=[colp, colm], legend=False)

# Add labels
ax.set_ylim(-0.6, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Shock intensity", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})

ax.legend([],[], frameon=False)
# # Set legend
# legend = ax.legend(fontsize=legendfontsize, frameon=False)
# for t, l in zip(legend.texts, ['', ""]):
#     t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)
# ax._legend.remove()
fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_shock.svg'), transparent=True)


######################################################################
# Regression plots for cross validation by participant y_fit
#####################################################################

corr_pain = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                            'painlevel_cvstats.csv'))
# Add the intercept to the data
fig1, ax1 = plt.subplots(figsize=(1.6, 2.2))
ax1.set_ylim((1, 10))
ax1.set_xlim((1, 10))

colors1 = plt.cm.Blues(np.linspace(0, 1, len(set(corr_pain.subject_id))))
for idx, s in enumerate(list(set(corr_pain.subject_id))):
    corr_sub = corr_pain[corr_pain.subject_id == s]
    sns.regplot(data=corr_sub, x='Y_true', y='Y_pred',
                ci=None, scatter=False, color=colors1[idx],
                ax=ax1, line_kws={'linewidth':1})

ax1.set_xlabel('Pain offer level', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_ylabel('Cross-validated prediction',
               {'fontsize': labelfontsize})
ax1.set_xticks(range(1, 11))
ax1.set_xticklabels(range(1, 11))
ax1.set_yticks(range(1, 11))
ax1.set_yticklabels(range(1, 11))
ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'maps_correlation_plots_bysub_pain.svg'),
             transparent=True)

#######################################################################
# Plot distribution of slopes across participants
#######################################################################

slopes_pain = []
for s in corr_pain.subject_id.unique():
    psub = corr_pain[(corr_pain.subject_id == s)]
    slope = stats.pearsonr(stats.zscore(psub['Y_true']),
                            stats.zscore(psub['Y_pred']))[0]
    slopes_pain.append(slope)

# Add the intercept to the data
fig1, ax1 = plt.subplots(figsize=(0.6, 2.2))


pt.half_violinplot(y=slopes_pain, inner=None,
                    jitter=True, lwidth=0, width=0.6,
                    offset=0.17, cut=1, ax=ax1,
                    color=colp,
                    linewidth=1, alpha=0.6, zorder=19)
sns.stripplot(y=slopes_pain,
                jitter=0.08, ax=ax1,
                color=colp,
                linewidth=1, alpha=0.6, zorder=1)
sns.boxplot(y=slopes_pain, whis=np.inf, linewidth=1, ax=ax1,
            width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
            whiskerprops={'zorder': 10, 'alpha': 1},
            color=colp,
            medianprops={'zorder': 11, 'alpha': 0.5})
ax1.set_ylabel('Pearson r', fontsize=labelfontsize, labelpad=0.7)
ax1.tick_params(axis='y', labelsize=ticksfontsize)
ax1.set_xticks([], [])

ax1.axhline(0, linestyle='--', color='k')
fig.tight_layout()
fig1.savefig(opj(outpath, 'slopes_bysub_pain.svg'), transparent=True)

#######################################################################
# ROC class results
#######################################################################

roc_res = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level/painlevel_roc_results.csv'))

fig, ax = plt.subplots(figsize=(1.6, 2.2))
x = np.arange(3)
width = 0.5
ax.bar(x=x[0]+width,
       height=roc_res["accuracy"][0],
       width=width, label=roc_res['comparison'][0].replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[0],
       yerr=roc_res['accuracy_se'][0])
ax.bar(x=x[1],
       height=roc_res["accuracy"][1],
       width=width, label=roc_res['comparison'][1].replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[1],
       yerr=roc_res['accuracy_se'][1])
ax.bar(x=x[2]-width,
       height=roc_res["accuracy"][2],
       width=width, label=roc_res['comparison'][2].replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[2],
       yerr=roc_res['accuracy_se'][2])

ax.set_xlabel('')
ax.set_ylim((0.2, 1))
ax.set_xticks([])
# ax.axhline(0.6315, linestyle='--', color='r')
ax.axhline(0.5, linestyle='--', color='k')

ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.set_ylabel('Accuracy', fontsize=labelfontsize)
ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
# ax.axhline(y=0.5, linestyle='--', color='k')
ax.set_xlabel('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='x', labelsize=labelfontsize)
ax.set_ylabel('Accuracy', fontsize=labelfontsize)
ax.legend(fontsize=legendfontsize, loc='lower left', title='Pain offer',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_painonly.svg'),
            dpi=600, transparent=True)


#######################################################################
# Plots for incremental prediction
#######################################################################

corr_pain = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                            'painlevel_cvstats.csv'))

corr_money = pd.read_csv(opj(basepath, 'derivatives/mvpa/money_offer_level',
                             'moneylevel_cvstats.csv'))

# Pain sl scores disttibution
par_scores = np.load(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                         'painlevel_parcels_scores.npy'))


fig = plt.figure(figsize=(2.5,1.5))
ax = fig.add_subplot(1, 1, 1)

sns.distplot(par_scores, hist=True, kde=True, color=colp,
             ax=ax, norm_hist=True)

ax.set_ylabel('Density', fontsize=labelfontsize)
# ax.set_xlim((-0.2, 0.6))

# ax.set_xlim((-0.05, 0.35))
# ax.set_title('Money', fontsize=titlefontsize)
ax.set_xlabel("Pearson r", fontsize=labelfontsize)
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.axvline(np.max(par_scores), ymin=0, ymax=0.8, color=colp, linestyle='--',
           label="Top parcel")
# ax.axvline(thr, ymin=0, ymax=0.8, color=colp, linestyle='-.',
#            label="Top 1%")
ax.axvline(corr_pain['r_xval'][0], ymin=0, ymax=0.8, color=colp,
           label="Whole-brain")
ax.legend(ncol=3, frameon=False, fontsize=legendfontsize, loc=(0, 1),
          handletextpad=0.3, handlelength=2, columnspacing=0.5)
fig.tight_layout()
fig.savefig(opj(outpath, 'dist_par_pain.svg'), dpi=600, transparent=True)


# Money scores distribution
par_scores = np.load(opj(basepath, 'derivatives/mvpa/money_offer_level',
                         'moneylevel_parcels_scores.npy'))

fig = plt.figure(figsize=(2.5,1.5))
ax = fig.add_subplot(1, 1, 1)

sns.distplot(par_scores, hist=True, kde=True, color=colm,
             ax=ax, norm_hist=True)

ax.set_ylabel('Density', fontsize=labelfontsize)
# ax.set_xlim((-0.2, 0.6))
# ax.set_title('Money', fontsize=titlefontsize)
ax.set_xlabel("Pearson r", fontsize=labelfontsize)
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.axvline(np.max(par_scores), ymin=0, ymax=0.8, color=colm, linestyle='--',
           label="Top parcel")
# ax.axvline(thr, ymin=0, ymax=0.8, color=colp, linestyle='-.',
#            label="Top 1%")
ax.axvline(corr_money['r_xval'][0], ymin=0, ymax=0.8, color=colm,
           label="Whole-brain")
ax.legend(ncol=3, frameon=False, fontsize=legendfontsize, loc=(0, 1),
          handletextpad=0.3, handlelength=2, columnspacing=0.5)
fig.tight_layout()
fig.savefig(opj(outpath, 'dist_par_money.svg'), dpi=600, transparent=True)

# Plot with incremental parcels

# Load the parcels and nvoxels

pain_incr = np.load(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                         'painlevel_incremental_parcel_scores.npy'))
pain_nvox = np.load(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                         'painlevel_nvox.npy'))

mon_incr = np.load(opj(basepath, 'derivatives/mvpa/money_offer_level',
                         'moneylevel_incremental_parcel_scores.npy'))
mon_nvox = np.load(opj(basepath, 'derivatives/mvpa/money_offer_level',
                         'moneylevel_nvox.npy'))

fig, ax = plt.subplots(figsize=(2.5, 1.5))
sns.regplot(mon_nvox, y=mon_incr, order=3, color=colm, scatter_kws={'s':1}, line_kws={'linewidth':1})
sns.regplot(pain_nvox, y=pain_incr, order=3, color=colp, scatter_kws={'s':1}, line_kws={'linewidth':1})

ymin, ymax = ax.get_ybound()
ax.set_xlim((0, 100000))
ax.set_ylim((0.2, 0.65))
px = pain_nvox[pain_incr == np.max(pain_incr)]
py = np.max(pain_incr)
l1 = mlines.Line2D([px, px], [0, py], label='Max score pain', color=colp, linestyle='dashed', linewidth=1)

mx = mon_nvox[mon_incr == np.max(mon_incr)]
my = np.max(mon_incr)
l2 = mlines.Line2D([mx, mx], [0, my], label='Max score money', color=colm, linestyle='dashed', linewidth=1)
ax.add_line(l1)
ax.add_line(l2)

ax.set_ylabel("Pearson's r", fontsize=labelfontsize)
ax.set_xlabel("Number of voxels (x 10$^3$)", fontsize=labelfontsize)
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.legend(fontsize=legendfontsize-1, frameon=False, handletextpad=0.3)
ax.ticklabel_format(axis="x", style="sci", scilimits=(3,3), useMathText=True)
ax.xaxis.get_offset_text().set_fontsize(0)

fig.tight_layout()
fig.savefig(opj(outpath, 'incremental_par_money.svg'), dpi=600, transparent=True)


# Load parcels labels sorted

pmapfdr = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                           'painlevel_bootz_fdr05.nii')), group_mask)

pmapfdr_masked = np.where(pmapfdr != 0, 6, 0)

par_labels_pain = np.load(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                            'painlevel_parcels_labels.npy'))
# Remove those under the max plotted
par_labels_pain = par_labels_pain[:np.argmax(pain_incr)+1]

from nilearn.image import resample_to_img
# Load the parcellation
parcel = apply_mask(resample_to_img(load_img(opj(basepath, 'external',
                                                 'CANlab_2018_combined_atlas_2mm.nii.gz')), group_mask, interpolation='nearest'), group_mask)
parcel_masked = parcel.copy()
# Zero out the parcels not included
for par in np.unique(parcel):
    if not any((par == par_labels_pain)):
        parcel_masked[parcel == par] = 0
    else:
        parcel_masked[parcel == par] = -5

parcel_top = np.where(parcel == par_labels_pain[0], 1, 0)


combined_par = parcel_masked + pmapfdr_masked

# PLot slices
to_plot = {'x': [-8, -4],
           'y': [12, 8],
           'z': [-10, 66]}
cmap = colors.ListedColormap([colp], N=1)
top = colors.ListedColormap([cold], N=1)
fdr = colors.ListedColormap([cole], N=1)

for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        disp = plot_stat_map(unmask(parcel_masked, group_mask), cmap=cmap,
                             colorbar=False,
                             bg_img=bgimg,
                             dim=-0.3,
                             black_bg=False,
                             display_mode=axis,
                             axes=ax,
                             vmax=6,
                             cut_coords=(c,),
                             alpha=1,
                             annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        disp.add_overlay(unmask(pmapfdr_masked, group_mask), cmap=fdr)
        disp.add_overlay(unmask(parcel_top, group_mask), cmap=top)

        fig.savefig(opj(outpath, 'parcel_pain_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)

par_labels_money = np.load(opj(basepath, 'derivatives/mvpa/money_offer_level',
                            'moneylevel_parcels_labels.npy'))
# Remove those under the max plotted
par_labels_money = par_labels_money[:np.argmax(mon_incr)]

from nilearn.image import resample_to_img
# Load the parcellation
parcel = apply_mask(resample_to_img(load_img(opj(basepath, 'external',
                                                 'CANlab_2018_combined_atlas_2mm.nii.gz')), group_mask, interpolation='nearest'), group_mask)
parcel_masked = parcel.copy()

mmapfdr = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                           'moneylevel_bootz_fdr05.nii')), group_mask)

mmapfdr_masked = np.where(mmapfdr != 0, 6, 0)

# Zero out the parcels not included
for par in np.unique(parcel):
    if not any((par == par_labels_money)):
        parcel_masked[parcel == par] = 0
    else:
        parcel_masked[parcel == par] = -5

parcel_top = np.where(parcel == par_labels_money[0], 1, 0)

combined_par = parcel_masked + pmapfdr_masked

# PLot slices
to_plot = {'x': [10],
           'y': [14, 8],
           'z': [-8]}
cmap = colors.ListedColormap([colm], N=1)

for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        disp = plot_stat_map(unmask(parcel_masked, group_mask), cmap=cmap,
                             colorbar=False,
                             bg_img=bgimg,
                             dim=-0.3,
                             black_bg=False,
                             display_mode=axis,
                             axes=ax,
                             vmax=6,
                             cut_coords=(c,),
                             alpha=1,
                             annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        disp.add_overlay(unmask(mmapfdr_masked, group_mask), cmap=fdr)
        disp.add_overlay(unmask(parcel_top, group_mask), cmap=top)

        fig.savefig(opj(outpath, 'parcel_money_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)

###################################################################
# Calculate ROC accuracy between levels for money images
###################################################################

comparisons = [[5, 1], [1, 10], [5, 10]]
# comparisons = [[1, 5]]

roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(corr_money.pvp_cv_cosine[corr_money.Y_true.isin(c)])
    outcome = list(corr_money.Y_true[corr_money.Y_true.isin(c)])
    outcome = np.where(np.asarray(outcome) == c[1], 1, 0).astype(bool)

    subs = np.asarray(corr_money.subject_id[corr_money.Y_true.isin(c)], dtype=object)
    subs = [int(s[4:]) for s in corr_money.subject_id[corr_money.Y_true.isin(c)]]
    subs = np.asarray(subs, dtype=object)


    roc = Roc(input_values=inputs,
              binary_outcome=outcome)

    roc.calculate()
    roc.summary()
    roc_results['accuracy'].append(roc.accuracy)
    roc_results['accuracy_se'].append(roc.accuracy_se)
    roc_results['accuracy_p'].append(roc.accuracy_p)

    plt.figure()
# Save
# Save
pd.DataFrame(roc_results).to_csv(opj(outpath, 'pvp_on_money' + '_roc_results.csv'))

#######################################################################
# ROC results
#######################################################################

roc_res = pd.DataFrame(roc_results)
fig, ax = plt.subplots(figsize=(1.6, 2.2))
x = np.arange(3)
width = 0.5
ax.bar(x=x[0]+width,
       height=roc_res["accuracy"][0],
       width=width, label=str(roc_res['comparison'][0]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[0],
       yerr=roc_res['accuracy_se'][0])
ax.bar(x=x[1],
       height=roc_res["accuracy"][1],
       width=width, label=str(roc_res['comparison'][1]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[1],
       yerr=roc_res['accuracy_se'][1])
ax.bar(x=x[2]-width,
       height=roc_res["accuracy"][2],
       width=width, label=str(roc_res['comparison'][2]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[2],
       yerr=roc_res['accuracy_se'][2])


ax.set_xlabel('')
ax.set_ylim((0.1, 0.8))
ax.set_xticks([])
# ax.axhline(0.6315, linestyle='--', color='r')
ax.axhline(0.5, linestyle='--', color='k')
# ax.spines['left'].set_color('w')

ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='y', labelsize=ticksfontsize, color='k')
ax.set_ylabel('Accuracy', fontsize=labelfontsize, color='k')
# [t.set_color('w') for t in ax.yaxis.get_ticklabels()]

ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
# ax.axhline(y=0.5, linestyle='--', color='k')
ax.set_xlabel('')

ax.legend(fontsize=legendfontsize, loc='lower left', title='Money offer',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_money_usingpvp.svg'),
            dpi=600, transparent=True)



###################################################################
# Calculate ROC accuracy between levels for shock images
###################################################################

comparisons = [[1, 10], [1, 5], [5, 10]]
# comparisons = [[1, 5]]

corr_shock.Y_true.value_counts()
roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(corr_shock.pvp_cv_cosine[corr_shock.Y_true.isin(c)])
    outcome = list(corr_shock.Y_true[corr_shock.Y_true.isin(c)])
    outcome = np.where(np.asarray(outcome) == c[1], 1, 0).astype(bool)

    subs = np.asarray(corr_shock.subject_id[corr_shock.Y_true.isin(c)], dtype=object)
    subs = [int(s[4:]) for s in corr_shock.subject_id[corr_shock.Y_true.isin(c)]]
    subs = np.asarray(subs, dtype=object)


    roc = Roc(input_values=inputs,
              binary_outcome=outcome)

    roc.calculate()
    # Use balanced accuracy
    roc_results['accuracy'].append(np.mean([roc.sensitivity, roc.specificity]))
    roc_results['accuracy_se'].append(roc.accuracy_se)
    # Binomial test using balanced accuracy
    roc_results['accuracy_p'].append(binom_test(x=[np.round(len(outcome)*np.mean(roc_results['accuracy'][-1])),
              len(outcome)-np.round(len(outcome)*np.mean(roc_results['accuracy'][-1]))] ))

# Save
# Save
pd.DataFrame(roc_results).to_csv(opj(outpath, 'pvp_on_shock' + '_roc_results.csv'))

#######################################################################
# ROC results
#######################################################################

roc_res = pd.DataFrame(roc_results)
fig, ax = plt.subplots(figsize=(1.6, 1.6))
x = np.arange(3)
width = 0.5
ax.bar(x=x[0]+width,
       height=roc_res["accuracy"][0],
       width=width, label=str(roc_res['comparison'][0]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[0],
       yerr=roc_res['accuracy_se'][0])
ax.bar(x=x[1],
       height=roc_res["accuracy"][1],
       width=width, label=str(roc_res['comparison'][1]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[1],
       yerr=roc_res['accuracy_se'][1])
ax.bar(x=x[2]-width,
       height=roc_res["accuracy"][2],
       width=width, label=str(roc_res['comparison'][2]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[2],
       yerr=roc_res['accuracy_se'][2])



ax.set_xlabel('')
ax.set_ylim((0.1, 0.8))
ax.set_xticks([])
# ax.axhline(0.6315, linestyle='--', color='r')
ax.axhline(0.5, linestyle='--', color='k')
# ax.spines['left'].set_color('w')

ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='y', labelsize=ticksfontsize, color='k')
ax.set_ylabel('Balanced accuracy', fontsize=labelfontsize, color='k')
# [t.set_color('w') for t in ax.yaxis.get_ticklabels()]

ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
# ax.axhline(y=0.5, linestyle='--', color='k')
ax.set_xlabel('')

ax.legend(fontsize=legendfontsize, loc='lower left', title='Shock intensity',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_shock_usingpvp.svg'),
            dpi=600, transparent=True)



###################################################################
# Calculate ROC accuracy between levels for shock images
###################################################################

comparisons = [[1, 10], [1, 5], [5, 10]]
# comparisons = [[1, 5]]

corr_shock.Y_true.value_counts()
roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(corr_shock.mvp_cv_cosine[corr_shock.Y_true.isin(c)])
    outcome = list(corr_shock.Y_true[corr_shock.Y_true.isin(c)])
    outcome = np.where(np.asarray(outcome) == c[1], 1, 0).astype(bool)

    subs = np.asarray(corr_shock.subject_id[corr_shock.Y_true.isin(c)], dtype=object)
    subs = [int(s[4:]) for s in corr_shock.subject_id[corr_shock.Y_true.isin(c)]]
    subs = np.asarray(subs, dtype=object)


    roc = Roc(input_values=inputs,
              binary_outcome=outcome)

    roc.calculate()
    # Use balanced accuracy
    roc_results['accuracy'].append(np.mean([roc.sensitivity, roc.specificity]))
    roc_results['accuracy_se'].append(roc.accuracy_se)
    # Binomial test using balanced accuracy
    roc_results['accuracy_p'].append(binom_test(x=[np.round(len(outcome)*np.mean(roc_results['accuracy'][-1])),
              len(outcome)-np.round(len(outcome)*np.mean(roc_results['accuracy'][-1]))] ))

# Save
# Save
pd.DataFrame(roc_results).to_csv(opj(outpath, 'mvp_on_shock' + '_roc_results.csv'))

#######################################################################
# ROC results
#######################################################################

roc_res = pd.DataFrame(roc_results)
fig, ax = plt.subplots(figsize=(1.6, 1.6))
x = np.arange(3)
width = 0.5
ax.bar(x=x[0]+width,
       height=roc_res["accuracy"][0],
       width=width, label=str(roc_res['comparison'][0]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Greens_r", n_colors=3)[0],
       yerr=roc_res['accuracy_se'][0])
ax.bar(x=x[1],
       height=roc_res["accuracy"][1],
       width=width, label=str(roc_res['comparison'][1]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Greens_r", n_colors=3)[1],
       yerr=roc_res['accuracy_se'][1])
ax.bar(x=x[2]-width,
       height=roc_res["accuracy"][2],
       width=width, label=str(roc_res['comparison'][2]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Greens_r", n_colors=3)[2],
       yerr=roc_res['accuracy_se'][2])



ax.set_xlabel('')
ax.set_ylim((0.1, 0.8))
ax.set_xticks([])
# ax.axhline(0.6315, linestyle='--', color='r')
ax.axhline(0.5, linestyle='--', color='k')
# ax.spines['left'].set_color('w')

ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='y', labelsize=ticksfontsize, color='k')
ax.set_ylabel('Balanced accuracy', fontsize=labelfontsize, color='k')
# [t.set_color('w') for t in ax.yaxis.get_ticklabels()]

ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
# ax.axhline(y=0.5, linestyle='--', color='k')
ax.set_xlabel('')

ax.legend(fontsize=legendfontsize, loc='lower left', title='Shock intensity',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_shock_usingmvp.svg'),
            dpi=600, transparent=True)

###################################################################
# Calculate ROC accuracy between levels for anticipation
###################################################################

comparisons = [[1, 5], [1, 10], [5, 10]]
# comparisons = [[1, 5]]

roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(corr_ant.pvp_cv_cosine[corr_ant.Y_true.isin(c)])
    outcome = list(corr_ant.Y_true[corr_ant.Y_true.isin(c)])
    outcome = np.where(np.asarray(outcome) == c[1], 1, 0).astype(bool)

    subs = np.asarray(corr_ant.subject_id[corr_ant.Y_true.isin(c)], dtype=object)
    subs = [int(s[4:]) for s in corr_ant.subject_id[corr_ant.Y_true.isin(c)]]
    subs = np.asarray(subs, dtype=object)


    roc = Roc(input_values=inputs,
              binary_outcome=outcome)


    roc.calculate()
    # Use balanced accuracy
    roc_results['accuracy'].append(np.mean([roc.sensitivity, roc.specificity]))
    roc_results['accuracy_se'].append(roc.accuracy_se)
    # Binomial test using balanced accuracy
    roc_results['accuracy_p'].append(binom_test(x=[np.round(len(outcome)*np.mean(roc_results['accuracy'][-1])),
              len(outcome)-np.round(len(outcome)*np.mean(roc_results['accuracy'][-1]))] ))

# Save
# Save
pd.DataFrame(roc_results).to_csv(opj(outpath, 'pvp_on_ant' + '_roc_results.csv'))

#######################################################################
# ROC results
#######################################################################

roc_res = pd.DataFrame(roc_results)
fig, ax = plt.subplots(figsize=(1.6, 1.6))
x = np.arange(3)
width = 0.5
ax.bar(x=x[0]+width,
       height=roc_res["accuracy"][0],
       width=width, label=str(roc_res['comparison'][0]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[0],
       yerr=roc_res['accuracy_se'][0])
ax.bar(x=x[1],
       height=roc_res["accuracy"][1],
       width=width, label=str(roc_res['comparison'][1]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[1],
       yerr=roc_res['accuracy_se'][1])
ax.bar(x=x[2]-width,
       height=roc_res["accuracy"][2],
       width=width, label=str(roc_res['comparison'][2]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Blues_r", n_colors=3)[2],
       yerr=roc_res['accuracy_se'][2])



ax.set_xlabel('')
ax.set_ylim((0.1, 0.8))
ax.set_xticks([])
# ax.axhline(0.6315, linestyle='--', color='r')
ax.axhline(0.5, linestyle='--', color='k')
# ax.spines['left'].set_color('w')

ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='y', labelsize=ticksfontsize, color='k')
ax.set_ylabel('Balanced accuracy', fontsize=labelfontsize, color='k')
# [t.set_color('w') for t in ax.yaxis.get_ticklabels()]

ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
# ax.axhline(y=0.5, linestyle='--', color='k')
ax.set_xlabel('')

ax.legend(fontsize=legendfontsize, loc='lower left', title='Anticipated level',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_ant_usingpvp.svg'),
            dpi=600, transparent=True)


###################################################################
# Calculate ROC accuracy between levels for anticipation
###################################################################

comparisons = [[1, 5], [1, 10], [5, 10]]
# comparisons = [[1, 5]]

roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(corr_ant.mvp_cv_cosine[corr_ant.Y_true.isin(c)])
    outcome = list(corr_ant.Y_true[corr_ant.Y_true.isin(c)])
    outcome = np.where(np.asarray(outcome) == c[1], 1, 0).astype(bool)

    subs = np.asarray(corr_ant.subject_id[corr_ant.Y_true.isin(c)], dtype=object)
    subs = [int(s[4:]) for s in corr_ant.subject_id[corr_ant.Y_true.isin(c)]]
    subs = np.asarray(subs, dtype=object)


    roc = Roc(input_values=inputs,
              binary_outcome=outcome)


    roc.calculate()
    # Use balanced accuracy
    roc_results['accuracy'].append(np.mean([roc.sensitivity, roc.specificity]))
    roc_results['accuracy_se'].append(roc.accuracy_se)
    # Binomial test using balanced accuracy
    roc_results['accuracy_p'].append(binom_test(x=[np.round(len(outcome)*np.mean(roc_results['accuracy'][-1])),
              len(outcome)-np.round(len(outcome)*np.mean(roc_results['accuracy'][-1]))] ))

    plt.figure()
# Save
# Save
pd.DataFrame(roc_results).to_csv(opj(outpath, 'pvp_on_ant' + '_roc_results.csv'))

#######################################################################
# ROC results
#######################################################################

roc_res = pd.DataFrame(roc_results)
fig, ax = plt.subplots(figsize=(1.6, 1.6))
x = np.arange(3)
width = 0.5
ax.bar(x=x[0]+width,
       height=roc_res["accuracy"][0],
       width=width, label=str(roc_res['comparison'][0]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Greens_r", n_colors=3)[0],
       yerr=roc_res['accuracy_se'][0])
ax.bar(x=x[1],
       height=roc_res["accuracy"][1],
       width=width, label=str(roc_res['comparison'][1]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Greens_r", n_colors=3)[1],
       yerr=roc_res['accuracy_se'][1])
ax.bar(x=x[2]-width,
       height=roc_res["accuracy"][2],
       width=width, label=str(roc_res['comparison'][2]).replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Greens_r", n_colors=3)[2],
       yerr=roc_res['accuracy_se'][2])



ax.set_xlabel('')
ax.set_ylim((0.1, 0.8))
ax.set_xticks([])
# ax.axhline(0.6315, linestyle='--', color='r')
ax.axhline(0.5, linestyle='--', color='k')
# ax.spines['left'].set_color('w')

ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='y', labelsize=ticksfontsize, color='k')
ax.set_ylabel('Balanced accuracy', fontsize=labelfontsize, color='k')
# [t.set_color('w') for t in ax.yaxis.get_ticklabels()]

ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
# ax.axhline(y=0.5, linestyle='--', color='k')
ax.set_xlabel('')

ax.legend(fontsize=legendfontsize, loc='lower left', title='Anticipated level',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_ant_usingmvp.svg'),
            dpi=600, transparent=True)


###################################################################
# Correlation with striatum networks
###################################################################
# Load atlas


atlas_map = load_img(opj(basepath, 'external/fsl_striatum_atlas',
                         'striatum-con-label-thr25-7sub-2mm.nii.gz'))
# Resample to group_mask
atlas_map = resample_to_img(atlas_map, group_mask, interpolation='nearest')
atlas_mask = new_img_like(atlas_map, np.where(atlas_map.get_data() != 0, 1, 0))
atlas_map = apply_mask(atlas_map, atlas_mask)
# Load PVP - MVP
pvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                   'painlevel_weightsxvalmean.nii')), atlas_mask)
mvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                   'moneylevel_weightsxvalmean.nii')), atlas_mask)

view_img(unmask(pvp, atlas_mask))
view_img(unmask(mvp, atlas_mask))


# Correlate with each network
corr_coef_pain, corr_coef_money, corr_coef_diff = [], [], []

for net in np.unique(atlas_map):
        network_mask = np.where(atlas_map == net, 1, 0)
        corr_coef_pain.append(pearsonr(network_mask, pvp)[0])
        corr_coef_money.append(pearsonr(network_mask, mvp)[0])

labels_7 = ['Limbic', 'Executive', 'Rostral\nmotor', 'Caudal\nmotor',
             'Parietal', 'Occipital', 'Temporal']

df_wedge = pd.DataFrame(data={'Networks': labels_7,
                              'Pain pattern': corr_coef_pain,
                              'Money pattern': corr_coef_money})
df_wedge_m = df_wedge.melt(id_vars=['Networks'])

N = len(df_wedge)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
radii = np.array(df_wedge['Pain pattern'])
radii2 = np.array(df_wedge['Money pattern'])
width = 2 * np.pi / N

fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(111, projection='polar')
for t, r1, r2 in zip(theta, radii, radii2):
  if r1 < 0:
    color = '#003d5c'
  else:
    color = colp
  r1 = np.abs(r1)
  bars = ax.bar(t, r1, width=width, bottom=0.0, alpha=1,
                color=color, linewidth=0, edgecolor='k')
  if r2 < 0:
    color = '#00805c'
  else:
    color = colm
  r2 = np.abs(r2)
  bars = ax.bar(t-0.225, r2, width=width/2, bottom=0.0, alpha=1,
                color=color, linewidth=0, edgecolor='k')

ax.yaxis.set_ticks([0, 0.45])
ax.yaxis.set_ticklabels(['', ''])
ax.xaxis.set_ticks(theta+width/2)
ax.yaxis.set_ticks([0.45])
ax.yaxis.set_ticklabels(["|r| = 0.45"], size=ticksfontsize-2)
# ax.tick_params(axis = "x", pad = 30)
# ax.xaxis.grid(False)
ax.xaxis.set_ticklabels([])


counter = 0
for t, o in zip(theta, np.ones_like(theta)):
  deg_rot = np.rad2deg(t)
  print(deg_rot)
  if deg_rot > 180:
    deg_rot =  360- deg_rot
  ax.text(t, 0.59, labels_7[counter], horizontalalignment='center',
          verticalalignment='center', rotation=0, fontsize=labelfontsize-2)
  counter += 1

fig.tight_layout()
fig.savefig(opj(outpath, 'striatum_7network_pearson_wedge_pvp.svg'), dpi=600,
            transparent=True)



###################################################################
# With cortical networks
###################################################################
from nilearn.image import resample_to_img
atlas_map = load_img(opj(basepath, 'external/Yeo_JNeurophysiol11_MNI152',
                         'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'))
group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))

# Resample to group_mask
atlas_map = resample_to_img(atlas_map, group_mask, interpolation='nearest')

# Remove voxles out of group mas
atlas_map = unmask(apply_mask(atlas_map, group_mask), group_mask)
# Mask
atlas_mask = new_img_like(atlas_map, np.where(atlas_map.get_data() != 0, 1, 0))
atlas_map = np.squeeze(apply_mask(atlas_map, atlas_mask))
# Load PVP - MVP
pvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                   'painlevel_weightsxvalmean.nii')), atlas_mask)
mvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                   'moneylevel_weightsxvalmean.nii')), atlas_mask)



# Correlate with each network
corr_coef_pain, corr_coef_money, corr_coef_diff = [], [], []

for net in np.unique(atlas_map):
        network_mask = np.where(atlas_map == net, 1, 0)
        corr_coef_pain.append(pearsonr(network_mask, pvp)[0])
        corr_coef_money.append(pearsonr(network_mask, mvp)[0])

labels_7 = ['Visual', 'Somato-\nmotor', 'Dorsal\natt.', 'Ventral\natt.',
             'Limbic', 'Frontoparietal', 'Default']

df_wedge = pd.DataFrame(data={'Networks': labels_7,
                              'Pain pattern': corr_coef_pain,
                              'Money pattern': corr_coef_money})
df_wedge_m = df_wedge.melt(id_vars=['Networks'])
df_wedge_m = df_wedge_m.sort_values(['Networks'])



N = len(df_wedge)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
radii = np.array(df_wedge['Pain pattern'])
radii2 = np.array(df_wedge['Money pattern'])
width = 2 * np.pi / N


fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(111, projection='polar')
for t, r1, r2 in zip(theta, radii, radii2):
  if r1 < 0:
    color = '#003d5c'
  else:
    color = colp
  r1 = np.abs(r1)
  bars = ax.bar(t, r1, width=width, bottom=0.0, alpha=1,
                color=color, linewidth=0, edgecolor='k')
  if r2 < 0:
    color = '#00805c'
  else:
    color = colm

ax.yaxis.set_ticks([0, 0.1])
ax.yaxis.set_ticklabels(['', ''])
ax.xaxis.set_ticks(theta+width/2)
ax.yaxis.set_ticks([0.1])
ax.yaxis.set_ticklabels(["|r| = 0.10"], size=ticksfontsize-2)

ax.xaxis.set_ticklabels([])


counter = 0
for t, o in zip(theta, np.ones_like(theta)):
  deg_rot = np.rad2deg(t)
  print(deg_rot)
  if deg_rot > 180:
    deg_rot =  360- deg_rot

  offset = 0.13
  ax.text(t, offset, labels_7[counter], horizontalalignment='center',
          verticalalignment='center', rotation=0, fontsize=labelfontsize-2)
  counter += 1

fig.tight_layout()
fig.savefig(opj(outpath, 'cortical_7network_pearson_wedge_pvp.svg'), dpi=600,
            transparent=True)


# Redo just for legend
N = len(df_wedge)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
radii = np.array(df_wedge['Pain pattern'])
radii2 = np.array(df_wedge['Money pattern'])
width = 2 * np.pi / N

fig = plt.figure(figsize=(1.5, 1.5))
ax = fig.add_subplot(111, projection='polar')
for t, r1, r2 in zip(theta, radii, radii2):
  if r1 < 0:
    color = '#003d5c'
  else:
    color = colp
  r1 = np.abs(r1)
  bars = ax.bar(t, r1, width=width, bottom=0.0, alpha=1,
                color=color, linewidth=0, edgecolor='k')
  if r2 < 0:
    color = '#00805c'
  else:
    color = colm
  r2 = np.abs(r2)

ax.yaxis.set_ticks([0, 0.1])
ax.yaxis.set_ticklabels(['', ''])
ax.xaxis.set_ticks(theta+width/2)
ax.xaxis.set_ticklabels([])

counter = 0
for t, o in zip(theta, np.ones_like(theta)):
  deg_rot = np.rad2deg(t)
  print(deg_rot)
  if deg_rot > 180:
    deg_rot =  360- deg_rot
  ax.text(t, 0.13, labels_7[counter], horizontalalignment='center',
          verticalalignment='center', rotation=0, fontsize=labelfontsize-3)
  counter += 1

legend_elements = [Patch(facecolor=colp, edgecolor=colp,
                         label='Positive'),
                   Patch(facecolor='#003d5c', edgecolor='#003d5c',
                         label='Negative'),]
ax.legend(handles=legend_elements, loc=(0.6, 1.05), ncol=1, handlelength=1.5,
          labels=['Positive', 'Negative'], handletextpad=0.2,
          fontsize=labelfontsize-3, columnspacing=0.4, frameon=True)

# ax.set_title('Cortical networks', fontsize=titlefontsize-3)
fig.savefig(opj(outpath, 'cortical_7network_pearson_wedge_pvp_legend.svg'), dpi=600,
            transparent=True)


#######################################################################
# Plot univariate
#######################################################################
import nilearn.plotting as npl
def plot_multi_supp(mapunt, map001, mapfdr05, outfigpath, display_mode='x',
                    cut_range=range(-60, 70, 10), title_offset=0.1,
                    figsize=(20, 10), fileext='.svg',
                    vmax=5,
                    cmap_label='Z score',
                    bgimg=None,
                    name3='FDR',
                    title='somemap', cmap='Specral_r'):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(3, 1, 1,)
    disp = npl.plot_stat_map(mapunt,
                         display_mode=display_mode,
                         cut_coords=cut_range,
                         annotate=False,
                         vmax=vmax,
                         cmap=cmap,
                         black_bg=False,
                         bg_img=bgimg,
                         draw_cross=False, colorbar=True,
                         axes=ax)
    disp.title(title + ' - Unthresholded', size=25, y=1.00 + title_offset,
            bgcolor="w", color='k')
    disp.annotate(left_right=False, size=15)
    lab = disp._colorbar_ax.get_yticklabels()
    disp._colorbar_ax.set_yticklabels(lab, fontsize=15)
    disp._colorbar_ax.set_ylabel(cmap_label, rotation=90, fontsize=20,
                                labelpad=10)
    ax = fig.add_subplot(3, 1, 2)
    disp = npl.plot_stat_map(map001,
                         display_mode=display_mode,
                         cut_coords=cut_range,
                         annotate=False,
                         vmax=vmax,
                         cmap=cmap,
                         bg_img=bgimg,
                         black_bg=False,
                         colorbar=False,
                         axes=ax)
    disp.title(title + ' - p < 0.001 uncorrected', size=25,
               y=1.00 + title_offset,
               bgcolor="w", color='k')
    disp.annotate(left_right=False, size=15)
    ax = fig.add_subplot(3, 1, 3)
    disp = npl.plot_stat_map(mapfdr05,
                         display_mode=display_mode,
                         cut_coords=cut_range,
                         annotate=False,
                         vmax=vmax,
                         cmap=cmap,
                         bg_img=bgimg,
                         black_bg=False,
                         draw_cross=False,
                         colorbar=False,
                         axes=ax)
    disp.title(title + ' - ' + name3 + ' p < 0.05', size=25,
               y=1.00 + title_offset,
               bgcolor="w", color='k')
    disp.annotate(left_right=False, size=15)

    fig.savefig(opj(outfigpath, title + '_variousthresh' + fileext ),
                dpi=600, bbox_inches='tight')


map_unthr = painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                           'painlevel_univariate_unthresholded.nii'))
map_001 = painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                           'painlevel_univariate_unc001.nii'))
map_fwe = painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                           'painlevel_univariate_fwe05.nii'))
thr = np.max(apply_mask(map_unthr, group_mask))

plot_multi_supp(map_unthr,
                map_001,
                map_fwe,
                display_mode='z',
                outfigpath=outpath, cut_range=range(-40, 80, 10),
                title_offset=0.18, figsize=(20, 9),
                fileext='.png', bgimg=bgimg,
                vmax=thr,
                cmap_label='T-value',
                name3='FWE',
                title='Parametric effect of pain offer level',
                cmap=npl.cm.cold_hot)
