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
from nltools.analysis import Roc
from scipy.stats import zscore
from nilearn.image import resample_to_img
from matplotlib.patches import Patch


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

###################################################################
# Slices plot
###################################################################

# Load corrected map
moneymapfdr = load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                           'moneylevel_bootz_fdr05.nii'))
view_img(moneymapfdr)
# PLot slices
to_plot = {'x': [-46, -8, 8, 4,  6, 48],
           'y': [12],
           'z': [-8, -2]}


cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1, 1))
        disp = plot_stat_map(moneymapfdr, cmap=cmap, colorbar=False,
                        bg_img=bgimg,
                        dim=-0.3,
                        black_bg=False,
                        display_mode=axis,
                        axes=ax,
                        vmax=8,
                        cut_coords=(c,),
                        alpha=1,
                        annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        fig.savefig(opj(outpath, 'mvp_fdr05_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)


# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1, 1))
thr = np.min(np.abs(moneymapfdr.get_data()[moneymapfdr.get_data() != 0]))
disp = plot_stat_map(moneymapfdr,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=True,
                display_mode='x',
                threshold=thr,
                axes=ax,
                vmax=8,
                cut_coords=(0,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Z score', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)
disp._colorbar_ax.yaxis.set_tick_params(pad=-0.5)

fig.savefig(opj(outpath, 'mvp_slices_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)


################################################################
# Slices plot in striatum ROI
###################################################################

# Load corrected map
money_unthr = load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                           'moneylevel_bootz.nii'))


atlas_map = load_img(opj(basepath, 'external/fsl_striatum_atlas',
                         'striatum-con-label-thr25-7sub-2mm.nii.gz'))

# Resample to group_mask
atlas_map = resample_to_img(atlas_map, group_mask, interpolation='nearest')
atlas_mask = new_img_like(atlas_map, np.where(atlas_map.get_data() != 0, 1, 0))
money_unthr_str = unmask(apply_mask(money_unthr, atlas_mask), atlas_mask)

# PLot slices
to_plot = {
            # 'x': [],
           'y': [6, 8, 10, 12, 14],
        #    'z': []
           }

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1, 1))
        disp = plot_stat_map(money_unthr_str, cmap=cmap, colorbar=False,
                        bg_img=bgimg,
                        dim=-0.3,
                        black_bg=False,
                        display_mode=axis,
                        axes=ax,
                        vmax=7,
                        cut_coords=(c,),
                        alpha=1,
                        annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        # disp.add_overlay(nsynth_mask_pm, cmap='binary_r', **{'alpha': 0.7})
        fig.savefig(opj(outpath, 'mlevel_uncSTRROI_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)

# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1, 1))
# thr = np.min(np.abs(painvsmoney.get_data()[painvsmoney.get_data() != 0]))
disp = plot_stat_map(money_unthr_str,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=True,
                display_mode='y',
                axes=ax,
                # threshold=0,
                vmax=7,
                cut_coords=(12,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Z score', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)

fig.savefig(opj(outpath, 'mlevel_uncSTRROI_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)

###################################################################
# y_pred plot
###################################################################

corr_money = pd.read_csv(opj(basepath, 'derivatives/mvpa',
                             'money_offer_level/moneylevel_cvstats.csv'))

fig, ax = plt.subplots(figsize=(1.5, 1.5))

from scipy.stats import zscore
corr_money['yfit_xval_z'] = zscore(corr_money['Y_pred'])
corr_money['Y_true'] = corr_money['Y_true'].astype(int)

corr_money['mvp_cv_cosine_z'] = zscore(corr_money['mvp_cv_cosine'])
ax = sns.pointplot(y="mvp_cv_cosine_z", x='Y_true',
                   data=corr_money,
                   scale=0.4, ci=68, errwidth=1, color=colm)  # 68% CI are SEM
# Add labels
ax.set_title("",
             {'fontsize': titlefontsize})
ax.set_xlabel("Money offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False)
for t, l in zip(legend.texts, ("Pain offer", "Money offer")):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_money.svg'), transparent=True)


###################################################################
# y_pred plot
###################################################################

corr_money = pd.read_csv(opj(basepath, 'derivatives/mvpa',
                             'money_offer_level/moneylevel_cvstats.csv'))

fig, ax = plt.subplots(figsize=(1.5, 1.5))

from scipy.stats import zscore
corr_money['yfit_xval_z'] = zscore(corr_money['Y_pred'])
corr_money['Y_true'] = corr_money['Y_true'].astype(int)

corr_money['mvp_cv_cosine_z'] = zscore(corr_money['mvp_cv_cosine'])
ax = sns.pointplot(y="mvp_cv_cosine_z", x='Y_true',
                   data=corr_money,
                   scale=0.4, ci=68, errwidth=1, color=colm)  # 68% CI are SEM
# Add labels
ax.set_title("",
             {'fontsize': titlefontsize})
ax.set_xlabel("Money offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False)
for t, l in zip(legend.texts, ("Pain offer", "Money offer")):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_money.svg'), transparent=True)


###################################################################
# y_pred plot with money
###################################################################

corr_pain = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                             'painlevel_cvstats.csv'))

fig, ax = plt.subplots(figsize=(1.5, 1.5))

corr_pain['Y_true'] = corr_pain['Y_true'].astype(int)


ax = sns.pointplot(corr_pain['Y_true'], zscore(corr_pain['mvp_cv_cosine']),
                   color=colm, label='Pain pattern',
                   scale=0.4, ci=68, errwidth=1)

# Add labels
ax.set_ylim(-0.7, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Pain offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(labels=['Money\npattern'],
                   fontsize=legendfontsize, frameon=False, loc=(0, 0.7))
# for t, l in zip(legend.texts, ("Money offer")):
#     t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_mvp_pain.svg'), transparent=True)


###################################################################
# y_pred plot with money and shock
###################################################################

corr_pain = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                             'painlevel_cvstats.csv'))

corr_shock = pd.read_csv(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                             'shocklevel_cvstats.csv'))


fig, ax = plt.subplots(figsize=(1.5, 1.5))

corr_money['Y_true'] = corr_money['Y_true'].astype(int)

ax = sns.pointplot(corr_pain['Y_true'], zscore(corr_pain['mvp_cv_cosine']),
                   color=colp,
                   scale=0.4, ci=68, errwidth=1)
ax = sns.pointplot(corr_money['Y_true'], zscore(corr_money['mvp_cv_cosine']),
                   color=colm,
                   scale=0.4, ci=68, errwidth=1)
ax = sns.pointplot(corr_money['Y_true'], zscore(corr_shock['mvp_cv_cosine']),
                   color=cold,
                   scale=0.4, ci=68, errwidth=1, label="Money offers")
# Add labels
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Level", {'fontsize': labelfontsize})
ax.set_ylabel("Cross-validated pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False)
for t, l in zip(legend.texts, ("Pain offer", "Money offer")):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)
fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_moneywithpainwshock.svg'), transparent=True)


######################################################################
# Regression plots for cross validation by participant y_fit
#####################################################################

corr_money = pd.read_csv(opj(basepath,
                             'derivatives/mvpa/money_offer_level/moneylevel_cvstats.csv'))
# Add the intercept to the data
fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
ax1.set_ylim((1, 10))
ax1.set_xlim((1, 10))

colors1 = plt.cm.Greens(np.linspace(0, 1, len(set(corr_money.subject_id))))
for idx, s in enumerate(list(set(corr_money.subject_id))):
    corr_sub = corr_money[corr_money.subject_id == s]
    sns.regplot(data=corr_sub, x='Y_true', y='Y_pred',
                ci=None, scatter=False, color=colors1[idx],
                ax=ax1, line_kws={'linewidth':1})

ax1.set_xlabel('Money offer level', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_ylabel('Predicted level',
               {'fontsize': labelfontsize})
ax1.set_xticks(range(1, 11))
ax1.set_xticklabels(range(1, 11))
ax1.set_yticks(range(1, 11))
ax1.set_yticklabels(range(1, 11))
ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'maps_correlation_plots_bysub_money.svg'),
             transparent=True)

#######################################################################
# Plot distribution of slopes across participants
#######################################################################

slopes_pain = []
for s in corr_money.subject_id.unique():
    psub = corr_money[(corr_money.subject_id == s)]
    slope = stats.pearsonr(stats.zscore(psub['Y_true']),
                            stats.zscore(psub['Y_pred']))[0]
    slopes_pain.append(slope)
    print(s)
    print(slope)
# Add the intercept to the data
fig1, ax1 = plt.subplots(figsize=(0.6, 1.5))


pt.half_violinplot(y=slopes_pain, inner=None,
                    jitter=True, lwidth=0, width=0.6,
                    offset=0.17, cut=1, ax=ax1,
                    color=colm,
                    linewidth=1, alpha=0.6, zorder=19)
sns.stripplot(y=slopes_pain,
                jitter=0.08, ax=ax1,
                color=colm,
                linewidth=1, alpha=0.6, zorder=1)
sns.boxplot(y=slopes_pain, whis=np.inf, linewidth=1, ax=ax1,
            width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
            whiskerprops={'zorder': 10, 'alpha': 1},
            color=colm,
            medianprops={'zorder': 11, 'alpha': 0.5})
ax1.set_ylabel('Pearson r', fontsize=labelfontsize, labelpad=0.7)
ax1.tick_params(axis='y', labelsize=ticksfontsize)
ax1.set_xticks([], [])
ax1.axhline(0, linestyle='--', color='k')
fig.tight_layout()
fig1.savefig(opj(outpath, 'slopes_bysub_money.svg'), transparent=True)


#######################################################################
# ROC results
#######################################################################

roc_res = pd.read_csv(opj(basepath, 'derivatives/mvpa/money_offer_level/moneylevel_roc_results.csv'))

fig, ax = plt.subplots(figsize=(1.5, 1.5))
x = np.arange(3)
width = 0.5
ax.bar(x=x[0]+width,
       height=roc_res["accuracy"][0],
       width=width, label=roc_res['comparison'][0].replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Greens_r", n_colors=3)[0],
       yerr=roc_res['accuracy_se'][0])
ax.bar(x=x[1],
       height=roc_res["accuracy"][1],
       width=width, label=roc_res['comparison'][1].replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Greens_r", n_colors=3)[1],
       yerr=roc_res['accuracy_se'][1])
ax.bar(x=x[2]-width,
       height=roc_res["accuracy"][2],
       width=width, label=roc_res['comparison'][2].replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Greens_r", n_colors=3)[2],
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
ax.legend(fontsize=legendfontsize, loc='lower left', title='Money offer',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_moneyonly.svg'),
            dpi=600, transparent=True)



###################################################################
# Calculate ROC accuracy between levels for pain images
###################################################################

comparisons = [[1, 5], [1, 10], [5, 10]]

roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(corr_pain.mvp_cv_cosine[corr_pain.Y_true.isin(c)])
    outcome = list(corr_pain.Y_true[corr_pain.Y_true.isin(c)])
    outcome = np.where(np.asarray(outcome) == c[1], 0, 1).astype(bool)

    subs = np.asarray(corr_pain.subject_id[corr_pain.Y_true.isin(c)], dtype=object)
    subs = [int(s[4:]) for s in corr_pain.subject_id[corr_pain.Y_true.isin(c)]]
    subs = np.asarray(subs, dtype=object)


    roc = Roc(input_values=inputs,
              binary_outcome=outcome)

    roc.calculate()
    roc.summary()
    roc_results['accuracy'].append(roc.accuracy)
    roc_results['accuracy_se'].append(roc.accuracy_se)
    roc_results['accuracy_p'].append(roc.accuracy_p)

# Save
pd.DataFrame(roc_results).to_csv(opj(outpath, 'mvp_on_pain' + '_roc_results.csv'))

#######################################################################
# ROC results
#######################################################################

roc_res = pd.DataFrame(roc_results)
fig, ax = plt.subplots(figsize=(1.5, 1.5))
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
ax.spines['left'].set_color('k')

# ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='y', labelsize=ticksfontsize, color='k')
ax.set_ylabel('Accuracy', fontsize=labelfontsize, color='k')
ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
# ax.axhline(y=0.5, linestyle='--', color='k')
ax.set_xlabel('')
[t.set_color('k') for t in ax.yaxis.get_ticklabels()]

ax.legend(fontsize=legendfontsize, loc='lower left', title='Pain offer',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_pain_usingmvp.svg'),
            dpi=600, transparent=True)

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


map_unthr = painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                           'moneylevel_univariate_unthresholded.nii'))
map_001 = painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                           'moneylevel_univariate_unc001.nii'))
map_fwe = painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                           'moneylevel_univariate_fwe05.nii'))
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
                title='Parametric effect of money offer level',
                cmap=npl.cm.cold_hot)


###################################################################
# Other patterns
###################################################################
current_palette = sns.color_palette('colorblind', 10)

# Shock pattern, NPS, SIIPS, PINES on pain and money
fig, ax = plt.subplots(figsize=(1.6, 1.6))

corr_money['Y_true'] = corr_money['Y_true'].astype(int)
corr_money['sip_cv_cosine_z'] = zscore(corr_money['sip_cv_cosine'])

corr_plot = corr_money.melt(id_vars=['Y_true', 'type'], value_vars=['sip_cv_cosine_z'])

ax = sns.pointplot(corr_plot['Y_true'], corr_plot['value'], hue=corr_plot['variable'],
                   scale=0.4, ci=68, errwidth=1, palette=['#de8f05'], legend=False)

# Add labels
ax.set_ylim(-0.6, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Money offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})

# # Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False, loc=(0, 0.6),
                   handletextpad=0)
for t, l in zip(legend.texts, ['Shock\nintensity\npattern', ""]):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_sip_onmoney.svg'), transparent=True)

colc = current_palette[3]
cole = current_palette[5]
colf = current_palette[6]

# Shock pattern, NPS, SIIPS, PINES on pain and money
fig, ax = plt.subplots(figsize=(1.6, 1.6))

corr_money['Y_true'] = corr_money['Y_true'].astype(int)
corr_money['nps_cosine_z'] = zscore(corr_money['nps_cosine'])

corr_plot = corr_money.melt(id_vars=['Y_true', 'type'], value_vars=['nps_cosine_z'])

ax = sns.pointplot(corr_plot['Y_true'], corr_plot['value'], hue=corr_plot['variable'],
                   scale=0.4, ci=68, errwidth=1, palette=[colc], legend=False)

# Add labels
ax.set_ylim(-0.6, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Money offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})

# # Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False, loc=(0, 0.8),
                   handletextpad=0)
for t, l in zip(legend.texts, ['NPS', ""]):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_nps_onmoney.svg'), transparent=True)


# Shock pattern, NPS, SIIPS, PINES on pain and money
fig, ax = plt.subplots(figsize=(1.6, 1.6))

corr_money['Y_true'] = corr_money['Y_true'].astype(int)
corr_money['siips_cosine_z'] = zscore(corr_money['siips_cosine'])

corr_plot = corr_money.melt(id_vars=['Y_true', 'type'], value_vars=['siips_cosine_z'])

ax = sns.pointplot(corr_plot['Y_true'], corr_plot['value'], hue=corr_plot['variable'],
                   scale=0.4, ci=68, errwidth=1, palette=[cole], legend=False)

# Add labels
ax.set_ylim(-0.6, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Money offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})

# # Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False, loc=(0, 0.8),
                   handletextpad=0)
for t, l in zip(legend.texts, ['SIIPS', ""]):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_siips_onmoney.svg'), transparent=True)



# Shock pattern, NPS, SIIPS, PINES on pain and money
fig, ax = plt.subplots(figsize=(1.6, 1.6))

corr_money['Y_true'] = corr_money['Y_true'].astype(int)
corr_money['pines_cosine_z'] = zscore(corr_money['pines_cosine'])

corr_plot = corr_money.melt(id_vars=['Y_true', 'type'], value_vars=['pines_cosine_z'])

ax = sns.pointplot(corr_plot['Y_true'], corr_plot['value'], hue=corr_plot['variable'],
                   scale=0.4, ci=68, errwidth=1, palette=[colf], legend=False)

# Add labels
ax.set_ylim(-0.6, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Money offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})

# # Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False, loc=(0, 0.8),
                   handletextpad=0)
for t, l in zip(legend.texts, ['PINES', ""]):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_pines_onmoney.svg'), transparent=True)