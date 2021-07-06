from typing import Collection
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
cole = current_palette[5]

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
shockmapfdr = load_img(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                           'shocklevel_bootz_fdr05.nii'))
# PLot slices
view_img(shockmapfdr)
to_plot = {'x': [-8, -14, 14, 20],
           'y': [24],
           'z': [-4, 6]}

cmap = plotting.cm.cold_hot
for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        disp = plot_stat_map(shockmapfdr, cmap=cmap, colorbar=False,
                        bg_img=bgimg,
                        dim=-0.3,
                        black_bg=False,
                        display_mode=axis,
                        axes=ax,
                        vmax=9,
                        cut_coords=(c,),
                        alpha=1,
                        annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        fig.savefig(opj(outpath, 'sip_fdr05_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)


# Plot a last random one to get the colorbar
fig, ax = plt.subplots(figsize=(1.5, 1.5))
thr = np.min(np.abs(shockmapfdr.get_data()[shockmapfdr.get_data() != 0]))
disp = plot_stat_map(shockmapfdr,
                        cmap=cmap, colorbar=True,
                bg_img=bgimg,
                dim=-0.3,
                black_bg=False,
                symmetric_cbar=True,
                display_mode='x',
                threshold=thr,
                axes=ax,
                vmax=9,
                cut_coords=(0,),
                alpha=1,
                annotate=False)
disp.annotate(size=ticksfontsize, left_right=False)
disp._colorbar_ax.set_ylabel('Z score', rotation=90, fontsize=labelfontsize,
                             labelpad=5)
lab = disp._colorbar_ax.get_yticklabels()
disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)

fig.savefig(opj(outpath, 'sip_slices_slicescbar.svg'), dpi=600,
        bbox_inches='tight', transparent=True)


###################################################################
# y_pred plot
###################################################################

corr_shock = pd.read_csv(opj(basepath, 'derivatives/mvpa/shock_intensity_level/shocklevel_cvstats.csv'))

fig, ax = plt.subplots(figsize=(1.6, 2.2))

from scipy.stats import zscore
corr_shock['yfit_xval_z'] = zscore(corr_shock['Y_pred'])
corr_shock['Y_true'] = corr_shock['Y_true'].astype(int)

ax = sns.pointplot(y="Y_pred", x='Y_true',
                   data=corr_shock,
                   scale=0.4, ci=68, errwidth=1, color=cold)  # 68% CI are SEM
# Add labels
ax.set_title("",
             {'fontsize': titlefontsize})
ax.set_xlabel("Shock intensity", {'fontsize': labelfontsize})
ax.set_ylabel("Cross-validated prediction",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False)
for t, l in zip(legend.texts, ("Pain offer", "Money offer")):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_shock_lineplot.svg'), transparent=True)

###################################################################
# y_pred plot with money and pain
###################################################################

corr_pain = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                             'painlevel_cvstats.csv'))
corr_money = pd.read_csv(opj(basepath,
                             'derivatives/mvpa/money_offer_level',
                             'moneylevel_cvstats.csv'))

corr_money['sip_cv_cosine_z'] = zscore(corr_money['sip_cv_cosine'])
corr_pain['sip_cv_cosine_z'] = zscore(corr_pain['sip_cv_cosine'])
corr_shock['sip_cv_cosine_z'] = zscore(corr_shock['sip_cv_cosine'])

corr_money['type'] = 'money'
corr_pain['type'] = "pain"
corr_shock['type'] = 'shock'


corr_all = pd.concat([corr_pain, corr_money, corr_shock]).melt(id_vars=['Y_true', 'type'], value_vars=['sip_cv_cosine_z'])
corr_all['sip_cv_cosine_z'] = zscore(corr_all['value'])
corr_all['Y_true'] = corr_all['Y_true'].astype(int)

fig, ax = plt.subplots(figsize=(2, 2.2))


ax = sns.pointplot('Y_true', 'value', hue='type', data=corr_all,
                   palette=[colp, colm, cold], markers=['o', '^', 'D'],
                   scale=0.4, ci=68, errwidth=1, label='Pain offer')

# Add labels
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Level", {'fontsize': labelfontsize})
ax.set_ylabel("Cross-validated pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False)
for t, l in zip(legend.texts, ("Pain offer", "Money offer", 'Shock\nintensity')):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_shockpainmoney.svg'), transparent=True)


###################################################################
# PLot NPS/SIIPS to pain
###################################################################

corr_pain = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                             'painlevel_cvstats.csv'))
corr_money = pd.read_csv(opj(basepath,
                             'derivatives/mvpa/money_offer_level',
                             'moneylevel_cvstats.csv'))

corr_money['type'] = 'money'
corr_pain['type'] = "pain"
corr_shock['type'] = 'shock'

corr_shock['nps_cosine_z'] = zscore(corr_shock['nps_cosine'])
corr_shock['siips_cosine_z'] = zscore(corr_shock['siips_cosine'])

corr_all = corr_shock.melt(id_vars=['Y_true'], value_vars=['nps_cosine_z', 'siips_cosine_z'])


fig, ax = plt.subplots(figsize=(2, 2.2))


ax = sns.pointplot('Y_true', 'value', hue='variable', data=corr_all,
                   palette=[colc, cole, cold], markers=['o', '^', 'D'],
                   scale=0.4, ci=68, errwidth=1, label='Pain offer')

# Add labels
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Shock intensity", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False)
for t, l in zip(legend.texts, ("NPS", "SIIPS")):
    t.set_text(l)
ax.tick_params('both', labelsize=ticksfontsize)

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_shocknpssiips.svg'), transparent=True)


######################################################################
# Regression plots for cross validation by participant y_fit
#####################################################################

corr_shock= pd.read_csv(opj(basepath,
                             'derivatives/mvpa/shock_intensity_level/shocklevel_cvstats.csv'))
# Add the intercept to the data
fig1, ax1 = plt.subplots(figsize=(1.6, 2.2))
ax1.set_ylim((1, 10))
ax1.set_xlim((1, 10))

colors1 = plt.cm.Oranges(np.linspace(0, 1, len(set(corr_shock.subject_id))))
for idx, s in enumerate(list(set(corr_shock.subject_id))):
    corr_sub = corr_shock[corr_shock.subject_id == s]
    sns.regplot(data=corr_sub, x='Y_true', y='Y_pred',
                ci=None, scatter=False, color=colors1[idx],
                ax=ax1, line_kws={'linewidth':1})

ax1.set_xlabel('Shock intensity', {'fontsize': labelfontsize})

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
fig1.savefig(opj(outpath, 'maps_correlation_plots_bysub_shock.svg'),
             transparent=True)

#######################################################################
# Plot distribution of slopes across participants
#######################################################################

slopes_pain = []
for s in corr_shock.subject_id.unique():
    psub = corr_shock[(corr_shock.subject_id == s)]
    slope = stats.pearsonr(stats.zscore(psub['Y_true']),
                            stats.zscore(psub['Y_pred']))[0]
    slopes_pain.append(slope)

# Add the intercept to the data
fig1, ax1 = plt.subplots(figsize=(0.6, 2.2))


pt.half_violinplot(y=slopes_pain, inner=None,
                    jitter=True, lwidth=0, width=0.6,
                    offset=0.17, cut=1, ax=ax1,
                    color=cold,
                    linewidth=1, alpha=0.6, zorder=19)
sns.stripplot(y=slopes_pain,
                jitter=0.08, ax=ax1,
                color=cold,
                linewidth=1, alpha=0.6, zorder=1)
sns.boxplot(y=slopes_pain, whis=np.inf, linewidth=1, ax=ax1,
            width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
            whiskerprops={'zorder': 10, 'alpha': 1},
            color=cold,
            medianprops={'zorder': 11, 'alpha': 0.5})
ax1.set_ylabel('Correlation coefficient', fontsize=labelfontsize, labelpad=0.7)
ax1.tick_params(axis='y', labelsize=ticksfontsize)
ax1.set_xticks([], [])
ax1.axhline(0, linestyle='--', color='k')
fig.tight_layout()
fig1.savefig(opj(outpath, 'slopes_bysub_shock.svg'), transparent=True)


#######################################################################
# ROC results
#######################################################################

roc_res = pd.read_csv(opj(basepath, 'derivatives/mvpa/shock_intensity_level/shocklevel_roc_results.csv'))

fig, ax = plt.subplots(figsize=(1.6, 2.2))
x = np.arange(3)
width = 0.5
ax.bar(x=x[0]+width,
       height=roc_res["accuracy"][0],
       width=width, label=roc_res['comparison'][0].replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Oranges_r", n_colors=3)[0],
       yerr=roc_res['accuracy_se'][0])
ax.bar(x=x[1],
       height=roc_res["accuracy"][1],
       width=width, label=roc_res['comparison'][1].replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Oranges_r", n_colors=3)[1],
       yerr=roc_res['accuracy_se'][1])
ax.bar(x=x[2]-width,
       height=roc_res["accuracy"][2],
       width=width, label=roc_res['comparison'][2].replace('[', '').replace(']', '').replace(',', ' vs '),
       color=sns.color_palette("Oranges_r", n_colors=3)[2],
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
ax.legend(fontsize=legendfontsize, loc='lower left', title='Shock intensity',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_shockonly.svg'),
            dpi=600, transparent=True)


####################################################
# Calculate ROC accuracy between levels for pain images
###################################################################

comparisons = [[1, 5], [1, 10], [5, 10]]
# comparisons = [[1, 5]]

roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(corr_pain.sip_cv_cosine[corr_money.Y_true.isin(c)])
    outcome = list(corr_pain.Y_true[corr_pain.Y_true.isin(c)])
    outcome = np.where(np.asarray(outcome) == c[1], 1, 0).astype(bool)

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

    plt.figure()
# Save
# Save
pd.DataFrame(roc_results).to_csv(opj(outpath, 'sip_on_pain' + '_roc_results.csv'))


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
ax.set_ylim((0.2, 1))
ax.set_xticks([])
# ax.axhline(0.6315, linestyle='--', color='r')
ax.axhline(0.5, linestyle='--', color='k')
ax.spines['left'].set_color('w')

ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='y', labelsize=ticksfontsize, color='w')
ax.set_ylabel('Accuracy', fontsize=labelfontsize, color='w')
ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
# ax.axhline(y=0.5, linestyle='--', color='k')
ax.set_xlabel('')
[t.set_color('w') for t in ax.yaxis.get_ticklabels()]

ax.legend(fontsize=legendfontsize, loc='lower left', title='Pain offer',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_pain_usingsip.svg'),
            dpi=600, transparent=True)
