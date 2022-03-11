import pandas as pd
import numpy as np
from os.path import join as opj
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, zscore
import matplotlib.pyplot as plt
from nilearn import plotting
import statsmodels.api as sm
import statsmodels.formula.api as smf
import ptitprince as pt

basepath = '/data'
###################################################################
# Plot options
###################################################################
outpath =  opj(basepath, 'derivatives/figures')

# colors
current_palette = sns.color_palette('colorblind', 10)
colp = current_palette[0]
colm = current_palette[2]

cold = current_palette[1]
colc = current_palette[4]
cole = current_palette[8]

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


#######################################################################
# Check relationship between pvp expression and individual differences
# in money pain weights
#######################################################################
# Offer part 1
pvp_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                        'painlevel_cvstats.csv')).groupby('subject_id').mean()
mvp_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/money_offer_level',
                     'moneylevel_cvstats.csv')).groupby('subject_id').mean()

# Correlation pattern expression ~ weight on decision
r_pvp, sig_pvp = pearsonr(pvp_stats['pvp_cv_cosine'], pvp_stats['coef_pain'])
r_mvp, sig_mvp = pearsonr(mvp_stats['mvp_cv_cosine'], mvp_stats['coef_money'])

# Correlation pattern expression ~ weight on rt
r_pvp_rt, sig_pvp_rt = pearsonr(pvp_stats['pvp_cv_cosine'], pvp_stats['coef_pain_rt'])
r_mvp_rt, sig_mvp_rt = pearsonr(mvp_stats['mvp_cv_cosine'], mvp_stats['coef_money_rt'])


# Decision
dec_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                        'decision_stats_out.csv'))

dec_stats_wide = dec_stats.groupby('subject_id').mean()



# Offer 2
r_dec_pvp_rt, sig_dec_pvp_rt = pearsonr(dec_stats_wide['pvp_cv_cosine'],
                                        dec_stats_wide['coef_pain_rt'])
r_dec_mvp_rt, sig_dec_mvp_rt = pearsonr(dec_stats_wide['mvp_cv_cosine'],
                                        dec_stats_wide['coef_money_rt'])

r_dec_pvp, sig_dec_pvp = pearsonr(dec_stats_wide['pvp_cv_cosine'],
                                  dec_stats_wide['coef_pain'])
r_dec_mvp, sig_dec_mvp = pearsonr(dec_stats_wide['mvp_cv_cosine'],
                                  dec_stats_wide['coef_money'])





# Confirm similar with different on presentation order
dec_stats_pfirst = dec_stats[dec_stats['painfirst'] == 1].groupby('subject_id').mean()
dec_stats_mfirst = dec_stats[dec_stats['painfirst'] == 0].groupby('subject_id').mean()

r_dec_pvp_rt_pf, sig_dec_pvp_rt_pf = pearsonr(dec_stats_pfirst['pvp_cv_cosine'],
                                               dec_stats_pfirst['coef_pain_rt'])
r_dec_mvp_rt_pf , sig_dec_mvp_rt_pf  = pearsonr(dec_stats_pfirst['mvp_cv_cosine'],
                                                dec_stats_pfirst['coef_money_rt'])

r_dec_pvp_rt_mf, sig_dec_pvp_rt_mf = pearsonr(dec_stats_mfirst['pvp_cv_cosine'],
                                               dec_stats_mfirst['coef_pain_rt'])
r_dec_mvp_rt_mf, sig_dec_mvp_rt_mf = pearsonr(dec_stats_mfirst['mvp_cv_cosine'],
                                                dec_stats_mfirst['coef_money_rt'])

r_dec_pvp_pf , sig_dec_pvp_pf  = pearsonr(dec_stats_pfirst['pvp_cv_cosine'],
                                               dec_stats_pfirst['coef_pain'])
r_dec_mvp_pf , sig_dec_mvp_pf  = pearsonr(dec_stats_pfirst['mvp_cv_cosine'],
                                                dec_stats_pfirst['coef_money'])

r_dec_pvp_mf, sig_dec_pvp_mf = pearsonr(dec_stats_mfirst['pvp_cv_cosine'],
                                               dec_stats_mfirst['coef_pain'])
r_dec_mvp_mf, sig_dec_mvp_mf = pearsonr(dec_stats_mfirst['mvp_cv_cosine'],
                                                dec_stats_mfirst['coef_money'])





# MAke figures


fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
sns.regplot(pvp_stats['coef_pain'], zscore(pvp_stats['pvp_cv_cosine']), color=colp,
            scatter_kws={"s": 2}, line_kws={'linewidth': 1})
ax1.set_ylabel('PVP avg. similarity', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_xlabel(r'$ \beta_{Pain}$' + ' Choice',
               {'fontsize': labelfontsize})

ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'individual_diff_o1_pvp_bpain.svg'),
             transparent=True, dpi=800)



fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
sns.regplot(mvp_stats['coef_money'], zscore(pvp_stats['mvp_cv_cosine']), color=colm,
            scatter_kws={"s": 2}, line_kws={'linewidth': 1})
ax1.set_ylabel('MVP avg. similarity', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_xlabel(r'$ \beta_{Money}$' + ' Choice',
               {'fontsize': labelfontsize})

ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'individual_diff_o1_mvp_bmoney.svg'),
             transparent=True, dpi=800)



fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
sns.regplot(pvp_stats['coef_pain_rt'], zscore(pvp_stats['pvp_cv_cosine']), color=colp,
            scatter_kws={"s": 2}, line_kws={'linewidth': 1})
ax1.set_ylabel('PVP avg. similarity', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_xlabel(r'$ \beta_{Pain}$' + ' Response time',
               {'fontsize': labelfontsize})

ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'individual_diff_o1_pvp_bpain_rt.svg'),
             transparent=True, dpi=800)



fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
sns.regplot(mvp_stats['coef_money_rt'], zscore(pvp_stats['mvp_cv_cosine']), color=colm,
            scatter_kws={"s": 2}, line_kws={'linewidth': 1})
ax1.set_ylabel('MVP avg. similarity', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_xlabel(r'$ \beta_{Money}$' + ' Response time',
               {'fontsize': labelfontsize})

ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'individual_diff_o1_mvp_bmoney_rt.svg'),
             transparent=True, dpi=800)





# Offer part 2

fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
sns.regplot(dec_stats_wide['coef_pain'], zscore(dec_stats_wide['pvp_cv_cosine']), color=colp,
            scatter_kws={"s": 2}, line_kws={'linewidth': 1})
ax1.set_ylabel('PVP avg. similarity', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_xlabel(r'$ \beta_{Pain}$' + ' Choice',
               {'fontsize': labelfontsize})

ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'individual_diff_o2_pvp_bpain.svg'),
             transparent=True, dpi=800)



fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
sns.regplot(dec_stats_wide['coef_money'], zscore(dec_stats_wide['mvp_cv_cosine']), color=colm,
            scatter_kws={"s": 2}, line_kws={'linewidth': 1})
ax1.set_ylabel('MVP avg. similarity', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_xlabel(r'$ \beta_{Money}$' + ' Choice',
               {'fontsize': labelfontsize})

ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'individual_diff_o2_mvp_bmoney.svg'),
             transparent=True, dpi=800)



fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
sns.regplot(dec_stats_wide['coef_pain_rt'], zscore(dec_stats_wide['pvp_cv_cosine']), color=colp,
            scatter_kws={"s": 2}, line_kws={'linewidth': 1})
ax1.set_ylabel('PVP avg. similarity', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_xlabel(r'$ \beta_{Pain}$' + ' Response time',
               {'fontsize': labelfontsize})

ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'individual_diff_o2_pvp_bpain_rt.svg'),
             transparent=True, dpi=800)



fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
sns.regplot(dec_stats_wide['coef_money_rt'], zscore(dec_stats_wide['mvp_cv_cosine']), color=colm,
            scatter_kws={"s": 2}, line_kws={'linewidth': 1})
ax1.set_ylabel('MVP avg. similarity', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_xlabel(r'$ \beta_{Money}$' + ' Response time',
               {'fontsize': labelfontsize})

ax1.tick_params(axis='both', labelsize=ticksfontsize)

fig1.tight_layout()
fig1.savefig(opj(outpath, 'individual_diff_o2_mvp_bmoney_rt.svg'),
             transparent=True, dpi=800)



#######################################################################
# Check relationship between pvp expression computationnally estimated
# pain value
#######################################################################
pvp_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                        'painlevel_cvstats.csv'))

# Pain value ~ pvp during offer

# Asses with mixed models
md = smf.mixedlm("sv_pain ~ pvp_cv_cosine", pvp_stats, groups=pvp_stats["subject_id"]).fit()
print(md.summary())



corr_subs = []
for s in np.unique(pvp_stats['subject_id']):
    sub_dat = pvp_stats[pvp_stats['subject_id'] == s]
    corr_subs.append(pearsonr(sub_dat['pvp_cv_cosine'], sub_dat['sv_pain'])[0])


# Add the intercept to the data
fig1, ax1 = plt.subplots(figsize=(0.6, 1.5))
pt.half_violinplot(y=corr_subs, inner=None,
                    jitter=True, lwidth=0, width=0.6,
                    offset=0.17, cut=1, ax=ax1,
                    color=colp,
                    linewidth=1, alpha=0.6, zorder=19)
sns.stripplot(y=corr_subs,
                jitter=0.08, ax=ax1,
                color=colp,
                linewidth=1, alpha=0.6, zorder=1)
sns.boxplot(y=corr_subs, whis=np.inf, linewidth=1, ax=ax1,
            width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
            whiskerprops={'zorder': 10, 'alpha': 1},
            color=colp,
            medianprops={'zorder': 11, 'alpha': 0.5})
ax1.set_ylabel('Pearson r', fontsize=labelfontsize, labelpad=0.7)
ax1.tick_params(axis='y', labelsize=ticksfontsize)
ax1.set_xticks([], [])
ax1.set_title('Pain value ~ PVP similarity\n(Offer phase)', fontsize=labelfontsize)

ax1.axhline(0, linestyle='--', color='k')
fig1.tight_layout()
fig1.savefig(opj(outpath, 'slopes_bysub_pred_svpain.svg'), transparent=True)



# Pain value ~ pvp during decision

# Same for decision
dec_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                        'decision_stats_out.csv'))

# Test with mixed model
md_dec = smf.mixedlm("sv_pain ~ pvp_cv_cosine", dec_stats,
                     groups=dec_stats["subject_id"]).fit()
print(md_dec.summary())


corr_subs = []
for s in np.unique(dec_stats['subject_id']):
    sub_dat = dec_stats[dec_stats['subject_id'] == s]
    corr_subs.append(pearsonr(sub_dat['pvp_cv_cosine'], sub_dat['sv_pain'])[0])

# Add the intercept to the data
fig1, ax1 = plt.subplots(figsize=(0.6, 1.5))
pt.half_violinplot(y=corr_subs, inner=None,
                    jitter=True, lwidth=0, width=0.6,
                    offset=0.17, cut=1, ax=ax1,
                    color=colp,
                    linewidth=1, alpha=0.6, zorder=19)
sns.stripplot(y=corr_subs,
                jitter=0.08, ax=ax1,
                color=colp,
                linewidth=1, alpha=0.6, zorder=1)
sns.boxplot(y=corr_subs, whis=np.inf, linewidth=1, ax=ax1,
            width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
            whiskerprops={'zorder': 10, 'alpha': 1},
            color=colp,
            medianprops={'zorder': 11, 'alpha': 0.5})
ax1.set_ylabel('Pearson r', fontsize=labelfontsize, labelpad=0.7)
ax1.tick_params(axis='y', labelsize=ticksfontsize)
ax1.set_title('Pain value ~ PVP similarity\n(Decision phase)', fontsize=labelfontsize)
ax1.set_xticks([], [])
ax1.axhline(0, linestyle='--', color='k')
fig1.tight_layout()
fig1.savefig(opj(outpath, 'slopes_bysub_pred_svpain_decision.svg'), transparent=True)




# Option value ~ pattern difference during decision

dec_stats['pvp_vs_mvp'] = dec_stats['pvp_cv_cosine'] - dec_stats['mvp_cv_cosine']

md_dec = smf.mixedlm("sv_both ~ pvp_vs_mvp", dec_stats,
                     groups=dec_stats["subject_id"]).fit()
print(md_dec.summary())


corr_subs = []
for s in np.unique(dec_stats['subject_id']):
    sub_dat = dec_stats[dec_stats['subject_id'] == s]
    corr_subs.append(pearsonr(sub_dat['pvp_vs_mvp'], sub_dat['sv_both'])[0])

# Add the intercept to the data
fig1, ax1 = plt.subplots(figsize=(0.6, 1.5))
pt.half_violinplot(y=corr_subs, inner=None,
                    jitter=True, lwidth=0, width=0.6,
                    offset=0.17, cut=1, ax=ax1,
                    color=cold,
                    linewidth=1, alpha=0.6, zorder=19)
sns.stripplot(y=corr_subs,
                jitter=0.08, ax=ax1,
                color=cold,
                linewidth=1, alpha=0.6, zorder=1)
sns.boxplot(y=corr_subs, whis=np.inf, linewidth=1, ax=ax1,
            width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
            whiskerprops={'zorder': 10, 'alpha': 1},
            color=cold,
            medianprops={'zorder': 11, 'alpha': 0.5})
ax1.set_ylabel('Pearson r', fontsize=labelfontsize, labelpad=0.7)
ax1.tick_params(axis='y', labelsize=ticksfontsize)
ax1.set_xticks([], [])
ax1.set_title('Option value ~ pattern difference\n(Decision phase)', fontsize=labelfontsize)

ax1.axhline(0, linestyle='--', color='k')
fig1.tight_layout()
fig1.savefig(opj(outpath, 'slopes_bysub_pred_svboth_decision.svg'), transparent=True)


