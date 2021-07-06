import os
from os.path import join as opj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ptitprince as pt
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import zscore
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

###################################################################
# Accuracy by participant raincloud
###################################################################

df = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                     'choice_prediction_results.csv'))

palette = [sns.color_palette('colorblind', 6)[0],
           sns.color_palette('colorblind', 6)[2],
           sns.color_palette('colorblind', 6)[1],
           sns.color_palette('colorblind', 6)[4],
           sns.color_palette('colorblind', 6)[1]]

plt_dat = df.melt(id_vars=['subject_id'], var_name='model',value_name='acc')

plt_dat = plt_dat[plt_dat.model.isin(['both_bal_acc_sub',
                                      'pvp_bal_acc_sub', 'mvp_bal_acc_sub'])]


plt_dat['model'] = pd.Categorical(plt_dat['model'],
                                  ["both_bal_acc_sub",
                                   "pvp_bal_acc_sub", "mvp_bal_acc_sub"])
plt_dat['acc'] = plt_dat['acc'].astype(float)
plt_dat.sort_values("model")

fig, ax = plt.subplots(figsize=(2.5, 2))
for s in plt_dat['subject_id']:
    sub_dat = plt_dat[plt_dat['subject_id'] == s]
    sns.lineplot(x='model', y="acc", data=sub_dat, color='gray', alpha=0.3,
                 linewidth=0.2, sort=False)

pt.half_violinplot(x='model', y="acc", data=plt_dat, inner=None,
                    jitter=True, lwidth=0, width=0.6,
                    offset=0.17, cut=1, ax=ax, palette=palette,
                    order=["pvp_bal_acc_sub", "mvp_bal_acc_sub", "both_bal_acc_sub"],
                    linewidth=1, alpha=0.6, zorder=19)
sns.stripplot(x='model', y="acc", data=plt_dat,
                jitter=0.08, ax=ax, palette=palette, size=2,
                order=["pvp_bal_acc_sub", "mvp_bal_acc_sub", "both_bal_acc_sub"],
                linewidth=0.5, alpha=1, zorder=20)
sns.boxplot(x='model', y="acc", data=plt_dat,
            whis=np.inf, linewidth=1, ax=ax, palette=palette,
            width=0.1, boxprops={"zorder": 10, 'alpha': 0.9},
            whiskerprops={'zorder': 10, 'alpha': 1},
            order=["pvp_bal_acc_sub", "mvp_bal_acc_sub", "both_bal_acc_sub"],
            medianprops={'zorder': 11, 'alpha': 0.9})

ax.axhline(0.5, linestyle='--', color='k')

ax.set_ylim((0.3, 1))
ax.set_ylabel('Balanced accuracy', fontsize=labelfontsize)
ax.set_xlabel('Pattern(s)', fontsize=labelfontsize)
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.set_xticklabels([ "Pain\noffer", 'Money\noffer', 'Money +\n Pain'])

ax.get_xticks()
# Add values across participants
means = [ np.mean(df['pvp_bal_acc_sub']),
         np.mean(df['mvp_bal_acc_sub']), np.mean(df['both_bal_acc_sub'])]
sems = [stats.sem(df['pvp_bal_acc_sub'])*1.96,
        stats.sem(df['mvp_bal_acc_sub'])*1.96, stats.sem(df['both_bal_acc_sub'])*1.96]

for x, y, err, color in zip(ax.get_xticks() - 0.5,
                            means,
                            sems,
                            palette):
    ax.errorbar(x=x, y=y,
                yerr=err, color=color,
                elinewidth=1, capthick=1,
                linestyle='-', fmt='D',
                capsize=1.5, markersize=1)
    ax.set_xlim((-0.75, 2.5))

fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_classacc_o2_rain_painmoney.svg'),
            dpi=600, transparent=True)


###################################################################
# Pattern expression heat maps
###################################################################

df = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision/decision_stats.csv'))
# Remove excluded trials
df = df[(df['vifs'] < 2)
        & (df['duration'] < 5.0)
        & (df['duration'] > 0.2)]

df["pvp_cv_cosine_z"] = zscore(df['pvp_cv_cosine']) 
heat_pvp = df.pivot_table('pvp_cv_cosine_z',
                         'money_rank',
                         'pain_rank',
                        aggfunc=np.mean)
heat_pvp_smooth = gaussian_filter(heat_pvp, sigma=1)



fig, ax = plt.subplots(figsize=(2.5, 2))
ax = sns.heatmap(heat_pvp_smooth, cmap='Blues',
                 vmin=-0.5, vmax=0.5,
                 cbar_kws={"ticks": [-0.5, 0.5]})
ax.set_title("Pain offer pattern similarity", {'fontsize': labelfontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels(range(1, 11), {'fontsize': ticksfontsize}, va='center')
ax.set_xticklabels(range(1, 11), {'fontsize': ticksfontsize}, ha='center')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ticksfontsize)
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_heat_pain.svg'), dpi=600, transparent=True)


df["mvp_cv_cosine_z"] = zscore(df['mvp_cv_cosine']) 
heat_pvp = df.pivot_table('mvp_cv_cosine_z',
                         'money_rank',
                         'pain_rank',
                        aggfunc=np.mean)
heat_pvp_smooth = gaussian_filter(heat_pvp, sigma=1)

current_palette = sns.color_palette()

fig, ax = plt.subplots(figsize=(2.5, 2))
ax = sns.heatmap(heat_pvp_smooth, cmap='Greens',
                 vmin=-0.5, vmax=0.5,
                 cbar_kws={"ticks": [-0.5, 0.5]})
ax.set_title("Money offer pattern similarity", {'fontsize': labelfontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels(range(1, 11), {'fontsize': ticksfontsize}, va='center')
ax.set_xticklabels(range(1, 11), {'fontsize': ticksfontsize}, ha='center')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ticksfontsize)
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_heat_money.svg'), dpi=600, transparent=True)

df["diff_cv_dot_z"] = zscore(df['pvp_cv_cosine']) - zscore(df['mvp_cv_cosine'])
heat_pvp = df.pivot_table('diff_cv_dot_z',
                         'money_rank',
                         'pain_rank',
                        aggfunc=np.mean)
heat_pvp_smooth = gaussian_filter(heat_pvp, sigma=1)


fig, ax = plt.subplots(figsize=(2.5, 2))
ax = sns.heatmap(heat_pvp_smooth, cmap='cividis_r',
                 vmin=-0.6, vmax=0.6,
                 cbar_kws={"ticks": [-0.6, 0.6]})
ax.set_title("Difference (pain - money)", {'fontsize': labelfontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels(range(1, 11), {'fontsize': ticksfontsize}, va='center')
ax.set_xticklabels(range(1, 11), {'fontsize': ticksfontsize}, ha='center')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ticksfontsize)
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_heat_diff.svg'), dpi=600, transparent=True)


#################################################################
# Plot the SVM plane
#################################################################

model_painmoney = np.load(opj(basepath, 'derivatives/mvpa/decision/model_painmoney.npy'),
                          allow_pickle=True).item()


def plot_svm_plane(model, outfigpath, h=0.2, labelfontsize=25,
                   titlefontsize=25, legendfontsize=25):
    from matplotlib.patches import Patch

    X = model['x']
    Y = model['y']

    X = model['clf']['scaler'].fit_transform(X)

    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,
                                                                   y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    fig, ax = plt.subplots(figsize=(2.5, 2))
    # title for the plots
    title = ('SVM decision surface')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, model['clf'], xx, yy,
                  cmap='Spectral', alpha=0.8)
    ax.scatter(X0, X1, c=Y, cmap='Spectral', s=6, edgecolors='k',
               linewidth=0.3,
               rasterized=True)
    cmap = plt.get_cmap('Spectral')
    ax.set_ylabel('Money offer pattern similarity', fontsize=labelfontsize)
    ax.set_xlabel('Pain offer pattern similarity', fontsize=labelfontsize)
    ax.set_xticks(())
    ax.set_yticks(())
    legend_elements = [Patch(facecolor=cmap(0.9), edgecolor='k',
                         label='Accept'),
                       Patch(facecolor=cmap(0), edgecolor='k',
                         label='Reject')]

    ax.legend(handles=legend_elements, fontsize=legendfontsize)
    ax.set_title(title, fontsize=titlefontsize)
    plt.tight_layout()
    plt.savefig(opj(outfigpath, model['name'] + '_svmplane.svg'),
                dpi=600, transparent=True)
    plt.show()

plot_svm_plane(model=model_painmoney, outfigpath=outpath,
               labelfontsize=labelfontsize, titlefontsize=labelfontsize,
               legendfontsize=legendfontsize)




###################################################################
# SAME WITH TRIALS SPLIT ACCORDING TO IF PAIN OR MONEY WAS PRESENTED
# FIRST
###################################################################

# Pain first
###################################################################
# Accuracy by participant raincloud
###################################################################

df = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                     'choice_prediction_results_painfirst.csv'))


palette = [sns.color_palette('colorblind', 6)[0],
           sns.color_palette('colorblind', 6)[2],
           sns.color_palette('colorblind', 6)[1],
           sns.color_palette('colorblind', 6)[4],
           sns.color_palette('colorblind', 6)[1]]

plt_dat = df.melt(id_vars=['subject_id'], var_name='model',value_name='acc')

plt_dat = plt_dat[plt_dat.model.isin(['both_bal_acc_sub',
                                      'pvp_bal_acc_sub', 'mvp_bal_acc_sub'])]


plt_dat['model'] = pd.Categorical(plt_dat['model'],
                                  ["both_bal_acc_sub",
                                   "pvp_bal_acc_sub", "mvp_bal_acc_sub"])
plt_dat['acc'] = plt_dat['acc'].astype(float)
plt_dat.sort_values("model")

fig, ax = plt.subplots(figsize=(2.5, 2))
for s in plt_dat['subject_id']:
    sub_dat = plt_dat[plt_dat['subject_id'] == s]
    sns.lineplot(x='model', y="acc", data=sub_dat, color='gray', alpha=0.3,
                 linewidth=0.2, sort=False)

pt.half_violinplot(x='model', y="acc", data=plt_dat, inner=None,
                    jitter=True, lwidth=0, width=0.6,
                    offset=0.17, cut=1, ax=ax, palette=palette,
                    order=["pvp_bal_acc_sub", "mvp_bal_acc_sub", "both_bal_acc_sub"],
                    linewidth=1, alpha=0.6, zorder=19)
sns.stripplot(x='model', y="acc", data=plt_dat,
                jitter=0.08, ax=ax, palette=palette, size=2,
                order=["pvp_bal_acc_sub", "mvp_bal_acc_sub", "both_bal_acc_sub"],
                linewidth=0.5, alpha=1, zorder=20)
sns.boxplot(x='model', y="acc", data=plt_dat,
            whis=np.inf, linewidth=1, ax=ax, palette=palette,
            width=0.1, boxprops={"zorder": 10, 'alpha': 0.9},
            whiskerprops={'zorder': 10, 'alpha': 1},
            order=["pvp_bal_acc_sub", "mvp_bal_acc_sub", "both_bal_acc_sub"],
            medianprops={'zorder': 11, 'alpha': 0.9})

ax.axhline(0.5, linestyle='--', color='k')

ax.set_ylim((0.3, 1))
ax.set_ylabel('Balanced accuracy', fontsize=labelfontsize)
ax.set_xlabel('Pattern(s)', fontsize=labelfontsize)
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.set_xticklabels([ "Pain\noffer", 'Money\noffer', 'Money +\n Pain'])

ax.get_xticks()
# Add values across participants
means = [ np.mean(df['pvp_bal_acc_sub']),
         np.mean(df['mvp_bal_acc_sub']), np.mean(df['both_bal_acc_sub'])]
sems = [stats.sem(df['pvp_bal_acc_sub'])*1.96,
        stats.sem(df['mvp_bal_acc_sub'])*1.96, stats.sem(df['both_bal_acc_sub'])*1.96]

for x, y, err, color in zip(ax.get_xticks() - 0.5,
                            means,
                            sems,
                            palette):
    ax.errorbar(x=x, y=y,
                yerr=err, color=color,
                elinewidth=1, capthick=1,
                linestyle='-', fmt='D',
                capsize=1.5, markersize=1)
    ax.set_xlim((-0.75, 2.5))

fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_classacc_o2_rain_painmoney_painfirst.svg'),
            dpi=600, transparent=True)



###################################################################
# Pattern expression heat maps
###################################################################

df = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision/decision_stats.csv'))

# Remove excluded trials
df = df[(df['vifs'] < 2)
        & (df['duration'] < 5.0)
        & (df['duration'] > 0.2)
        # Pain first
        & (df['painfirst'] == 1)]

# Read pain-money model
df["pvp_cv_cosine_z"] = zscore(df['pvp_cv_cosine']) 
heat_pvp = df.pivot_table('pvp_cv_cosine_z',
                         'money_rank',
                         'pain_rank',
                        aggfunc=np.mean)
heat_pvp_smooth = gaussian_filter(heat_pvp, sigma=1)


fig, ax = plt.subplots(figsize=(2.5, 2))
ax = sns.heatmap(heat_pvp_smooth, cmap='Blues',
                 vmin=-0.5, vmax=0.5,
                 cbar_kws={"ticks": [-0.5, 0.5]})
ax.set_title("Pain offer pattern similarity", {'fontsize': labelfontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels(range(1, 11), {'fontsize': ticksfontsize}, va='center')
ax.set_xticklabels(range(1, 11), {'fontsize': ticksfontsize}, ha='center')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ticksfontsize)
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_heat_pain_painfirst.svg'), dpi=600, transparent=True)


df["mvp_cv_cosine_z"] = zscore(df['mvp_cv_cosine'])
heat_pvp = df.pivot_table('mvp_cv_cosine_z',
                         'money_rank',
                         'pain_rank',
                        aggfunc=np.mean)
heat_pvp_smooth = gaussian_filter(heat_pvp, sigma=1)



fig, ax = plt.subplots(figsize=(2.5, 2))
ax = sns.heatmap(heat_pvp_smooth, cmap='Greens',
                 vmin=-0.5, vmax=0.5,
                 cbar_kws={"ticks": [-0.5, 0.5]})
ax.set_title("Money offer pattern similarity", {'fontsize': labelfontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels(range(1, 11), {'fontsize': ticksfontsize}, va='center')
ax.set_xticklabels(range(1, 11), {'fontsize': ticksfontsize}, ha='center')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ticksfontsize)
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_heat_money_painfirst.svg'), dpi=600, transparent=True)



df["diff_cv_dot_z"] = zscore(df['pvp_cv_cosine']) - zscore(df['mvp_cv_cosine'])
heat_pvp = df.pivot_table('diff_cv_dot_z',
                         'money_rank',
                         'pain_rank',
                        aggfunc=np.mean)
heat_pvp_smooth = gaussian_filter(heat_pvp, sigma=1)


fig, ax = plt.subplots(figsize=(2.5, 2))
ax = sns.heatmap(heat_pvp_smooth, cmap='cividis_r',
                 vmin=-0.6, vmax=0.6,
                 cbar_kws={"ticks": [-0.6, 0.6]})
ax.set_title("Difference (pain - money)", {'fontsize': labelfontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels(range(1, 11), {'fontsize': ticksfontsize}, va='center')
ax.set_xticklabels(range(1, 11), {'fontsize': ticksfontsize}, ha='center')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ticksfontsize)
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_heat_diff_painfirst.svg'), dpi=600, transparent=True)

#################################################################
# Plot the SVM plane
#################################################################

model_painmoney = np.load(opj(basepath, 'derivatives/mvpa/decision/model_painmoney_painfirst.npy'),
                          allow_pickle=True).item()


plot_svm_plane(model=model_painmoney, outfigpath=outpath,
               labelfontsize=labelfontsize, titlefontsize=labelfontsize,
               legendfontsize=legendfontsize)


# SAME WITH Money FIRST
###################################################################
# Accuracy by participant raincloud
###################################################################

df = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                     'choice_prediction_results_moneyfirst.csv'))


palette = [sns.color_palette('colorblind', 6)[0],
           sns.color_palette('colorblind', 6)[2],
           sns.color_palette('colorblind', 6)[1],
           sns.color_palette('colorblind', 6)[4],
           sns.color_palette('colorblind', 6)[1]]

plt_dat = df.melt(id_vars=['subject_id'], var_name='model',value_name='acc')

plt_dat = plt_dat[plt_dat.model.isin(['both_bal_acc_sub',
                                      'pvp_bal_acc_sub', 'mvp_bal_acc_sub'])]


plt_dat['model'] = pd.Categorical(plt_dat['model'],
                                  ["both_bal_acc_sub",
                                   "pvp_bal_acc_sub", "mvp_bal_acc_sub"])
plt_dat['acc'] = plt_dat['acc'].astype(float)
plt_dat.sort_values("model")

fig, ax = plt.subplots(figsize=(2.5, 2))
for s in plt_dat['subject_id']:
    sub_dat = plt_dat[plt_dat['subject_id'] == s]
    sns.lineplot(x='model', y="acc", data=sub_dat, color='gray', alpha=0.3,
                 linewidth=0.2, sort=False)

pt.half_violinplot(x='model', y="acc", data=plt_dat, inner=None,
                    jitter=True, lwidth=0, width=0.6,
                    offset=0.17, cut=1, ax=ax, palette=palette,
                    order=["pvp_bal_acc_sub", "mvp_bal_acc_sub", "both_bal_acc_sub"],
                    linewidth=1, alpha=0.6, zorder=19)
sns.stripplot(x='model', y="acc", data=plt_dat,
                jitter=0.08, ax=ax, palette=palette, size=2,
                order=["pvp_bal_acc_sub", "mvp_bal_acc_sub", "both_bal_acc_sub"],
                linewidth=0.5, alpha=1, zorder=20)
sns.boxplot(x='model', y="acc", data=plt_dat,
            whis=np.inf, linewidth=1, ax=ax, palette=palette,
            width=0.1, boxprops={"zorder": 10, 'alpha': 0.9},
            whiskerprops={'zorder': 10, 'alpha': 1},
            order=["pvp_bal_acc_sub", "mvp_bal_acc_sub", "both_bal_acc_sub"],
            medianprops={'zorder': 11, 'alpha': 0.9})

ax.axhline(0.5, linestyle='--', color='k')

ax.set_ylim((0.3, 1))
ax.set_ylabel('Balanced accuracy', fontsize=labelfontsize)
ax.set_xlabel('Pattern(s)', fontsize=labelfontsize)
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.set_xticklabels([ "Pain\noffer", 'Money\noffer', 'Money +\n Pain'])

ax.get_xticks()
# Add values across participants
means = [ np.mean(df['pvp_bal_acc_sub']),
         np.mean(df['mvp_bal_acc_sub']), np.mean(df['both_bal_acc_sub'])]
sems = [stats.sem(df['pvp_bal_acc_sub'])*1.96,
        stats.sem(df['mvp_bal_acc_sub'])*1.96, stats.sem(df['both_bal_acc_sub'])*1.96]

for x, y, err, color in zip(ax.get_xticks() - 0.5,
                            means,
                            sems,
                            palette):
    ax.errorbar(x=x, y=y,
                yerr=err, color=color,
                elinewidth=1, capthick=1,
                linestyle='-', fmt='D',
                capsize=1.5, markersize=1)
    ax.set_xlim((-0.75, 2.5))

fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_classacc_o2_rain_painmoney_moneyfirst.svg'),
            dpi=600, transparent=True)



###################################################################
# Pattern expression heat maps
###################################################################

df = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision/decision_stats.csv'))

# Remove excluded trials
df = df[(df['vifs'] < 2)
        & (df['duration'] < 5.0)
        & (df['duration'] > 0.2)
        # Money first
        & (df['painfirst'] == 0)]

# Read pain-money model
df["pvp_cv_cosine_z"] = zscore(df['pvp_cv_cosine']) 
heat_pvp = df.pivot_table('pvp_cv_cosine_z',
                         'money_rank',
                         'pain_rank',
                        aggfunc=np.mean)
heat_pvp_smooth = gaussian_filter(heat_pvp, sigma=1)


current_palette = sns.color_palette()
colp = current_palette[3]
colm = current_palette[2]
fig, ax = plt.subplots(figsize=(2.5, 2))
ax = sns.heatmap(heat_pvp_smooth, cmap='Blues',
                 vmin=-0.5, vmax=0.5,
                 cbar_kws={"ticks": [-0.5, 0.5]})
ax.set_title("Pain offer pattern similarity", {'fontsize': labelfontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels(range(1, 11), {'fontsize': ticksfontsize}, va='center')
ax.set_xticklabels(range(1, 11), {'fontsize': ticksfontsize}, ha='center')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ticksfontsize)
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_heat_pain_moneyfirst.svg'), dpi=600, transparent=True)


df["mvp_cv_cosine_z"] = zscore(df['mvp_cv_cosine'])
heat_pvp = df.pivot_table('mvp_cv_cosine_z',
                         'money_rank',
                         'pain_rank',
                        aggfunc=np.mean)
heat_pvp_smooth = gaussian_filter(heat_pvp, sigma=1)


current_palette = sns.color_palette()
colp = current_palette[3]
colm = current_palette[2]
fig, ax = plt.subplots(figsize=(2.5, 2))
ax = sns.heatmap(heat_pvp_smooth, cmap='Greens',
                 vmin=-0.5, vmax=0.5,
                 cbar_kws={"ticks": [-0.5, 0.5]})
ax.set_title("Money offer pattern similarity", {'fontsize': labelfontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels(range(1, 11), {'fontsize': ticksfontsize}, va='center')
ax.set_xticklabels(range(1, 11), {'fontsize': ticksfontsize}, ha='center')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ticksfontsize)
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_heat_money_moneyfirst.svg'), dpi=600, transparent=True)



df["diff_cv_dot_z"] = zscore(df['pvp_cv_cosine']) - zscore(df['mvp_cv_cosine'])
heat_pvp = df.pivot_table('diff_cv_dot_z',
                         'money_rank',
                         'pain_rank',
                        aggfunc=np.mean)
heat_pvp_smooth = gaussian_filter(heat_pvp, sigma=1)


current_palette = sns.color_palette()
colp = current_palette[3]
colm = current_palette[2]
fig, ax = plt.subplots(figsize=(2.5, 2))
ax = sns.heatmap(heat_pvp_smooth, cmap='cividis_r',
                 vmin=-0.6, vmax=0.6,
                 cbar_kws={"ticks": [-0.6, 0.6]})
ax.set_title("Difference (pain - money)", {'fontsize': labelfontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels(range(1, 11), {'fontsize': ticksfontsize}, va='center')
ax.set_xticklabels(range(1, 11), {'fontsize': ticksfontsize}, ha='center')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ticksfontsize)
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(opj(outpath, 'pattern_heat_diff_moneyfirst.svg'), dpi=600, transparent=True)


#################################################################
# Pointplot patterns O2
#################################################################

fig, ax = plt.subplots(figsize=(1.8, 2))
df['pvp_cv_cosine_z'] = zscore(df['pvp_cv_cosine'])
df['mvp_cv_cosine_z'] = zscore(df['mvp_cv_cosine'])
df['pain_rank'] = df['pain_rank'].astype(int)

palette = [colp, colm]
df_plt = df.melt(id_vars=['pain_rank'],value_vars=['mvp_cv_cosine_z', 'pvp_cv_cosine_z'])
ax = sns.pointplot('pain_rank', 'value', hue='variable', data=df_plt,
                   palette=[colm, colp], markers=['o', '^'],
                   scale=0.4, ci=68, errwidth=1, label='Pain offer')

# Add labels
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Pain offer level", {'fontsize': labelfontsize})
ax.set_ylabel("CV pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False)
for t, l in zip(legend.texts, ("Money pattern", "Pain pattern")):
    t.set_text(l)

ax.tick_params('both', labelsize=ticksfontsize)
ax.set_ylim((-0.4, 0.4))

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o2_pain_rank_moneyfirst.svg'), dpi=600, transparent=True)



fig, ax = plt.subplots(figsize=(2, 2))
df['pvp_cv_cosine_z'] = zscore(df['pvp_cv_cosine'])
df['pain_rank'] = df['pain_rank'].astype(int)

palette = [colp, colm]
ax = sns.pointplot('pain_rank', 'pvp_cv_cosine_z', data=df, color=colp,
                   scale=0.4, ci=68, errwidth=1, label='Pain offer')

# Add labels
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Pain offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
# Set legend
legend = ax.legend(fontsize=legendfontsize, frameon=False)
for t, l in zip(legend.texts, ("Money pattern", "Pain pattern")):
    t.set_text(l)

ax.tick_params('both', labelsize=ticksfontsize)
ax.set_ylim((-0.4, 0.4))

fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o2_pain_rank_ponly_moneyfirst.svg'), dpi=600, transparent=True)


fig, ax = plt.subplots(figsize=(1.8, 2))
df['pvp_cv_cosine_z'] = zscore(df['pvp_cv_cosine'])
df['mvp_cv_cosine_z'] = zscore(df['mvp_cv_cosine'])
df['money_rank'] = df['money_rank'].astype(int)

palette = [colp, colm]
df_plt = df.melt(id_vars=['money_rank'],value_vars=['mvp_cv_cosine_z', 'pvp_cv_cosine_z'])
ax = sns.pointplot('money_rank', 'value', hue='variable', data=df_plt,
                   palette=[colm, colp], markers=['o', '^'],
                   scale=0.4, ci=68, errwidth=1, label='Pain offer',
                   legend=False)
ax.set_ylim((-0.4, 0.4))

# Add labels
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Money offer level", {'fontsize': labelfontsize})
ax.set_ylabel("CV pattern similarity",
              {'fontsize': labelfontsize}, color='w')
# Set legend
# legend = ax.legend(fontsize=legendfontsize, frameon=False)
# for t, l in zip(legend.texts, ("Money offer\npattern", "Pain offer\npattern")):
#     t.set_text(l)
ax.get_legend().remove()
ax.spines['left'].set_color('w')
ax.tick_params('x', labelsize=ticksfontsize)

ax.tick_params('y', labelsize=ticksfontsize, color='w')
[t.set_color('w') for t in ax.yaxis.get_ticklabels()]
fig.tight_layout()

fig.savefig(opj(outpath,
                'pattern_expression_o2_money_rank_moneyfirst.svg'), dpi=600, transparent=True)


#################################################################
# Plot the SVM plane
#################################################################

model_painmoney = np.load(opj(basepath, 'derivatives/mvpa/decision/model_painmoney_moneyfirst.npy'),
                          allow_pickle=True).item()


plot_svm_plane(model=model_painmoney, outfigpath=outpath,
               labelfontsize=labelfontsize, titlefontsize=labelfontsize,
               legendfontsize=legendfontsize)


