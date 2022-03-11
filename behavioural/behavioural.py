
import os
from os.path import join as opj
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
from scipy.ndimage.filters import gaussian_filter
import statsmodels.api as sm


###################################################################
# OS specific parameters
###################################################################

# basepath = '/lustre04/scratch/mpcoll/2020_npng_newprep'
basepath = '/data'

###################################################################
# Fixed parameters
###################################################################

# Paths
sourcedir = opj(basepath, 'source')
outdir = opj(basepath, 'derivatives/behavioural')
outfigdir = opj(basepath, 'derivatives/figures/behavioural')
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists(outfigdir):
    os.mkdir(outfigdir)

# Get participants
r = re.compile('.*sub-*')
subs = list(filter(r.match, os.listdir(sourcedir)))


###################################################################
# Plot parameters
###################################################################
# colors
current_palette = sns.color_palette('colorblind', 6)
colp = current_palette[0]
colm = current_palette[2]

cold = current_palette[1]
colc = current_palette[3]

# Label size
labelfontsize = 9
titlefontsize = np.round(labelfontsize*1)
ticksfontsize = np.round(labelfontsize*0.8)
legendfontsize = np.round(labelfontsize*0.8)
colorbarfontsize = legendfontsize

# Font
plt.rcParams['font.family'] = 'Helvetica'

# Despine
plt.rc("axes.spines", top=False, right=False)

####################################################################
# Loop sub and run and concatenante all csv files
data = pd.DataFrame()

for s in subs:
    for run in range(5):
        subdat = pd.read_csv(opj(sourcedir, s,
                                 'func', s + '_task-npng_run-'
                                 + str(run+1) + '_events.tsv'),
                             sep='\t')
        subdat['sub'] = s

        data = pd.concat([data, subdat])

# Put in a smaller dataframe to get rid of unused columns
df = pd.DataFrame()
df['sub'] = data['sub']
df['subject_id'] = data['sub'].copy()
df['rt'] = data['offer2_duration']*1000
df['accept'] = data['accept']
df['pain_level'] = data['pain_level']
df['pain_rank'] = data['pain_rank']
df['pain_intensity'] = data['intensity_level']
df['money_level'] = data['money_level']
df['money_rank'] = round((data['money_level']+1.11)/1.11)
df['run'] = data['run']
df['painfirst'] = data['painfirst']
df_pfirst = df[df['painfirst'] == 1]
df_mfirst = df[df['painfirst'] == 0]

# Remove unanswered trials
df = df[df['rt'] < 5000]
df = df[df['rt'] > 200]


# Get average acc and Rt
acc = []
refused = []
for p in list(set(df['sub'])):
    dfp = df[df['sub'] == p]
    acc.append(np.mean(dfp['accept']))
    refused.append(np.sum(dfp['money_level'][dfp['accept'] == 0]))


print('% accepted: ' + str(np.mean(acc)))
print('Potential $ refused: ' + str(np.mean(refused)))
print('Mean RT: ' + str(np.mean(df['rt'])))

# ## Demographics
demo = pd.read_csv(opj(sourcedir, 'participants.csv'))

keep = [True if s in list(subs) else False for s in demo.subject_id]
demo = demo[keep]

# Behavioural plots


# Choice difficulty
# Init figure
fig = plt.figure(figsize=(2.5, 2))

cmap1 = 'viridis'
cmap2 = 'cividis'
# Acceptance across offers

heat_acc = df[['accept', 'money_rank', 'pain_rank']]
heat_acc['accept'] = heat_acc['accept']*100
# Make a table

heat_acc = heat_acc.pivot_table('accept', 'money_rank', 'pain_rank',
                                aggfunc=np.mean)

heat_acc_smooth = gaussian_filter(heat_acc, sigma=1)
ax = sns.heatmap(heat_acc_smooth, cmap=cmap2, annot=False)
ax.figure.axes[-1].set_ylabel('% Accept',
                              size=colorbarfontsize)
ax.set_title("Acceptance", {'fontsize': titlefontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=labelfontsize)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels([int(r) for r in range(1, 11)], {'fontsize': ticksfontsize},
                   va='center')
ax.invert_yaxis()
ax.invert_xaxis()

# ax.text(-1.4, -0.4, 'C', fontsize=letterfontsize, fontweight='bold')
ax.set_xticklabels(list(range(1, 11)), {'fontsize': ticksfontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=colorbarfontsize)
# ax.text(-0.2, 1.02, 'A', transform=ax.transAxes,
#         size=letterfontsize, weight='normal')
fig.tight_layout()
fig.savefig(opj(outfigdir, 'behavioural_matrices_accept.svg'), dpi=600,
            transparent=True, bbox_inches='tight')


fig = plt.figure(figsize=(2.5, 2))

# Acceptance across offers

heat_acc = df[['accept', 'money_rank', 'pain_rank']]
heat_acc['accept'] = heat_acc['accept']*100
# Make a table

heat_acc = heat_acc.pivot_table('accept', 'money_rank', 'pain_rank',
                                aggfunc=np.mean)

heat_acc_smooth = gaussian_filter(heat_acc, sigma=1)
ax = sns.heatmap(heat_acc_smooth, cmap=cmap2, annot=False)
ax.figure.axes[-1].set_ylabel('% Accept',
                              size=colorbarfontsize)
ax.set_title("Acceptance", {'fontsize': titlefontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=labelfontsize)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels([int(r) for r in range(1, 11)], {'fontsize': ticksfontsize},
                   va='center')
ax.invert_yaxis()
ax.invert_xaxis()

# ax.text(-1.4, -0.4, 'C', fontsize=letterfontsize, fontweight='bold')
ax.set_xticklabels(list(range(1, 11)), {'fontsize': ticksfontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=colorbarfontsize)
# ax.text(-0.2, 1.02, 'A', transform=ax.transAxes,
#         size=letterfontsize, weight='normal')
fig.tight_layout()
fig.savefig(opj(outfigdir, 'behavioural_matrices_accept.svg'), dpi=600,
            transparent=True, bbox_inches='tight')


fig = plt.figure(figsize=(2.5, 2))

# Acceptance across offers

heat_acc = df_pfirst[['accept', 'money_rank', 'pain_rank']]
heat_acc['accept'] = heat_acc['accept']*100
# Make a table

heat_acc = heat_acc.pivot_table('accept', 'money_rank', 'pain_rank',
                                aggfunc=np.mean)

heat_acc_smooth = gaussian_filter(heat_acc, sigma=1)
ax = sns.heatmap(heat_acc_smooth, cmap=cmap2, annot=False)
ax.figure.axes[-1].set_ylabel('% Accept',
                              size=colorbarfontsize)
ax.set_title("Acceptance - Pain first", {'fontsize': titlefontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=labelfontsize)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels([int(r) for r in range(1, 11)], {'fontsize': ticksfontsize},
                   va='center')
ax.invert_yaxis()
ax.invert_xaxis()

# ax.text(-1.4, -0.4, 'C', fontsize=letterfontsize, fontweight='bold')
ax.set_xticklabels(list(range(1, 11)), {'fontsize': ticksfontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=colorbarfontsize)
# ax.text(-0.2, 1.02, 'A', transform=ax.transAxes,
#         size=letterfontsize, weight='normal')
fig.tight_layout()
fig.savefig(opj(outfigdir, 'behavioural_matrices_accept_pfirst.svg'), dpi=600,
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(2.5, 2))

# Acceptance across offers

heat_acc = df_mfirst[['accept', 'money_rank', 'pain_rank']]
heat_acc['accept'] = heat_acc['accept']*100
# Make a table

heat_acc = heat_acc.pivot_table('accept', 'money_rank', 'pain_rank',
                                aggfunc=np.mean)

heat_acc_smooth = gaussian_filter(heat_acc, sigma=1)
ax = sns.heatmap(heat_acc_smooth, cmap=cmap2, annot=False)
ax.figure.axes[-1].set_ylabel('% Accept',
                              size=colorbarfontsize)
ax.set_title("Acceptance - Money first", {'fontsize': titlefontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=labelfontsize)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels([int(r) for r in range(1, 11)], {'fontsize': ticksfontsize},
                   va='center')
ax.invert_yaxis()
ax.invert_xaxis()

# ax.text(-1.4, -0.4, 'C', fontsize=letterfontsize, fontweight='bold')
ax.set_xticklabels(list(range(1, 11)), {'fontsize': ticksfontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=colorbarfontsize)
# ax.text(-0.2, 1.02, 'A', transform=ax.transAxes,
#         size=letterfontsize, weight='normal')
fig.tight_layout()
fig.savefig(opj(outfigdir, 'behavioural_matrices_accept_mfirst.svg'), dpi=600,
            transparent=True, bbox_inches='tight')

# Response time across offers
fig = plt.figure(figsize=(2.5, 2))

# keep only relevant columns
heat_rt = df[['rt', 'money_rank', 'pain_rank']]
# Make a table
heat_rt = heat_rt.pivot_table('rt', 'money_rank', 'pain_rank',
                              aggfunc=np.mean)
heat_rt_smooth = gaussian_filter(heat_rt, sigma=1)

# count = count + 1
# fig.add_subplot(4, 2, count)
ax = sns.heatmap(heat_rt_smooth, cmap=cmap1)
ax.figure.axes[-1].set_ylabel('Response time (ms)',
                              size=colorbarfontsize)

# moneyticks = [str(int(np.floor(float(m)))) for m in moneyticks]
ax.set_title("Response time", {'fontsize': titlefontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=labelfontsize)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels([int(r) for r in range(1, 11)], {'fontsize': ticksfontsize},
                   va='center')
ax.set_xticklabels(list(range(1, 11)), {'fontsize': ticksfontsize})
cbar = ax.collections[0].colorbar
ax.invert_yaxis()
ax.invert_xaxis()
cbar.ax.tick_params(labelsize=colorbarfontsize)

# ax.text(-1.4, -0.4, 'D', fontsize=letterfontsize, fontweight='bold')

# ax.text(-0.2, 1.02, 'B', transform=ax.transAxes,
#         size=letterfontsize, weight='normal')
fig.tight_layout()
fig.savefig(opj(outfigdir, 'behavioural_matrices_rt.svg'), dpi=600,
            transparent=True)


# Response time across offers
fig = plt.figure(figsize=(2.5, 2))

# keep only relevant columns
heat_rt = df_pfirst[['rt', 'money_rank', 'pain_rank']]
# Make a table
heat_rt = heat_rt.pivot_table('rt', 'money_rank', 'pain_rank',
                              aggfunc=np.mean)
heat_rt_smooth = gaussian_filter(heat_rt, sigma=1)

# count = count + 1
# fig.add_subplot(4, 2, count)
ax = sns.heatmap(heat_rt_smooth, cmap=cmap1)
ax.figure.axes[-1].set_ylabel('Response time (ms)',
                              size=colorbarfontsize)

# moneyticks = [str(int(np.floor(float(m)))) for m in moneyticks]
ax.set_title("Response time - Pain first", {'fontsize': titlefontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=labelfontsize)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels([int(r) for r in range(1, 11)], {'fontsize': ticksfontsize},
                   va='center')
ax.set_xticklabels(list(range(1, 11)), {'fontsize': ticksfontsize})
cbar = ax.collections[0].colorbar
ax.invert_yaxis()
ax.invert_xaxis()
cbar.ax.tick_params(labelsize=colorbarfontsize)

# ax.text(-1.4, -0.4, 'D', fontsize=letterfontsize, fontweight='bold')

# ax.text(-0.2, 1.02, 'B', transform=ax.transAxes,
#         size=letterfontsize, weight='normal')
fig.tight_layout()
fig.savefig(opj(outfigdir, 'behavioural_matrices_rt_pfirst.svg'), dpi=600,
            transparent=True)


# Response time across offers
fig = plt.figure(figsize=(2.5, 2))

# keep only relevant columns
heat_rt = df_mfirst[['rt', 'money_rank', 'pain_rank']]
# Make a table
heat_rt = heat_rt.pivot_table('rt', 'money_rank', 'pain_rank',
                              aggfunc=np.mean)
heat_rt_smooth = gaussian_filter(heat_rt, sigma=1)

# count = count + 1
# fig.add_subplot(4, 2, count)
ax = sns.heatmap(heat_rt_smooth, cmap=cmap1)
ax.figure.axes[-1].set_ylabel('Response time (ms)',
                              size=colorbarfontsize)

# moneyticks = [str(int(np.floor(float(m)))) for m in moneyticks]
ax.set_title("Response time - Money first", {'fontsize': titlefontsize})
ax.set_xlabel("Pain rank", {'fontsize': labelfontsize})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=labelfontsize)
ax.set_ylabel("Money rank", {'fontsize': labelfontsize})
ax.set_yticklabels([int(r) for r in range(1, 11)], {'fontsize': ticksfontsize},
                   va='center')
ax.set_xticklabels(list(range(1, 11)), {'fontsize': ticksfontsize})
cbar = ax.collections[0].colorbar
ax.invert_yaxis()
ax.invert_xaxis()
cbar.ax.tick_params(labelsize=colorbarfontsize)

# ax.text(-1.4, -0.4, 'D', fontsize=letterfontsize, fontweight='bold')

# ax.text(-0.2, 1.02, 'B', transform=ax.transAxes,
#         size=letterfontsize, weight='normal')
fig.tight_layout()
fig.savefig(opj(outfigdir, 'behavioural_matrices_rt_mfirst.svg'), dpi=600,
            transparent=True)

df = df.reset_index()
df.to_csv(opj(outdir, 'behavioural_data.csv'))



for sub in df['sub'].unique():
    subdat = df[df['sub'] == sub]

    # Get weights on decision using logistic regression
    log_reg = sm.Logit(np.asarray(subdat['accept']),
                       np.asarray(subdat[['pain_rank', 'money_rank']])).fit()

    # Add to dataframe
    df.loc[df["sub"] == sub, "coef_pain"] = log_reg.params[0]
    df.loc[df["sub"] == sub, "coef_money"] = log_reg.params[1]

   # Get weights on rt using linear regression
    lin_reg = sm.OLS(np.asarray(subdat['rt']),
                     np.asarray(subdat[['pain_rank', 'money_rank']])).fit()


    # Add to dataframe
    df.loc[df["sub"] == sub, "coef_pain_rt"] = lin_reg.params[0]
    df.loc[df["sub"] == sub, "coef_money_rt"] = lin_reg.params[1]


    subdat_acc = subdat[subdat['accept'] == 1]
   # Get weights on rt using linear regression
    lin_reg = sm.OLS(np.asarray(subdat_acc['rt']),
                     np.asarray(subdat_acc[['pain_rank', 'money_rank']])).fit()


    # Add to dataframe
    df.loc[df["sub"] == sub, "coef_pain_rt_acc"] = lin_reg.params[0]
    df.loc[df["sub"] == sub, "coef_money_rt_acc"] = lin_reg.params[1]


    subdat_rej = subdat[subdat['accept'] == 0]

   # Get weights on rt using linear regression
    lin_reg = sm.OLS(np.asarray(subdat_rej['rt']),
                     np.asarray(subdat_rej[['pain_rank', 'money_rank']])).fit()


    # Add to dataframe
    df.loc[df["sub"] == sub, "coef_pain_rt_rej"] = lin_reg.params[0]
    df.loc[df["sub"] == sub, "coef_money_rt_rej"] = lin_reg.params[1]


# Save for R models
df.to_csv(opj(outdir, 'behavioural_data.csv'))
df.groupby('subject_id').mean().reset_index().to_csv(opj(outdir,
                                                  'behavioural_data_wide.csv'))