
import os
from os.path import join as opj
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
from scipy.ndimage.filters import gaussian_filter


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
sub = list(filter(r.match, os.listdir(sourcedir)))


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
df['rt'] = data['offer2_duration']*1000
df['accept'] = data['accept']
df['pain_level'] = data['pain_level']
df['pain_rank'] = data['pain_rank']
df['pain_intensity'] = data['intensity_level']
df['money_level'] = data['money_level']
df['money_rank'] = round((data['money_level']+1.11)/1.11)
df['run'] = data['run']

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

# Choice difficulty
fig = plt.figure(figsize=(2.5, 2))
df.loc[:, 'choicediff'] = 10 - np.abs(df['money_rank']-df['pain_rank'])

# keep only relevant columns
heat_cdiff = df[['choicediff', 'money_rank', 'pain_rank']]
# Make a table
heat_cdiff = heat_cdiff.pivot_table('choicediff', 'money_rank', 'pain_rank',
                              aggfunc=np.mean)
heat_cdiff_smooth = gaussian_filter(heat_cdiff, sigma=1)

# count = count + 1
# fig.add_subplot(4, 2, count)
ax = sns.heatmap(heat_cdiff_smooth, cmap=cmap1)
ax.figure.axes[-1].set_ylabel('Choice difficulty',
                              size=colorbarfontsize)

# moneyticks = [str(int(np.floor(float(m)))) for m in moneyticks]
ax.set_title("Choice difficulty", {'fontsize': titlefontsize})
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

fig.tight_layout()
fig.savefig(opj(outfigdir, 'behavioural_matrices_choicediff.svg'), dpi=600,
            transparent=True)

def print_stats(datain, name):
    print(name)
    print('N: ' + str(len(datain)))
    print('mean: ' + str(np.mean(datain)))
    print('min: ' + str(np.min(datain)))
    print('max: ' + str(np.max(datain)))
    print('std: ' + str(np.std(datain)))

    out = [name
           + '\n N: ' + str(len(datain))
           + '\n mean: ' + str(np.mean(datain))
           + '\n min: ' + str(np.min(datain))
           + '\n max: ' + str(np.max(datain))
           + '\n std: ' + str(np.std(datain))]

    return out[0]


############################################
# Print text file for results in manuscript
############################################

file1 = open(opj(outfigdir, "behavioural_results.txt"), "w")
file1.write("Demographics \n")
file1.write(print_stats(demo.age, 'Age'))
file1.write('\nN males ' + str(np.sum(demo.is_male)))
file1.write("\nBehaviour\n")
file1.write(print_stats(acc, 'Acceptance'))
# Average response time by part
rtavg = []
for p in list(set(df['sub'])):
    dffpart = df[df['sub'] == p]
    rtavg.append(np.mean(dffpart.rt))
file1.write(print_stats(rtavg, '\nResponse time'))
file1.close()


# Save for R models
df.to_csv(opj(outdir, 'behavioural_data.csv'))
