from nilearn.image import load_img, new_img_like, resample_to_img
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn.masking import apply_mask
from scipy.spatial.distance import cosine
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom_test
from nltools import Roc
############################################################
# Paths
############################################################

name = 'pinesdat' # Analysis name
basepath = '/data'
sourcedir = opj(basepath, 'external', 'pines_data')
outpath = opj(basepath, 'derivatives/figures')

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


############################################################
# Make mask
############################################################
pines_files = os.listdir(sourcedir)

# Create a mask in which both datasets have voxels
pines_img = load_img(opj(sourcedir, pines_files[0]))
pines_msk = new_img_like(pines_img, np.where(pines_img.get_data() == 0, 0, 1))

group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))
group_mask = resample_to_img(group_mask, pines_msk, interpolation='nearest')

both_msk = new_img_like(pines_img, np.where(pines_msk.get_data() + group_mask.get_data() == 2, 1, 0))
both_msk.to_filename(opj(sourcedir, 'pines_mask.nii.gz'))

############################################################
# Load data
############################################################
pines_files = [f for f in os.listdir(sourcedir) if '.nii' in f and 'mask' not in f]
pines_sub = [f.split('_')[2] for f in pines_files]
pines_rating = [int(f.split('_')[4]) for f in pines_files]
pines_files = [opj(sourcedir, f) for f in pines_files]

meta = pd.DataFrame(data={'files': pines_files,
                            'subject_id': pines_sub,
                            'rating': pines_rating})

# Ratings
Y = np.asarray(pines_rating).astype(float)

# Participants
sub_ids_dat = np.asarray(pines_sub)

# Brain Data
X = []
X += [apply_mask(fname, both_msk) for fname in tqdm(pines_files)]
X = np.squeeze(np.stack(X))

np.save(opj(sourcedir, name + '_features.npy'), X)
np.save(opj(sourcedir, name + '_targets.npy'), Y)
np.save(opj(sourcedir, name + '_groups.npy'), sub_ids_dat)

############################################################
# Apply patterns
############################################################

def pattern_expression_nocv(dat, pattern, stats, name):
    """Calculate similarity between maps using dot product and cosine product.
       Non-crossvalidated - to use with external data/patterns.

    Args:
        dat ([array]): images to calculate similarity on (array of shape n images x n voxels)
        pattern ([array]): Pattern weights
        stats ([pd df]): Data frame with subejct id and fods for each in columns
        name ([string]): Name to add to ouput columns
    Returns:
        [df]: stats df with dot and cosine columns added
    """
    pexpress = np.zeros(dat.shape[0]) + 9999
    cosim = np.zeros(dat.shape[0]) + 9999

    for xx in range(dat.shape[0]):
            pexpress[xx] = np.dot(dat[xx, :], pattern)
            cosim[xx] = 1- cosine(dat[xx, :], pattern)
    stats[name + '_dot'] = pexpress
    stats[name + '_cosine'] = cosim

    return stats

pvp = apply_mask(resample_to_img(load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                   'painlevel_weightsxvalmean.nii')), both_msk), both_msk)

mvp = apply_mask(resample_to_img(load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                   'moneylevel_weightsxvalmean.nii')), both_msk), both_msk)

meta = pattern_expression_nocv(X, pvp, meta, 'pvp')
meta = pattern_expression_nocv(X, mvp, meta, 'mvp')

###################################################################
# Single interval classification PVP
###################################################################
comparisons = [[1, 3],
               [1, 5],
               [3, 5]]

roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(meta.pvp_cosine[meta.rating.isin(c)])
    outcome = list(meta.rating[meta.rating.isin(c)])
    outcome = np.where(outcome == np.min(outcome), 0, 1).astype(bool)

    roc = Roc(input_values=inputs,
              binary_outcome=outcome)

    roc.calculate()
    roc.summary()
    roc_results['accuracy'].append(np.mean([roc.sensitivity, roc.specificity]))
    roc_results['accuracy_se'].append(roc.accuracy_se)
    # Binomial test using balanced accuracy
    roc_results['accuracy_p'].append(binom_test(x=[np.round(len(outcome)*np.mean(roc_results['accuracy'][-1])),
              len(outcome)-np.round(len(outcome)*np.mean(roc_results['accuracy'][-1]))] ))

# Plot
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
ax.axhline(0.5, linestyle='--', color='k')
ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='y', labelsize=ticksfontsize, color='k')
ax.set_ylabel('Balanced accuracy', fontsize=labelfontsize, color='k')
ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
ax.set_xlabel('')
ax.legend(fontsize=legendfontsize, loc='lower left', title='Emotion rating',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_pines_usingpvp.svg'),
            dpi=600, transparent=True)



###################################################################
# Single interval classification MVP
###################################################################
roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(meta.mvp_cosine[meta.rating.isin(c)])
    outcome = list(meta.rating[meta.rating.isin(c)])
    outcome = np.where(outcome == np.min(outcome), 0, 1).astype(bool)
    roc = Roc(input_values=inputs,
              binary_outcome=outcome)
    roc.calculate()
    roc.summary()
    roc_results['accuracy'].append(np.mean([roc.sensitivity, roc.specificity]))
    roc_results['accuracy_se'].append(roc.accuracy_se)
    roc_results['accuracy_p'].append(binom_test(x=[np.round(len(outcome)*np.mean(roc_results['accuracy'][-1])),
              len(outcome)-np.round(len(outcome)*np.mean(roc_results['accuracy'][-1]))] ))

# Plot
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
ax.axhline(0.5, linestyle='--', color='k')

ax.set_xticklabels('')
ax.tick_params(axis='both', labelsize=ticksfontsize)
ax.tick_params(axis='y', labelsize=ticksfontsize, color='k')
ax.set_ylabel('Balanced accuracy', fontsize=labelfontsize, color='k')
ax.legend(fontsize=legendfontsize, ncol=1, loc='lower left')
ax.set_xlabel('')
ax.legend(fontsize=legendfontsize, loc='lower left', title='Emotion rating',
          title_fontsize=legendfontsize)
fig.tight_layout()
fig.savefig(opj(outpath, 'forced_choice_acc_pines_usingmvp.svg'),
            dpi=600, transparent=True)


# Line plot
meta['rating'] = meta['rating'].astype(int)
meta['pvp_cosine_z'] = zscore(meta['pvp_cosine'])
meta['mvp_cosine_z'] = zscore(meta['mvp_cosine'])

corr_plot = meta.melt(id_vars=['rating', 'type'], value_vars=['pvp_cosine_z', 'mvp_cosine_z'])

fig, ax = plt.subplots(figsize=(1.6, 1.6))

ax = sns.pointplot(corr_plot['rating'], corr_plot['value'], hue=corr_plot['variable'],
                   scale=0.4, ci=68, errwidth=1, palette=[colp, colm], legend=False)

# Add labels
ax.set_ylim(-0.6, 1)
ax.set_title("", {'fontsize': titlefontsize})
ax.set_xlabel("Emotion rating", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})

ax.legend([],[], frameon=False)
ax.tick_params('both', labelsize=ticksfontsize)
fig.tight_layout()
fig.savefig(opj(outpath,
                'pattern_expression_o1_lineplot_emotion.svg'), transparent=True)
