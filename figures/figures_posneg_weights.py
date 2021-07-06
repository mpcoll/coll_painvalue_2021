from nilearn.image import load_img, new_img_like, resample_to_img
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask, unmask
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import norm
from nltools.stats import fdr, holm_bonf
from tqdm import tqdm
from joblib import Parallel, delayed
from nilearn.plotting import view_img
from nilearn.decoding.searchlight import search_light
from nilearn.image import index_img
from sklearn.metrics import make_scorer
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import zscore
import matplotlib.pyplot as plt

basepath = '/data'

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


def pattern_expression_cv(dat, cv_imgs, pattern_stats, stats, name):
    """Calculate similarity between maps using dot product and cosine product.
       Crossvalidated - to use with data from same dataset.

    Args:
        dat ([array]): images to calculate similarity on (array of shape n images x n voxels)
        cv_imgs ([array]): Pattern weights for each fold (n folds x n voxels)
        pattern_stats ([pd df]): Data frame with subejct id and folds for corresponding pattern
        stats ([pd df]): Data frame with subejct id and folds for corresponding patte
        name ([string]): Name to add to ouput columns
    Returns:
        [df]: stats df with dot and cosine columns added
    """
    pexpress = np.zeros(dat.shape[0]) + 9999
    cosim = np.zeros(dat.shape[0]) + 9999
    sub_ids_pat = np.asarray(pattern_stats['subject_id'])
    sub_ids_dat =  np.asarray(stats['subject_id'])

    for fold in np.unique(pattern_stats['folds_test']):
        cv_map = cv_imgs[int(fold-1), :]
        sub_folds = np.unique(sub_ids_pat[np.where(np.asarray(pattern_stats['folds_test']) == fold)])
        for xx in range(dat.shape[0]):
            if sub_ids_dat[xx] in sub_folds:
                pexpress[xx] = np.dot(dat[xx, :], cv_map)
                cosim[xx] = 1- cosine(dat[xx, :], cv_map)

        stats[name + '_dot'] = pexpress
        stats[name + '_cosine'] = cosim
    return stats


def pattern_expression_cv_posneg(dat, cv_imgs, pattern_stats, stats, name):
    """Calculate similarity between maps using dot product and cosine product.
       Crossvalidated - to use with data from same dataset.

    Args:
        dat ([array]): images to calculate similarity on (array of shape n images x n voxels)
        cv_imgs ([array]): Pattern weights for each fold (n folds x n voxels)
        pattern_statsname = 'par_crosspainaffect' # Analysis name
 ([pd df]): Data frame with subejct id and folds for corresponding pattern
        stats ([pd df]): Data frame with subejct id and folds for corresponding patte
        name ([string]): Name to add to ouput columns
    Returns:
        [df]: stats df with dot and cosine columns added
    """
    pexpress = np.zeros(dat.shape[0]) + 9999
    cosim = np.zeros(dat.shape[0]) + 9999
    sub_ids_pat = np.asarray(pattern_stats['subject_id'])
    sub_ids_dat =  np.asarray(stats['subject_id'])


    for fold in np.unique(pattern_stats['folds_test']):
        cv_map = cv_imgs[int(fold-1), :]
        sub_folds = np.unique(sub_ids_pat[np.where(np.asarray(pattern_stats['folds_test']) == fold)])
        for xx in range(dat.shape[0]):
            if sub_ids_dat[xx] in sub_folds:
                pexpress[xx] = np.dot(dat[xx, :], cv_map)
                cosim[xx] = 1- cosine(dat[xx, :], cv_map)

        stats[name + '_dot'] = pexpress
        stats[name + '_cosine'] = cosim

    for fold in np.unique(pattern_stats['folds_test']):
        cv_map = cv_imgs[int(fold-1), :]
        sub_folds = np.unique(sub_ids_pat[np.where(np.asarray(pattern_stats['folds_test']) == fold)])
        for xx in range(dat.shape[0]):
            if sub_ids_dat[xx] in sub_folds:
                pexpress[xx] = np.dot(dat[xx, cv_map >  0], cv_map[cv_map >   0])
                cosim[xx] = 1- cosine(dat[xx, cv_map >   0], cv_map[cv_map >   0])

        stats[name + 'pos_dot'] = pexpress
        stats[name + 'pos_cosine'] = cosim

    for fold in np.unique(pattern_stats['folds_test']):
        cv_map = cv_imgs[int(fold-1), :]
        sub_folds = np.unique(sub_ids_pat[np.where(np.asarray(pattern_stats['folds_test']) == fold)])
        for xx in range(dat.shape[0]):
            if sub_ids_dat[xx] in sub_folds:
                pexpress[xx] = np.dot(dat[xx, cv_map <  0], cv_map[cv_map <   0.])
                cosim[xx] = 1- cosine(dat[xx, cv_map <   0], cv_map[cv_map <   0])
        stats[name + 'neg_dot'] = pexpress
        stats[name + 'neg_cosine'] = cosim
    return stats



# Load data
group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))


# Load cross-validated maps
pvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                   'painlevel_weights_xval.nii.gz')), group_mask)

# Load cross-validated stats to get subs in each fold
pvp_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                            'painlevel_cvstats.csv'))
# Load pvp data
pvp_dat = np.load(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                         'painlevel_features.npy'))


# Load cross-validated maps
mvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                   'moneylevel_weights_xval.nii.gz')), group_mask)

# Load cross-validated stats to get subs in each fold
mvp_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/money_offer_level',
                            'moneylevel_cvstats.csv'))
# Load mvp data
mvp_dat = np.load(opj(basepath, 'derivatives/mvpa/money_offer_level',
                         'moneylevel_features.npy'))

mvp_stats = pattern_expression_cv_posneg(mvp_dat, pvp, pvp_stats, mvp_stats,
                                      'pvp_cv')



mvp_stats['pvp_cvpos_cosine_z'] = zscore(mvp_stats['pvp_cvpos_cosine'])
mvp_stats['pvp_cvneg_cosine_z'] = zscore(mvp_stats['pvp_cvneg_cosine'])
mvp_stats['Y_true'] = mvp_stats['Y_true'].astype(int)

plot_dat = mvp_stats.melt(id_vars=['Y_true', 'subject_id'],
                          value_vars=['pvp_cvpos_cosine_z', 'pvp_cvneg_cosine_z'])

plot_dat['zval'] = zscore(plot_dat['value'])


fig, ax = plt.subplots(figsize=(1.6, 2.2))

sns.pointplot(plot_dat['Y_true'],  plot_dat['value'],
              hue=plot_dat['variable'], scale=0.4, ci=68, errwidth=1,
              legend=False, palette=['#0B9EF3', '#045787'],
              markers=['^', 'o'])

legend = ax.legend(fontsize=legendfontsize, frameon=False,
                   handletextpad=0, loc=(0.1, 0.8))
for t, l in zip(legend.texts, ("+ weights", "- weights")):
    t.set_text(l)

# Add labels
ax.set_ylim((-0.6, 0.8))

ax.set_title("",  {'fontsize': titlefontsize})
ax.set_xlabel("Money offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
ax.tick_params(axis='both', labelsize=ticksfontsize)
fig.tight_layout()

fig.savefig(opj("/data/derivatives/figures",
                'pattern_expression_crosspm_negpos.svg'), transparent=True)


pvp_stats = pattern_expression_cv_posneg(pvp_dat, mvp, mvp_stats, pvp_stats,
                                        'mvp_cv')

pvp_stats['mvp_cvpos_cosine_z'] = zscore(pvp_stats['mvp_cvpos_cosine'])
pvp_stats['mvp_cvneg_cosine_z'] = zscore(pvp_stats['mvp_cvneg_cosine'])
pvp_stats['Y_true'] = pvp_stats['Y_true'].astype(int)

plot_dat = pvp_stats.melt(id_vars=['Y_true', 'subject_id'],
                          value_vars=['mvp_cvpos_cosine_z', 'mvp_cvneg_cosine_z'])


fig, ax = plt.subplots(figsize=(1.6, 2.2))

sns.pointplot(plot_dat['Y_true'],  plot_dat['value'],
              hue=plot_dat['variable'], scale=0.4, ci=68, errwidth=1,
              legend=False, palette=['#41D48A', '#009048'],
              markers=['^', 'o'])

legend = ax.legend(fontsize=legendfontsize, frameon=False,
                   handletextpad=0, loc=(0.1, 0.8))
for t, l in zip(legend.texts, ("+ weights", "- weights")):
    t.set_text(l)

# Add labels
ax.set_ylim((-0.6, 0.8))
ax.set_title("",  {'fontsize': titlefontsize})
ax.set_xlabel("Pain offer level", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
ax.tick_params(axis='both', labelsize=ticksfontsize)
fig.tight_layout()

fig.savefig(opj("/data/derivatives/figures",
                'pattern_expression_crossmp_negpos.svg'), transparent=True)

