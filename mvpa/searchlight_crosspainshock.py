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

###################################################################
# Paths
###################################################################

name = 'sl_crossps' # Analysis name
basepath = '/data'
outpath =  opj(basepath, 'derivatives/mvpa/searchlights/searchlight_crosspainshock')
if not os.path.exists(outpath):
    os.makedirs(outpath)

###################################################################
# Computing parameters
###################################################################

nbootstraps = 5000
nstop = 20 # frequency to stop/save bootstraps
njobs = 20

# Group mask
group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))

def pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]
pears_score = make_scorer(pearson)


# Load the spheres mask
def load_sparse_lil(filename):
    from scipy.sparse import lil_matrix
    loader = np.load(filename, allow_pickle=True)
    result = lil_matrix(tuple(loader["shape"]), dtype=str(loader["dtype"]))
    result.data = loader["data"]
    result.rows = loader["rows"]
    return result

# Load spheres matrix
sl_spheres = load_sparse_lil(opj(basepath, 'derivatives', 'sl_6mmshperes.npz'))


# ###################################################################
# Get data
###################################################################
# Modalities name
mod1, mod2 = 'pain', 'money'

# Load pvp data
mod1_dat = np.load(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                         'painlevel_features.npy'))

mod1_subs = np.load(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                         'painlevel_groups.npy'), allow_pickle=True)

mod1_y = np.load(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                         'painlevel_targets.npy'), allow_pickle=True)

# Load shock data
mod2_dat = np.load(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                         'shocklevel_features.npy'))

mod2_subs = np.load(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                         'shocklevel_groups.npy'), allow_pickle=True)

mod2_y = np.load(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                         'shocklevel_targets.npy'), allow_pickle=True)

# Subs should be the same
# assert np.array_equal(mod1_subs, mod2_subs)
info_df = pd.DataFrame(dict(subject_id = list(mod1_subs) + list(mod2_subs),
                            modality=[mod1]*len(mod1_subs) + [mod2]*len(mod2_subs)))

X = np.concatenate([mod1_dat, mod2_dat])
Y  = np.concatenate([mod1_y, mod2_y])

assert X.shape[0] == Y.shape[0] == len(info_df)

###################################################################
# Create cross-prediction cross validation
###################################################################

# Get subs in each train/test split (10 x 2 = 20 folds)
cv = GroupKFold(10).split(mod1_dat, groups=mod1_subs)

cross_predict_cv = []
pain_cv = []
money_cv = []
folds = 0
for train, test in cv:
    train_subs = mod1_subs[train]
    test_subs = mod1_subs[test]

    # Train on modality 1, test on 2
    folds += 1
    info_df['fold_'  + str(folds)] = 'nan'
    train1 = np.where((info_df['modality'] == mod1)
                      & (info_df['subject_id'].isin(train_subs)))[0]
    test1 = np.where((info_df['modality'] == mod2)
                     & (info_df['subject_id'].isin(test_subs)))[0]
    cross_predict_cv.append((train1, test1))

    # Save info to double check
    info_df['fold_'  + str(folds)].iloc[train1] = 'train'
    info_df['fold_'  + str(folds)].iloc[test1] = 'test'

    # Vice versa
    folds += 1
    info_df['fold_'  + str(folds)] = 'nan'
    train2 = np.where((info_df['modality'] == mod2)
                      & (info_df['subject_id'].isin(train_subs)))[0]
    test2 = np.where((info_df['modality'] == mod1)
                     & (info_df['subject_id'].isin(test_subs)))[0]
    cross_predict_cv.append((train2, test2))

    info_df['fold_'  + str(folds)].iloc[train2] = 'train'
    info_df['fold_'  + str(folds)].iloc[test2] = 'test'

# Save cv info
info_df.to_csv(opj(outpath, name + '_cvinfo.csv'))


# ###################################################################
# # Run the cv searchlight
# ###################################################################
sl_pcr = Pipeline(steps=[('scaler', StandardScaler()),
                         ('pca', PCA(0.8)),
                         ('lasso', LinearRegression())])

def pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]
pears_score = make_scorer(pearson)

scores = search_light(X, Y, sl_pcr, sl_spheres, scoring=pears_score,
                      cv=cross_predict_cv, n_jobs=njobs, verbose=10)

sl_scores = unmask(scores, group_mask)
view_img(unmask(scores, group_mask))
sl_scores.to_filename(opj(outpath, name + '_slscores.nii'))

# ###################################################################
# # Bootstrap the searchlight
# ###################################################################

bootname = 'bootslgm_pcr'
if not os.path.exists(opj(outpath, bootname)):
    os.mkdir(opj(outpath, bootname))

def bootstrap_cross_sl(spheres_matrix, X, Y, info_df, clf, scoring, njobs):

    mod1_idx = np.where(info_df['modality'] == mod1)[0]
    mod2_idx = np.where(info_df['modality'] == mod2)[0]
    bootids1 = np.random.choice(mod1_idx,
                                size=len(mod1_idx),
                                replace=True)
    bootids2 = np.random.choice(mod2_idx,
                                size=len(mod2_idx),
                                replace=True)
    # Cross predict mock cv
    cross_predict_mock_cv = []
    # Train pain pred money
    cross_predict_mock_cv.append((np.arange(len(bootids1)),
                                len(bootids1) + np.arange(len(bootids2))))
    # Train money pred pain
    cross_predict_mock_cv.append((len(bootids1) + np.arange(len(bootids2)),
                                  np.arange(len(bootids1))))
    bootids = np.concatenate([bootids1, bootids2])

    # Fit the searchlight
    scores = search_light(X[bootids],
                          Y[bootids], clf, spheres_matrix,
                          scoring=scoring,
                          cv=cross_predict_mock_cv,
                          n_jobs=njobs, verbose=0)

    # Return the scores
    return scores


# Run in parrallel and stop/save regurarly to run in multiple
for i in tqdm(range(nbootstraps//nstop)):
    # Check if file alrady exist in case bootstrap done x times
    outbootfile = ['bootsamples_' + str(nstop) + 'samples_'
                    + str(i+1) + '_of_'
                    + str(nbootstraps//nstop) + '.npy']
    print("Running bootloop " + str(i + 1) + ' out of '
          + str(nbootstraps//nstop))
    if not os.path.exists(opj(outpath, bootname, outbootfile[0])):
        bootstrapped = Parallel(n_jobs=njobs,
                                verbose=1)(delayed(bootstrap_cross_sl)(X=X,
                                                                       Y=Y,
                                                                       spheres_matrix=sl_spheres,
                                                                       scoring=pears_score,
                                                                       clf=sl_pcr,
                                                                       njobs=1,
                                                                       info_df=info_df)
                                           for i in range(nstop))
        bootstrapped = np.stack(bootstrapped)
        np.save(opj(outpath, bootname, outbootfile[0]), bootstrapped)


# Load all boostraps
bootstrapped = np.vstack([np.load(opj(outpath, bootname, f))
                         for f in os.listdir(opj(outpath, bootname))
                         if 'bootsamples_'  in f])


# Zscore
bootstrapped = bootstrapped[~np.isnan(bootstrapped).any(axis=1), :]
boot_z = np.mean(bootstrapped, axis=0)/np.std(bootstrapped, axis=0)
assert np.sum(np.isnan(boot_z)) == 0

# Two-tailed p-vals
boot_pval =  2*(1 - norm.cdf(np.abs(boot_z)))

sl_scores = apply_mask(load_img(opj(outpath, name + '_slscores.nii')), group_mask)

# Correct and save
boot_z_bonf = np.where(boot_pval < (0.05/len(boot_pval)), boot_z, 0)
scores_bonf = np.where(boot_pval < 0.05/len(sl_scores), sl_scores, 0)

unmask(boot_z_bonf, group_mask).to_filename(opj(outpath, name + '_bootz_fwe05.nii'))
unmask(boot_z, group_mask).to_filename(opj(outpath, name + '_bootz.nii'))
unmask(scores_bonf, group_mask).to_filename(opj(outpath, name + '_scores_fwe05'))
