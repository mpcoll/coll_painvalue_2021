from nilearn.image import load_img, new_img_like, concat_imgs
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask, unmask
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import norm
from nltools.stats import fdr
from tqdm import tqdm
from joblib import Parallel, delayed
from nltools.analysis import Roc
from nilearn.plotting import view_img
from nilearn.decoding import SearchLight


###################################################################
# Paths
###################################################################
name = 'choiceval' # Analysis name
basepath = '/data'
model_dir = opj(basepath, 'derivatives/glms/model_choiceval')
outpath =  opj(basepath, 'derivatives/mvpa/choiceval_level')
if not os.path.exists(outpath):
    os.mkdir(outpath)

###################################################################
# Computing parameters
###################################################################

nbootstraps = 10000
nstop = 200 # frequency to stop/save bootstraps
njobs = 40

# Analysis mask
group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))



###################################################################
# Get data for prediction
###################################################################
meta = pd.read_csv(opj(model_dir, 'metadata.csv'))

# Get image in specified path
subs, imgpath = [], []
for ii in meta['imgpathz']:
    subs.append(ii.split('/')[-1][:6])
    imgpath.append(opj(model_dir, ii.split('/')[-1]))
meta['subject_id'] = subs
meta['imgpath'] = imgpath

# Subjects
sub_ids_dat = np.asarray(meta['subject_id']).astype(object)


# Pain level
Y = np.asarray(meta['choiceval']).astype(np.float)

# Brain Data
X = []
X += [apply_mask(fname, group_mask) for fname in tqdm(meta['imgpath'])]
X = np.squeeze(np.stack(X))

# Save
np.save(opj(outpath, name + '_features.npy'), X)
np.save(opj(outpath, name + '_targets.npy'), Y)
np.save(opj(outpath, name + '_groups.npy'), sub_ids_dat)

###################################################################
# Level prediction
###################################################################

# Standardize, PCA, LASSO
lassopcr = Pipeline(steps=[
                           ('scaler', StandardScaler()),
                           ('pca', PCA()),
                           ('lasso', Lasso())])

# Set up the cross_valdation
outer_cv = GroupKFold(n_splits=10).split(X, Y, groups=sub_ids_dat)

# Cross validated prediction
Y_pred = np.zeros(len(Y)) + 9999
r_folds, rmse_folds, r2_folds = np.zeros(len(Y)), np.zeros(len(Y)), np.zeros(len(Y))
weights_xval = []
folds_test = np.zeros(len(Y)) + 9999

cv_count = 0
r2_perc_all = []
for train, test in outer_cv:
    cv_count += 1
    # Fit the classifier
    lassopcr.fit(X[train], Y[train])

    # Get the cv prediction
    Y_pred[test] = lassopcr.predict(X[test])
    folds_test[test] = cv_count

    # Get the cv weights
    xval_weights = np.dot(lassopcr['pca'].components_.T, lassopcr['lasso'].coef_)
    weights_xval.append(unmask(xval_weights, group_mask))

    # Fold metrics
    r_folds[test] = pearsonr(Y[test], Y_pred[test])[0]
    r2_folds[test] = r2_score(Y[test], Y_pred[test])
    rmse_folds[test] = np.sqrt(mean_squared_error(Y[test], Y_pred[test]))

# Overal metrics
r = pearsonr(Y, Y_pred)[0]
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = r2_score(Y, Y_pred)

# Ouptut data frame
out = pd.DataFrame(data=dict(Y_true=Y, Y_pred=Y_pred, r_xval=[r]*len(Y),
                             rmse_xval=[rmse]*len(Y),
                             r2_xval=[r2]*len(Y), subject_id=sub_ids_dat,
                             folds_test=folds_test,
                             r_folds=r_folds, r2_folds=r2_folds,
                             rmse_folds=rmse_folds
                             ))

out.to_csv(opj(outpath, name + '_cvstats.csv'))


# Average the cross validation weights to get the final model
weights_xval_mean = new_img_like(group_mask,
                                 np.mean(np.stack([w.get_data()
                                                   for w in weights_xval]),
                                         axis=0))

weights_xval_mean.to_filename(opj(outpath, name + '_weightsxvalmean.nii'))

# Save weights in xval
concat_imgs(weights_xval).to_filename(opj(outpath,
                                          name + '_weights_xval.nii.gz'))


###################################################################
# Boostrap weights and threshold map
###################################################################
if not os.path.exists(opj(outpath, 'bootsamples')):
    os.mkdir(opj(outpath, 'bootsamples'))

# Bootstrap function
def bootstrap_weights(X, Y):
    # Randomly select observations
    boot_ids = np.random.choice(np.arange(len(X)),
                                size=len(X),
                                replace=True)

    # Fit the classifier on this sample and get the weights
    lassopcr.fit(X[boot_ids], Y[boot_ids])

    # Return the weights and stats
    return np.dot(lassopcr['pca'].components_.T, lassopcr['lasso'].coef_)

# Run in parrallel and stop/save regurarly to run in multiple
for i in tqdm(range(nbootstraps//nstop)):
    # Check if file alrady exist in case bootstrap done x times
    outbootfile = ['bootsamples_' + str(nstop) + 'samples_'
                    + str(i+1) + '_of_'
                    + str(nbootstraps//nstop) + '.npy']
    print("Running permloop " + str(i + 1) + ' out of ' + str(nbootstraps//nstop))
    if not os.path.exists(opj(outpath, 'bootsamples', outbootfile[0])):
        bootstrapped = Parallel(n_jobs=njobs,
                                verbose=0)(delayed(bootstrap_weights)(X=X, Y=Y)
                                           for i in range(nstop))
        bootstrapped = np.stack(bootstrapped)
        np.save(opj(outpath, 'bootsamples', outbootfile[0]), bootstrapped)

# Load all boostraps
bootstrapped = np.vstack(np.stack([np.load(opj(outpath, 'bootsamples', f))
                         for f in os.listdir(opj(outpath, 'bootsamples'))
                         if 'bootsamples' in f], axis=0))

assert bootstrapped.shape[0] == nbootstraps

# Get bootstraped statistics and threshold (as in nltools)
# Zscore
boot_z = bootstrapped.mean(axis=0)/bootstrapped.std(axis=0)

# boot_z[bootstrapped.mean(axis=0) == 0] = 0
unmask(boot_z, group_mask).to_filename(opj(outpath, name + '_bootz.nii'))

# P vals
boot_pval =  2 * (1 - norm.cdf(np.abs(boot_z)))
unmask(boot_pval, group_mask).to_filename(opj(outpath,
                                              name + '_boot_pvals.nii'))

# FDR orrected z
boot_z_fdr = np.where(boot_pval < fdr(boot_pval, q=0.05), boot_z, 0)
boot_z_unc001 = np.where(boot_pval < 0.001, boot_z, 0)
boot_z_unc005 = np.where(boot_pval < 0.005, boot_z, 0)
boot_z_unc01 = np.where(boot_pval < 0.01, boot_z, 0)


unmask(boot_z_fdr, group_mask).to_filename(opj(outpath,
                                               name + '_bootz_fdr05.nii'))
unmask(boot_z_unc001, group_mask).to_filename(opj(outpath,
                                               name + '_bootz_unc001.nii'))
unmask(boot_z_unc005, group_mask).to_filename(opj(outpath,
                                               name + '_bootz_unc005.nii'))
unmask(boot_z_unc01, group_mask).to_filename(opj(outpath,
                                               name + '_bootz_unc01.nii'))




view_img(unmask(boot_z_fdr, group_mask))

###################################################################
# Load PVP and MVP to compare
###################################################################

cval = apply_mask(load_img(opj(outpath, name + '_weightsxvalmean.nii')),
                  group_mask)
pvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa', 'pain_offer_level',
                    'painlevel_weightsxvalmean.nii')), group_mask)
mvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa', 'money_offer_level',
                    'moneylevel_weightsxvalmean.nii')), group_mask)


pearsonr(mvp, pvp)


view_img(unmask(boot_z_fdr, group_mask))
view_img(unmask(boot_z_fdr, group_mask))
view_img(unmask(boot_z_fdr, group_mask))
