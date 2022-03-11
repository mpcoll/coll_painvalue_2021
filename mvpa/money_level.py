from nilearn.image import load_img, new_img_like, concat_imgs
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask, unmask
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LinearRegression
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
from nltools import Brain_Data
from sklearn.metrics import make_scorer

###################################################################
# Paths
################################################d###################

basepath = '/data'
name = 'moneylevel' # Analysis name
model_dir = opj(basepath, 'derivatives/glms/model_offer1')
outpath =  opj(basepath, 'derivatives/mvpa/money_offer_level')
if not os.path.exists(outpath):
    os.makedirs(outpath)

###################################################################
# Computing parameters
###################################################################

nbootstraps = 10000
nstop = 200 # frequency to stop/save bootstraps
njobs = 40

# Analysis mask
group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))
bgimg = opj(basepath, 'external/tpl-MNI152NLin2009cAsym_space-MNI_res-01_T1w_brain.nii')

###################################################################
# Get data for prediction
###################################################################

meta = pd.read_csv(opj(model_dir, 'metadata.csv'))
# Fix this at some point
subs, imgpath = [], []
for ii in meta['imgpathz']:
    subs.append(ii.split('/')[-1][:6])
    imgpath.append(opj(model_dir, ii.split('/')[-1]))
meta['subject_id'] = subs
meta['imgpath'] = imgpath

# Keep only money images
meta = meta[meta['painfirst'] == 0]

# Subjects
sub_ids_dat = np.asarray(meta['subject_id']).astype(object)

# Pain level
Y = np.asarray(meta['level']).astype(np.float)

# Brain Data
X = []
X += [apply_mask(fname, group_mask)
        for fname in tqdm(meta['imgpath'])]
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
                             rmse_folds=rmse_folds))
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
# Permute the prediction
###################################################################

# # Set up the cross_valdation
def permute_cv_prediction(X, Y, sub_ids_dat):
    Y_pred = np.zeros(len(Y)) + 9999

    outer_cv = GroupKFold(n_splits=10).split(X, Y, groups=sub_ids_dat)

    Y_shuffle = Y.copy()
    # Shuffle within subjects
    for sub in np.unique(sub_ids_dat):
        Y_shuffle[sub_ids_dat == sub] = np.random.permutation(Y[sub_ids_dat == sub])

    for train, test in outer_cv:
        # Fit the classifier on shuffled labels
        lassopcr.fit(X[train], Y_shuffle[train])

        # Predict unshuffled
        Y_pred[test] = lassopcr.predict(X[test])

    r = pearsonr(Y, Y_pred)[0]
    r2 = r2_score(Y, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y, Y_pred))

    return r, r2, rmse

if not os.path.exists(opj(outpath, 'permsamples')):
    os.mkdir(opj(outpath, 'permsamples'))

for i in tqdm(range(nbootstraps//nstop)):
    # Check if file alrady exist in case bootstrap done x times
    outbootfile = ['permsamples' + str(nstop) + 'samples_'
                    + str(i+1) + '_of_'
                    + str(nbootstraps//nstop) + '.csv']
    print("Running permloop " + str(i + 1) + ' out of ' + str(nbootstraps//nstop))
    if not os.path.exists(opj(outpath, 'permsamples', outbootfile[0])):
        permuted = Parallel(n_jobs=5, verbose=11)(delayed(permute_cv_prediction)(X=X, Y=Y,
                                                              sub_ids_dat=sub_ids_dat)
                                           for i in range(10))

        pd.DataFrame(dict(r_perm=[p[0] for p in permuted],
                     r2_perm=[p[1] for p in permuted],
                     rmse_perm=[p[2] for p in permuted])).to_csv(opj(outpath,
                                                                     'permsamples',
                                                                     outbootfile[0]))


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


# ###################################################################
# # Calculate ROC accuracy between levels
# ###################################################################

stats = pd.read_csv(opj(outpath, name + '_cvstats.csv'))
comparisons = [[1, 5], [1, 10], [5, 10]]

roc_results = dict(accuracy=[], accuracy_se=[], accuracy_p=[],
                   comparison=comparisons)

for c in comparisons:
    inputs = np.asarray(stats.Y_pred[stats.Y_true.isin(c)])
    outcome = list(stats.Y_true[stats.Y_true.isin(c)])
    outcome = np.where(outcome == np.min(outcome), 0, 1).astype(bool)

    subs = np.asarray(stats.subject_id[stats.Y_true.isin(c)], dtype=object)
    subs = [int(s[4:]) for s in stats.subject_id[stats.Y_true.isin(c)]]
    subs = np.asarray(subs, dtype=object)


    roc = Roc(input_values=inputs,
              binary_outcome=outcome)

    roc.calculate()
    roc.summary()
    roc_results['accuracy'].append(roc.accuracy)
    roc_results['accuracy_se'].append(roc.accuracy_se)
    roc_results['accuracy_p'].append(roc.accuracy_p)

# Save
roc_results = pd.DataFrame(roc_results).to_csv(opj(outpath,
                                                   name + '_roc_results.csv'))


####################################################################
# Run the prediction in a parcellation
# ###################################################################
parcel_pcr = Pipeline(steps=[
                           ('scaler', StandardScaler()),
                           ('pca', PCA(0.8)),
                           ('lasso', LinearRegression())])

def pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]
pears_score = make_scorer(pearson)

# Load parcellation, resample and apply groupmask
from nilearn.image import resample_to_img
parcel = apply_mask(resample_to_img(load_img(opj(basepath, 'external',
                                                 'CANlab_2018_combined_atlas_2mm.nii.gz')),
                                    group_mask, interpolation='nearest'), group_mask)
parcel = np.round(parcel)
from sklearn.model_selection import cross_val_score
parcels_score, par_lab = [], []
parcels_score_map = parcel.copy()
for par in tqdm(np.unique(parcel[parcel != 0])):

    X_par = X[:, np.where(parcel == par)[0]]
    cv = GroupKFold(n_splits=10)
    score = cross_val_score(parcel_pcr, X=X_par, y=Y, scoring=pears_score,
                            groups=sub_ids_dat, cv=cv)
    parcels_score.append(np.mean(score))
    parcels_score_map[np.where(parcel == par)[0]] = np.mean(score)
    par_lab.append(par)


# Rerun in top parcel just to get all stats
parcel_max = np.where(parcel == par_lab[np.argmax(parcels_score)], 1, 0)
X_par = X[:, np.where(parcel == parcel_max)[0]]
r2 = np.mean(cross_val_score(parcel_pcr, X=X_par, y=Y, scoring="r2",
                        groups=sub_ids_dat, cv=cv))
r = np.mean(cross_val_score(parcel_pcr, X=X_par, y=Y, scoring=pears_score,
                        groups=sub_ids_dat, cv=cv))
rmse = np.sqrt(np.mean(cross_val_score(parcel_pcr, X=X_par, y=Y, scoring=make_scorer(mean_squared_error),
                        groups=sub_ids_dat, cv=cv)))


# Save for plot
unmask(parcel_max, group_mask).to_filename(opj(outpath, name + '_parcels_max_cv.nii'))
unmask(parcels_score_map, group_mask).to_filename(opj(outpath, name + '_parcels_scores_cv.nii'))

# Run in an incrementaly larger number of parcels
top_par = np.asarray(par_lab)[np.flip(np.argsort(parcels_score))]

total_mask = np.zeros(parcel.shape[0])
all_scores, total_masks, n_vox = [], [],[]
for par in tqdm(top_par):
    par_mask = np.where(parcel == par, 1, 0)
    total_mask = par_mask + total_mask
    total_mask = np.squeeze(np.where(total_mask != 0, 1, 0))
    X_par = X[:, total_mask != 0]
    total_masks.append(total_mask)
    n_vox.append(np.sum(total_mask))
    score = cross_val_score(parcel_pcr, X=X_par, y=Y, scoring=pears_score,
                            groups=sub_ids_dat, cv=GroupKFold(n_splits=10))

    all_scores.append(np.mean(score))


np.save(opj(outpath, name + '_incremental_parcel_scores.npy'), all_scores)
np.save(opj(outpath, name + '_parcels_scores.npy'), parcels_score)
np.save(opj(outpath, name + '_parcels_labels.npy'), top_par)
np.save(opj(outpath, name + '_nvox.npy'), n_vox)


###################################################################
# Univariate with nltools
###################################################################

meta = pd.read_csv(opj(model_dir, 'metadata.csv'))

# Get image in specified path
subs, imgpath = [], []
# Load beta images
for ii in meta['imgpath']:
    subs.append(ii.split('/')[-1][:6])
    imgpath.append(opj(model_dir, ii.split('/')[-1]))
meta['subject_id'] = subs
meta['imgpath'] = imgpath

# KEep only pain images
meta = meta[meta['painfirst'] == 0]

sub_excl = ['sub-04', 'sub-08', 'sub-16',
            'sub-22', 'sub-23', 'sub-29',
            'sub-24', 'sub-30', 'sub-34',
            'sub-71']

meta = meta[~meta['subject_id'].isin(sub_excl)]


# Subjects
sub_ids_dat = np.asarray(meta['subject_id']).astype(object)

# Pain level
Y = np.asarray(meta['level']).astype(np.float)

nldata = Brain_Data(list(meta['imgpath']), mask=group_mask, X=meta)

# Univariate regression to predict level
all_sub_betas = Brain_Data()
all_subs = []
for s in tqdm(meta['subject_id'].unique()):
    sdat = nldata[np.where(meta['subject_id'] == s)[0]]
    sdat.X = pd.DataFrame(data={'Intercept':np.ones(sdat.shape()[0]), name:sdat.X['level']})
    stats = sdat.regress()
    all_sub_betas = all_sub_betas.append(stats['beta'][1])
    all_subs.append(s)

# Threhsold and save
t_stats = all_sub_betas.ttest(threshold_dict={'unc':.001})
t_stats['thr_t'].to_nifti().to_filename(opj(outpath, name + '_univariate_unc001.nii'))
t_stats['t'].to_nifti().to_filename(opj(outpath, name + '_univariate_unthresholded.nii'))

t_stats = all_sub_betas.ttest(threshold_dict={'fdr':.05})
t_stats['thr_t'].to_nifti().to_filename(opj(outpath, name + '_univariate_fdr05.nii'))

t_stats = all_sub_betas.ttest(threshold_dict={'holm-bonf':.05})
t_stats['thr_t'].to_nifti().to_filename(opj(outpath, name + '_univariate_fwe05.nii'))



# Cross-validated univariate beta maps for comparison with multivariate
# Use same cv scheme as mvpa
outer_cv = GroupKFold(n_splits=10).split(Y, groups=sub_ids_dat)
uni_cv_maps = []
for train, test in outer_cv:
    subs_idx = [all_subs.index(np.unique(sub_ids_dat[train])[i])
                for i in range(len(np.unique(sub_ids_dat[train])))]
    uni_cv_maps.append(all_sub_betas[subs_idx].mean().to_nifti())


concat_imgs(uni_cv_maps).to_filename(opj(outpath,
                                          name + '_unviariate_betas_unthresholded_xval.nii.gz'))


