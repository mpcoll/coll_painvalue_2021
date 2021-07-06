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
from nilearn.decoding.searchlight import search_light


###################################################################
# Paths
###################################################################

name = 'anticipation' # Analysis name
basepath = '/data'
prepdir = opj(basepath, 'derivatives/fmriprep')
model_dir = opj(basepath, 'derivatives/glms/model_anticipation')
outpath =  opj(basepath, 'derivatives/mvpa/anticipation_level')
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
Y = np.asarray(meta['level']).astype(np.float)

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

    # Collect fold metrics
    r_folds[test] = pearsonr(Y[test], Y_pred[test])[0]
    r2_folds[test] = r2_score(Y[test], Y_pred[test])
    rmse_folds[test] = np.sqrt(mean_squared_error(Y[test], Y_pred[test]))

# Collect overall metrics
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
                                verbose=0)(delayed(bootstrap_weights)(X=X, Y=Y,
                                                                      sub_ids_dat=sub_ids_dat)
                                           for i in range(nstop))
        bootstrapped = np.stack(bootstrapped)
        np.save(opj(outpath, 'bootsamples', outbootfile[0]), bootstrapped)

# Load all boostraps
bootstrapped = np.vstack(np.stack([np.load(opj(outpath, 'bootsamples', f))
                         for f in os.listdir(opj(outpath, 'bootsamples'))
                         if 'bootsamples' in f], axis=0))

assert bootstrapped.shape[0] == nbootstraps

# Get bootstraped statistics and threshold (as in nltools)
# Zscore and save
boot_z = bootstrapped.mean(axis=0)/bootstrapped.std(axis=0)
unmask(boot_z, group_mask).to_filename(opj(outpath, name + '_bootz.nii'))

# P vals and save
boot_pval =  2 * (1 - norm.cdf(np.abs(boot_z)))
unmask(boot_pval, group_mask).to_filename(opj(outpath,
                                              name + '_boot_pvals.nii'))

# Various thresholds and save
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

# ###################################################################d
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


###################################################################
# Univariate with nltools
###################################################################

from nltools import Brain_Data

meta = pd.read_csv(opj(model_dir, 'metadata.csv'))

# Get image in specified path
subs, imgpath = [], []
# Load beta images
for ii in meta['imgpath']:
    subs.append(ii.split('/')[-1][:6])
    imgpath.append(opj(model_dir, ii.split('/')[-1]))
meta['subject_id'] = subs
meta['imgpath'] = imgpath

# Subjects
sub_ids_dat = np.asarray(meta['subject_id']).astype(object)

# Pain level
Y = np.asarray(meta['level']).astype(np.float)

nldata = Brain_Data(list(meta['imgpath']), mask=group_mask, X=meta)

# Univariate regression to predict level
all_sub_betas = Brain_Data()
for s in tqdm(meta['subject_id'].unique()):
    sdat = nldata[np.where(meta['subject_id'] == s)[0]]
    sdat.X = pd.DataFrame(data={'Intercept':np.ones(sdat.shape()[0]), name:sdat.X['level']})
    stats = sdat.regress()
    all_sub_betas = all_sub_betas.append(stats['beta'][1])

# Threhsold and save
t_stats = all_sub_betas.ttest(threshold_dict={'unc':.001})
t_stats['thr_t'].to_nifti().to_filename(opj(outpath, name + '_univariate_unc001.nii'))
t_stats['t'].to_nifti().to_filename(opj(outpath, name + '_univariate_unthresholded.nii'))
t_stats['thr_t'].plot()
t_stats = all_sub_betas.ttest(threshold_dict={'fdr':.05})
t_stats['thr_t'].to_nifti().to_filename(opj(outpath, name + '_univariate_fdr05.nii'))
t_stats = all_sub_betas.ttest(threshold_dict={'holm-bonf':.05})
t_stats['thr_t'].to_nifti().to_filename(opj(outpath, name + '_univariate_fwe05.nii'))


#######################################################################
# Plot univariate for supp materials
#######################################################################
import nilearn.plotting as npl
import matplotlib.pyplot as plt
def plot_multi_supp(mapunt, map001, mapfdr05, outfigpath, display_mode='x',
                    cut_range=range(-60, 70, 10), title_offset=0.1,
                    figsize=(20, 10), fileext='.svg',
                    vmax=5,
                    cmap_label='Z score',
                    bgimg=None,
                    name3='FDR',
                    title='somemap', cmap='Specral_r'):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(3, 1, 1,)
    disp = npl.plot_stat_map(mapunt,
                         display_mode=display_mode,
                         cut_coords=cut_range,
                         annotate=False,
                         vmax=vmax,
                         cmap=cmap,
                         black_bg=False,
                         bg_img=bgimg,
                         draw_cross=False, colorbar=True,
                         axes=ax)
    disp.title(title + ' - Unthresholded', size=25, y=1.00 + title_offset,
            bgcolor="w", color='k')
    disp.annotate(left_right=False, size=15)
    lab = disp._colorbar_ax.get_yticklabels()
    disp._colorbar_ax.set_yticklabels(lab, fontsize=15)
    disp._colorbar_ax.set_ylabel(cmap_label, rotation=90, fontsize=20,
                                labelpad=10)
    ax = fig.add_subplot(3, 1, 2)
    disp = npl.plot_stat_map(map001,
                         display_mode=display_mode,
                         cut_coords=cut_range,
                         annotate=False,
                         vmax=vmax,
                         cmap=cmap,
                         bg_img=bgimg,
                         black_bg=False,
                         colorbar=False,
                         axes=ax)
    disp.title(title + ' - p < 0.001 uncorrected', size=25,
               y=1.00 + title_offset,
               bgcolor="w", color='k')
    disp.annotate(left_right=False, size=15)
    ax = fig.add_subplot(3, 1, 3)
    disp = npl.plot_stat_map(mapfdr05,
                         display_mode=display_mode,
                         cut_coords=cut_range,
                         annotate=False,
                         vmax=vmax,
                         cmap=cmap,
                         bg_img=bgimg,
                         black_bg=False,
                         draw_cross=False,
                         colorbar=False,
                         axes=ax)
    disp.title(title + ' - ' + name3 + ' p < 0.05', size=25,
               y=1.00 + title_offset,
               bgcolor="w", color='k')
    disp.annotate(left_right=False, size=15)

    fig.savefig(opj(outfigpath, title + '_variousthresh' + fileext ),
                dpi=600, bbox_inches='tight')


map_unthr = painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/anticipation_level',
                           'anticipation_univariate_unthresholded.nii'))
map_001 = painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/anticipation_level',
                           'anticipation_univariate_unc001.nii'))
map_fwe = painmapfdr = load_img(opj(basepath, 'derivatives/mvpa/anticipation_level',
                           'anticipation_univariate_fwe05.nii'))
thr = np.max(apply_mask(map_unthr, group_mask))
outfigpath =  opj(basepath, 'derivatives/figures')
bgimg = opj(basepath, 'external/tpl-MNI152NLin2009cAsym_space-MNI_res-01_T1w_brain.nii')
# Font
plt.rcParams['font.family'] = 'Helvetica'
plot_multi_supp(map_unthr,
                map_001,
                map_fwe,
                display_mode='z',
                outfigpath=outfigpath, cut_range=range(-40, 80, 10),
                title_offset=0.18, figsize=(20, 9),
                fileext='.png', bgimg=bgimg,
                vmax=thr,
                cmap_label='T-value',
                name3='FWE',
                title='Parametric effect of anticipated shock intensity',
                cmap=npl.cm.cold_hot)