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
import seaborn as sns
from nilearn import image, plotting
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colorbar, colors

###################################################################
# Paths
###################################################################

name = 'paincompvalue' # Analysis name
basepath = '/data'
prepdir = opj(basepath, 'derivatives/fmriprep')
model_dir = opj(basepath, 'derivatives/glms/model_offer1')
outpath =  opj(basepath, 'derivatives/mvpa/pain_compvalue')
if not os.path.exists(outpath):
    os.makedirs(outpath)

###################################################################
# Computing parameters
###################################################################

nbootstraps = 10000
nstop = 200 # frequency to stop/save bootstraps
njobs = 15 # Number of cpus to use

# Analysis mask
group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))
###################################################################
# Get sub


###################################################################
# Get data for prediction
###################################################################

meta = pd.read_csv(opj(model_dir, 'metadata.csv'))
meta2 = pd.read_csv(opj(basepath, 'derivatives', 'behavioural',
                        'behavioural_with_compestimates.csv')).groupby(['sub', 'pain_rank']).mean().reset_index()

from scipy import stats
meta['sv_pain'] = 999999.0
meta['sv_pain_z'] = 999999.0

for sub in meta2['sub'].unique():
    sub_dat = meta2[meta2['sub'] == sub]
    sub_dat['sv_pain_z'] = stats.zscore(sub_dat['sv_pain'])
    # sub_dat['sv_pain_log'] = stats.zscore(sub_dat['sv_pain_log'])

    for plev in sub_dat['pain_rank']:
        row = np.where((meta['subject_id'] == sub) & (meta['painfirst'] == 1) & (meta['pain_rank'] == plev))[0][0]
        meta.set_value(row, 'sv_pain_z', np.asarray(sub_dat[sub_dat['pain_rank'] == plev]['sv_pain_z'])[0])
        meta.set_value(row, 'k_pain', np.asarray(sub_dat[sub_dat['pain_rank'] == plev]['k_pain'])[0])



# Get image in specified path
subs, imgpath = [], []
for ii in meta['imgpathz']:
    subs.append(ii.split('/')[-1][:6])
    imgpath.append(opj(model_dir, ii.split('/')[-1]))
meta['subject_id'] = subs
meta['imgpath'] = imgpath

# Keep only pain images
meta = meta[meta['painfirst'] == 1]

# Subjects
sub_ids_dat = np.asarray(meta['subject_id']).astype(object)


# Pain level
Y = np.asarray(meta['sv_pain_z']).astype(np.float)

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



###################################################################
# Calculate similarity with
###################################################################

# Correlation between PVP and computational estimates predictive pattern
weights_xval_mean_level = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level/painlevel_weightsxvalmean.nii')), group_mask)
weights_xval_mean_value = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/pain_compvalue/paincompvalue_weightsxvalmean.nii')), group_mask)

pearsonr(weights_xval_mean_value, weights_xval_mean_level)


# Plot overlap in bootstrapped FDR corrected maps
pvp_fdr = load_img(opj(basepath,
                       'derivatives/mvpa/pain_offer_level/painlevel_bootz_unc001.nii'))
comp_fdr = load_img(opj(basepath,
                        'derivatives/mvpa/pain_compvalue/paincompvalue_bootz_unc001.nii'))





###################################################################
# PLot all for sup mat
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



pvp_fdr_mask = 1 * (np.abs(image.get_data(pvp_fdr)) > 3.0)
comp_fdr_mask = 2 * (np.abs(image.get_data(comp_fdr)) > 3.0)
combined_mask = image.new_img_like(pvp_fdr, pvp_fdr_mask + comp_fdr_mask)

fig, (img_ax, cbar_ax) = plt.subplots(
    1,
    2,
    gridspec_kw={"width_ratios": [10.0, 0.1], "wspace": 0.0},
    figsize=(10, 2),
)

cmap = ListedColormap([colp, colc, cole])
plotting.plot_roi(combined_mask, cmap=cmap, axes=img_ax, display_mode="x")
norm = colors.Normalize(vmin=0, vmax=3)
cbar = colorbar.ColorbarBase(
    cbar_ax,
    ticks=[0.5, 1.5, 2.5],
    norm=norm,
    orientation="vertical",
    cmap=cmap,
    spacing="proportional",
)
cbar_ax.set_yticklabels(["PVP", "Comp.\nestimates", "Both"])
# Plot to get cbar
fig.savefig(opj(outpath, 'pvp_over_comppattern_cbar.svg'),
             transparent=True, dpi=800)

# Plot same coordinates as PVP

# view_img(combined_mask, bg_img=bgimg)
# PLot slices
to_plot = {'x': [-38, -8, -6, 6, 10, 12],
           'y': [10],
           'z': [-10, 6]}

for axis, coord in to_plot.items():
    for c in coord:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        disp = plotting.plot_roi(combined_mask, cmap=cmap, colorbar=False,
                        bg_img=bgimg,
                        dim=-0.3,
                        black_bg=False,
                        display_mode=axis,
                        axes=ax,
                        vmax=3,
                        cut_coords=(c,),
                        alpha=1,
                        annotate=False)
        disp.annotate(size=ticksfontsize, left_right=False)
        fig.savefig(opj(outpath, 'pmapvscompmap_fdr05_' + axis
                        + str(c) + '.svg'),
                    transparent=True, bbox_inches='tight', dpi=600)


from scipy.stats import zscore
fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
sns.regplot(zscore(weights_xval_mean_value), zscore(weights_xval_mean_level),
            color=colp,
            scatter_kws={"s": 0.2, 'rasterized': True,
                         'alpha': 0.05, 'color':'gray'},
            line_kws={'linewidth': 1})
ax1.set_ylabel('Comp. pattern weights', {'fontsize': labelfontsize})

ax1.set_title('',
              {'fontsize': titlefontsize})

ax1.set_xlabel('PVP pattern weights',
               {'fontsize': labelfontsize})

ax1.tick_params(axis='both', labelsize=ticksfontsize)


ax1.set_ylim((-6, 5))
ax1.set_xlim((-6, 5))

fig1.tight_layout()
fig1.savefig(opj(outpath, 'pvp_vs_comppattern.svg'),
             transparent=True, dpi=800)


