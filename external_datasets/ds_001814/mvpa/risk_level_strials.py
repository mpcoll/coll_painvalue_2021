from nilearn.image import load_img, resample_to_img
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn.masking import apply_mask
from scipy.spatial.distance import cosine
import statsmodels.formula.api as smf
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
############################################################
# Paths
############################################################

name = 'risklevel' # Analysis name
basepath = '/data' # Basepath for data


sourcedir = opj(basepath, 'external', 'ds_001814')
model_dir  = opj(basepath, 'external', 'ds_001814', 'derivatives/glms/model_risk_strials')
group_mask = load_img(opj(basepath, 'external', 'ds_001814', 'derivatives/group_mask.nii.gz'))
outpath =  opj(basepath, 'external', 'ds_001814', 'derivatives/mvpa/risk_offer_level_strials')
prepdir = opj(basepath, 'external', 'ds_001814', 'derivatives/fmriprep')


if not os.path.exists(outpath):
        os.makedirs(outpath)

###################################################################
# Get data for prediction
###################################################################

# Exclusion based on participants.csv
sub_excl = ['sub-hc035', 'sub-hc024', 'sub-hc020', 'sub-hc018',
            'sub-hc016', 'sub-hc014', 'sub-hc012', 'sub-hc003']

meta = pd.read_csv(opj(model_dir, 'metadata.csv'))

# Get image in specified path
subs, imgpath = [], []
for ii in meta['imgpathz']:
    subs.append(ii.split('/')[-1][:9])
    imgpath.append(opj(model_dir, ii.split('/')[-1]))
meta['subject_id'] = subs
meta['imgpath'] = imgpath

meta = meta[~meta['subject_id'].isin(sub_excl)]


# Subjects
sub_ids_dat = np.asarray(meta['subject_id']).astype(object)

# Pain level
Y = np.asarray(meta['level']).astype(np.float)

# Brain Data
X = []
X += [apply_mask(fname, group_mask) for fname in tqdm(meta['imgpath'])]
X = np.squeeze(np.stack(X))


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

# Load patterns and apply mask from this study
pvp = apply_mask(resample_to_img(load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                   'painlevel_weightsxvalmean.nii')), group_mask), group_mask)

meta = pattern_expression_nocv(X, pvp, meta, 'pvp')

meta['level_z'] = zscore(meta['level'])
meta['pvp_cosine_z'] = zscore(meta['pvp_cosine'])

# Test relationship with mixed effect
md = smf.mixedlm("pvp_cosine_z ~ level_z", meta,
                 groups=meta["subject_id"])
mdf = md.fit()
print(mdf.summary())


###################################################################
# Plot
###################################################################

# colors
current_palette = sns.color_palette('colorblind', 6)
colp = current_palette[0]

# Label size
labelfontsize = 9
titlefontsize = np.round(labelfontsize*1.5)
ticksfontsize = np.round(labelfontsize*0.8)
legendfontsize = np.round(labelfontsize*0.8)

# Font
plt.rcParams['font.family'] = 'Helvetica'

# Despine
plt.rc("axes.spines", top=False, right=False)
meta['level'] = meta['level'].astype(int)

fig, ax = plt.subplots(figsize=(1.6, 2.2))

ax = sns.pointplot(y="pvp_cosine_z", x='level',
                   data=meta,
                   scale=0.4, ci=68, errwidth=1, color=colp)  # 68% CI are SEM
# Add labels
ax.set_title("",  {'fontsize': titlefontsize})
ax.set_xlabel("Pain risk", {'fontsize': labelfontsize})
ax.set_ylabel("Pattern similarity",
              {'fontsize': labelfontsize})
ax.set_xticklabels(['10%', '50%', '90%'], size=ticksfontsize)
ax.tick_params(axis='both', labelsize=ticksfontsize)

fig.tight_layout()
fig.savefig(opj(basepath, 'derivatives', 'figures', 'zorowitz_risk_levels_pvp.svg'),
            transparent=True, )
