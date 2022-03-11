from nilearn.image import load_img
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from tqdm import tqdm

###################################################################
# Paths
###################################################################
name = 'decision' # Analysis name
basepath = '/data'
prepdir = opj(basepath, 'derivatives/fmriprep')
model_dir = opj(basepath, 'derivatives/glms/model_offer2')

outpath =  opj(basepath, 'derivatives/mvpa/decision')
if not os.path.exists(outpath):
    os.mkdir(outpath)

# Analysis mask
group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))

###################################################################
# Load data
###################################################################

print("Loading data...")

meta = pd.read_csv(opj(model_dir, 'metadata.csv'))

imgpath = []
for ii in meta['imgpathz']:
    imgpath.append(opj(model_dir, ii.split('/')[-1]))
meta['imgpath'] = imgpath

# Subjects
sub_ids_dat = np.asarray(meta['subject_id'])

# Brain Data
X = []
X += [apply_mask(fname, group_mask)
        for fname in tqdm(meta['imgpath'])]
X = np.squeeze(np.stack(X))
np.save(opj(outpath, name + '_features.npy'), X)
meta.to_csv(opj(outpath, name + '_stats.csv'))
