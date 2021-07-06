from nilearn.image import load_img, new_img_like
import os
from os.path import join as opj
import numpy as np

###################################################################
# Paths
###################################################################

basepath = '/data'
prepdir = opj(basepath, 'external/ds_001814/derivatives/fmriprep')

###################################################################
# Get sub
###################################################################
subs_id = [s for s in os.listdir(prepdir) if 'sub-' in s
           and '.html' not in s]

# Build group mask
msk_thrs = 1 # Proportion of sessions mask with this voxel to incude

# Load all part masks data
all_masks, files = [], []
for s in subs_id:
    funcdir = opj(prepdir, s, 'func')
    # Get masks for all sessions
    all_masks += [load_img(opj(funcdir, f)).get_data()
                  for f in os.listdir(funcdir)
                  if '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz' in f]
    files += [opj(funcdir, f) for f in os.listdir(funcdir)
                  if '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz' in f]

# In a single array
all_masks = np.stack(all_masks)

# Get proportion of voxels in mask
group_mask = np.where(np.sum(all_masks,
                             axis=0)/all_masks.shape[0] >= msk_thrs, 1, 0)


# Make it NIFTI
group_mask = new_img_like(files[0], group_mask)
# Save
group_mask.to_filename(opj(basepath,
                           'external/ds_001814/derivatives/group_mask.nii.gz'))
