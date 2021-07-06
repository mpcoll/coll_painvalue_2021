from nilearn.image import load_img, new_img_like
import os
import numpy as np
from os.path import join as opj
import numpy as np
from nilearn import masking
from nilearn.image.resampling import coord_transform
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity

###################################################################
# Paths
###################################################################

basepath = '/data'
prepdir = opj(basepath, 'derivatives/fmriprep')

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
    all_masks += [load_img(opj(funcdir, f)).get_data() for f in os.listdir(funcdir)
                  if '_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz' in f]

    files += [opj(funcdir, f) for f in os.listdir(funcdir)
                  if '_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz' in f]

# In a single array
all_masks = np.stack(all_masks)

# Get proportion of voxels in mask
group_mask = np.where(np.sum(all_masks, axis=0)/all_masks.shape[0] >= msk_thrs, 1, 0)

# Make nifti and save
group_mask = new_img_like(files[0], group_mask)

group_mask.to_filename(opj(basepath, 'derivatives/group_mask.nii.gz'))


###################################################################
# Generate whole brain seachlight spheres mask to avoid do it each time
###################################################################

group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))

process_mask_img = None
mask_img = group_mask
radius = 6

# Get the seeds
process_mask_img = process_mask_img
if process_mask_img is None:
    process_mask_img = mask_img

# Compute world coordinates of the seeds
process_mask, process_mask_affine = masking._load_mask_img(
    process_mask_img)
process_mask_coords = np.where(process_mask != 0)
process_mask_coords = coord_transform(
    process_mask_coords[0], process_mask_coords[1],
    process_mask_coords[2], process_mask_affine)
process_mask_coords = np.asarray(process_mask_coords).T

# Get the masks for each sphere
_, spheres_matrix = _apply_mask_and_get_affinity(
                    seeds=process_mask_coords, niimg=None, radius=radius,
                    allow_overlap=True,
                    mask_img=mask_img)

np.savez(opj(basepath, 'derivatives/sl_6mmshperes.npz'),
         dtype=spheres_matrix.dtype.str, data=spheres_matrix.data,
    rows=spheres_matrix.rows, shape=spheres_matrix.shape)


