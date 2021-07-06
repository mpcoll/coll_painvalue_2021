#-*- coding: utf-8  -*-

import os
import pandas as pd
import numpy as np
import warnings
import sys
from os.path import join as opj
from joblib import Parallel, delayed
from nilearn.image import load_img, concat_imgs
from nilearn.glm.first_level import  first_level_from_bids, FirstLevelModel, make_first_level_design_matrix

###################################################################
# OS specific parameters
###################################################################

basepath = '/lustre04/scratch/mpcoll/2020_npng_newprep'
# basepath = '/data'

# Uses as lot of memory so keep this relatively low
# if RAM is limited or try caching
njobs = 15

###################################################################
# Fixed parameters
###################################################################

# Paths
sourcepath = opj(basepath, 'source')
preppath = opj(basepath, 'derivatives/fmriprep')
outpath  = opj(basepath, 'derivatives/glms/model_shocks')
group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))
if not os.path.exists(outpath):
        os.mkdir(outpath)

# Number of scans/session
nscans_sess = 510

# Fmriprep space
space = 'MNI152NLin2009cAsym'

# Confounds to use in GLM
confounds_use = ['trans_x',
                 'trans_y',
                 'trans_z',
                 'rot_x',
                 'rot_y',
                 'rot_z',
                 'trans_x_derivative1',
                 'trans_y_derivative1',
                 'trans_z_derivative1',
                 'trans_x_power2',
                 'trans_y_power2',
                 'trans_z_power2',
                 'trans_x_derivative1_power2',
                 'trans_y_derivative1_power2',
                 'trans_z_derivative1_power2',
                 'rot_x_derivative1',
                 'rot_y_derivative1',
                 'rot_z_derivative1',
                 'rot_x_power2',
                 'rot_y_power2',
                 'rot_z_power2',
                 'rot_x_derivative1_power2',
                 'rot_y_derivative1_power2',
                 'rot_z_derivative1_power2',
                 'a_comp_cor_00',
                 'a_comp_cor_01',
                 'a_comp_cor_02',
                 'a_comp_cor_03',
                 'a_comp_cor_04']

###################################################################
# Generate models
###################################################################

models, imgs, events, confounds = \
first_level_from_bids(dataset_path=sourcepath,
                        task_label='npng',
                        space_label=space,
                        img_filters=[('desc', 'preproc')],
                        high_pass=1/128,
                        mask_img=group_mask,
                        hrf_model='spm',
                        smoothing_fwhm=6.0,
                        signal_scaling=0, # Default time axis scaling
                        n_jobs=1,
                        derivatives_folder=preppath)

# Slice timing reference slice from fmriprep website:
# Slice time correction is performed using AFNI 3dTShift.
# All slices are realigned in time to the middle of each TR.
for m in models:
    m.slice_time_ref = 0.5
# No relevant info printed, igonre user warning
warnings.simplefilter("ignore", UserWarning)


###################################################################
# Fit models
###################################################################

def run_first(model, img, event, conf, outpath, confounds_use):
    warnings.filterwarnings("ignore")

    # Create a new object to fit to avoid overloading memory (bug?)
    model_fit = FirstLevelModel(t_r=model.t_r,
                                slice_time_ref=model.slice_time_ref,
                                hrf_model=model.hrf_model,
                                high_pass=model.high_pass,
                                mask_img=model.mask_img,
                                smoothing_fwhm=model.smoothing_fwhm,
                                signal_scaling=0,
                                subject_label=model.subject_label,
                                standardize=model.standardize,
                                n_jobs=model.n_jobs)

    # Get part label
    s =  model.subject_label

    conf_out, events_out = [], []  # Loop sessions
    for run, (ev, cf) in enumerate(zip(event, conf)):
        # Get confounds
        # cf.columns = [c + '_r' +str(run+1) if 'motion_out' in c else c for c in cf.columns ]
        # conf_use = confounds_use + [c for c in cf.columns if 'motion_out' in c]
        conf_use = confounds_use
        conf_in = cf[conf_use]
        # As in SPM, add constant for first runs last is gobal constant.
        if run < len(event)-1:
            conf_in.loc[:, 'const_run_' + str(run + 1)] = 1
        # Replace nan for first scan
        conf_out.append(conf_in.fillna(0))

        # Arrange events in bids format
        ev.loc[:, 'subject_id'] = s
        ev.loc[:, 'session'] = run
        ev.loc[:, 'money_rank'] = np.round((ev['money_level']+1)/1.11)
        ev.loc[:, 'trials'] = np.arange(1 , len(ev)+1)
        ev.loc[:, 'offer1_duration'] = 2
        ev.loc[:, 'rt'] = list(ev.loc[:, 'offer2_duration']).copy()
        ev.loc[:, 'shock_duration_accept'] = 0
        ev.loc[:, 'choice_duration'] = 2 # Key press/feedback

        # Onsets
        ev_o = ev.melt(id_vars=['trials',
                                'run',
                                'subject_id',
                                'accept',
                                'painfirst',
                                'pain_rank',
                                'money_rank'],
                       value_vars=['offer1_onset', # Pain/money offers
                                   'offer2_onset', # Decisions
                                   'choice_onset', # Key press/feedback
                                   'delay3_onset_accept', # Anticipation
                                   'shock_onset_accept', # Shock
                                   ],
                                value_name='onset', var_name="trial_type")
        # Durations
        ev_d = ev.melt(id_vars=['trials',
                                'run',
                                'subject_id',
                                'accept',
                                'painfirst',
                                'pain_rank',
                                'money_rank'],
                       value_vars=['offer1_duration', # Pain/money offers
                                   'offer2_duration', # Decisions
                                   'choice_duration', # Key press/feedback
                                   'delay3_duration_accept', # Anticipation
                                   'shock_duration_accept', # Shock
                                   ],
                       value_name='duration', var_name="trial_type")

        ev_o.loc[:, 'duration'] = ev_d['duration']
        ev_o = ev_o[ev_o['onset'] != 9999] # Remove undefined events

        # Rename events according to trial type and pain/money level
        new_types, to_keep, level = [], [], []
        ev_o.loc[:, 'trial_type_old'] = ev_o['trial_type'][:]
        for tt, plev in zip(ev_o['trial_type'], ev_o['pain_rank']):
                if tt == 'shock_onset_accept':
                    to_keep.append(1)
                    new_types.append(str('shock_pain_level_' + str(int(plev))))
                    level.append(plev)
                else:
                    to_keep.append(0)
                    new_types.append(tt)
                    level.append(999)
        ev_o.loc[:, 'trial_type'] = new_types
        ev_o.loc[:, 'to_keep'] = to_keep
        ev_o.loc[:, 'level'] = level
        # Adjust onset to concatenate runs
        ev_o.loc[:, 'onset'] = ev_o['onset'] + (model.t_r*nscans_sess*run)
        events_out.append(ev_o)


    # Concatenate sessions
    img_cont = concat_imgs(img)
    img = None # Clear memory
    events_out_cont = pd.concat(events_out).reset_index(drop=True)
    conf_out_cont = pd.concat(conf_out, sort=True).reset_index(drop=True).fillna(0)

    # Fit
    model_fit.fit(img_cont, events_out_cont[['onset', 'duration', 'trial_type']],
                   conf_out_cont)

    # Save onsets duration to check
    events_out_cont[['onset', 'duration', 'trial_type']].to_csv(opj(outpath, 'sub-' + s + '_design.csv'))

    # Generate and save report with a random contrast
    report = model_fit.generate_report('offer2_onset')
    report.save_as_html(opj(outpath, 'sub-' + s + '_glm_report.html'))
    img_cont = None # Clear memory

    # Keep only events of interest
    events_all = pd.concat(events_out)
    events_save = events_all[events_all['to_keep'] == 1].groupby(['trial_type']).mean().reset_index()
    events_save.loc[:, 'subject_id'] = 'sub-' + s
    # assert len(events_save) == 20  # Sanity check

    # Save contrasts
    filenames, filenames_z = [], []
    cols, run_idx = [], []
    for idx, dm in enumerate(model_fit.design_matrices_):
        cols = cols + list(dm.columns)
        run_idx += list(np.repeat(idx+1, len(list(dm.columns))))
    for trial in np.unique(events_save['trial_type']):
            # Create the single trial contrast and split to get an array/session
            cont = np.where(np.asarray(cols) == trial, 1, 0)
            cont_run = []
            for ii in range(1, np.max(run_idx)+1):
                cont_run.append(cont[np.asarray(run_idx) == ii])
            # Contrast
            map = model_fit.compute_contrast(cont_run, output_type='effect_size')
            # Save
            fname = opj(outpath, 'sub-' + s + '_' + trial + '.nii.gz')
            map.to_filename(fname)
            filenames.append(fname.split('/')[-1])
            # Compute z also
            map = model_fit.compute_contrast(cont_run, output_type='z_score')
            # Save
            fname = opj(outpath, 'sub-' + s + '_' + trial + '_z.nii.gz')
            map.to_filename(fname)
            filenames_z.append(fname.split('/')[-1])
    # events_save.loc[:, 'imgpath'] = filenames
    events_save.loc[:, 'imgpathz'] = filenames_z
    # Clear memory
    model_fit = None
    return events_save

# Run in parallel
metadata = Parallel(n_jobs=njobs,
                        verbose=11)(delayed(run_first)(model=models[i],
                                                      img=imgs[i],
                                                      event=events[i],
                                                      confounds_use=confounds_use,
                                                      conf=confounds[i],
                        outpath=outpath) for i in range(len(models)))

metadata = pd.concat(metadata)
metadata.to_csv(opj(outpath, 'metadata.csv'))

# To keep track in slurm output
print('########################## FINISHED ' + str(sys.argv[0]) + ' ##################################')
