from nilearn.image import load_img, resample_to_img
from os.path import join as opj
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from scipy.spatial.distance import cosine

basepath = '/data'

group_mask = load_img(opj(basepath, 'derivatives/group_mask.nii.gz'))

widestats = pd.read_csv(opj(basepath, 'derivatives/behavioural/behavioural_data_wide.csv'))
widestats['accept_avg'] = widestats['accept']
widestats['rt_avg'] = widestats['rt']
varnames = ['coef_pain', 'accept_avg', 'rt_avg', 'coef_money', 'coef_pain_rt',
            'coef_money_rt', 'coef_pain_rt_acc', 'coef_pain_rt_rej',
            'coef_money_rt_acc', 'coef_money_rt_rej']
###################################################################
# Helper function
###################################################################

# Loop cv fold
def pattern_expression_cv(dat, cv_imgs, pattern_stats, stats, name):
    """Calculate similarity between maps using dot product and cosine product.
       Crossvalidated - to use with data from same dataset.

    Args:
        dat ([array]): images to calculate similarity on (array of shape n images x n voxels)
        cv_imgs ([array]): Pattern weights for each fold (n folds x n voxels)
        pattern_stats ([pd df]): Data frame with subejct id and folds for corresponding pattern
        stats ([pd df]): Data frame with subejct id and folds for corresponding patte
        name ([string]): Name to add to ouput columns
    Returns:
        [df]: stats df with dot and cosine columns added
    """
    pexpress = np.zeros(dat.shape[0]) + 9999
    cosim = np.zeros(dat.shape[0]) + 9999
    sub_ids_pat = np.asarray(pattern_stats['subject_id'])
    sub_ids_dat =  np.asarray(stats['subject_id'])

    for fold in np.unique(pattern_stats['folds_test']):
        # Pick weights for this fold
        cv_map = cv_imgs[int(fold-1), :]
        # Pick left out subs for this fold
        sub_folds = np.unique(sub_ids_pat[np.where(np.asarray(pattern_stats['folds_test']) == fold)])
        # Calculate on left out subjects
        for xx in range(dat.shape[0]):
            if sub_ids_dat[xx] in sub_folds:
                # Dot product
                pexpress[xx] = np.dot(dat[xx, :], cv_map)
                # Cosine distance
                cosim[xx] = 1- cosine(dat[xx, :], cv_map)

        stats[name + '_dot'] = pexpress
        stats[name + '_cosine'] = cosim
    return stats


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



def add_wide_data(longstats, widestats, varnames):
    """Add wide data to long data frames

    Args:
        longstats (pd): long dataframe
        widestats(pd): wide dataframe
        varnames (list): varnames to add to long

    Returns:
        longstats : stats with added variables from widestats
    """
    for v in varnames:
        longstats[v] = 9999
        for sub in widestats['subject_id'].unique():
            longstats.set_value(np.where(sub == longstats['subject_id'])[0],
                                v,
                                np.asarray(widestats[sub == widestats['subject_id']][v]))

    return longstats


###################################################################
# Load all datasets/patterns
###################################################################

# Pain value
# Load cross-validated maps
pvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                   'painlevel_weights_xval.nii.gz')), group_mask)

# Load cross-validated stats to get subs in each fold
pvp_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                            'painlevel_cvstats.csv'))
# Load pvp data
pvp_dat = np.load(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                         'painlevel_features.npy'))

# Money value
# Load cross-validated maps
mvp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                   'moneylevel_weights_xval.nii.gz')), group_mask)

# Load cross-validated stats to get subs in each fold
mvp_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/money_offer_level',
                            'moneylevel_cvstats.csv'))
# Load mvp data
mvp_dat = np.load(opj(basepath, 'derivatives/mvpa/money_offer_level',
                         'moneylevel_features.npy'))

# Shock intensity
# Load cross-validated maps
sip = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                   'shocklevel_weights_xval.nii.gz')), group_mask)

# Load cross-validated stats to get subs in each fold
si_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                            'shocklevel_cvstats.csv'))
# Load si data
si_dat = np.load(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                         'shocklevel_features.npy'))


# Anticipation
ant_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/anticipation_level',
                            'anticipation_cvstats.csv'))

antp = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/anticipation_level',
                   'anticipation_weights_xval.nii.gz')), group_mask)

ant_dat = np.load(opj(basepath, 'derivatives/mvpa/anticipation_level',
                         'anticipation_features.npy'))

# Decision
dec_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                            'decision_stats.csv'))
if 'sub' not in str(dec_stats['subject_id'][0]):
    dec_stats['subject_id'] = ['sub-' + str(s).zfill(2) for s in dec_stats['subject_id']]
dec_dat = np.load(opj(basepath, 'derivatives/mvpa/decision',
                      'decision_features.npy'))



###################################################################
# Apply pain value pattern
###################################################################
pvp_stats = pattern_expression_cv(pvp_dat, pvp, pvp_stats, pvp_stats, 'pvp_cv')
mvp_stats = pattern_expression_cv(mvp_dat, pvp, pvp_stats, mvp_stats, 'pvp_cv')
si_stats = pattern_expression_cv(si_dat, pvp, pvp_stats, si_stats, 'pvp_cv')
ant_stats = pattern_expression_cv(ant_dat, pvp, pvp_stats, ant_stats, 'pvp_cv')
dec_stats = pattern_expression_cv(dec_dat, pvp, pvp_stats, dec_stats, 'pvp_cv')

###################################################################
# Apply money value pattern
###################################################################
pvp_stats = pattern_expression_cv(pvp_dat, mvp, mvp_stats, pvp_stats, 'mvp_cv')
mvp_stats = pattern_expression_cv(mvp_dat, mvp, mvp_stats, mvp_stats, 'mvp_cv')
si_stats = pattern_expression_cv(si_dat, mvp, mvp_stats, si_stats, 'mvp_cv')
ant_stats = pattern_expression_cv(ant_dat, mvp, mvp_stats, ant_stats, 'mvp_cv')
dec_stats = pattern_expression_cv(dec_dat, mvp, mvp_stats, dec_stats, 'mvp_cv')

###################################################################
# Apply shock intensity pattern
###################################################################
pvp_stats = pattern_expression_cv(pvp_dat, sip, si_stats, pvp_stats, 'sip_cv')
mvp_stats = pattern_expression_cv(mvp_dat, sip, si_stats, mvp_stats, 'sip_cv')
si_stats = pattern_expression_cv(si_dat, sip, si_stats, si_stats, 'sip_cv')
ant_stats = pattern_expression_cv(ant_dat, sip, si_stats, ant_stats, 'sip_cv')
dec_stats = pattern_expression_cv(dec_dat, sip, si_stats, dec_stats, 'sip_cv')

###################################################################
# Apply anticipation pattern
###################################################################
pvp_stats = pattern_expression_cv(pvp_dat, antp, ant_stats, pvp_stats, 'antp_cv')
mvp_stats = pattern_expression_cv(mvp_dat, antp, ant_stats, mvp_stats, 'antp_cv')
si_stats = pattern_expression_cv(si_dat, antp, ant_stats, si_stats, 'antp_cv')
ant_stats = pattern_expression_cv(ant_dat, antp, ant_stats, ant_stats, 'antp_cv')
dec_stats = pattern_expression_cv(dec_dat, antp, ant_stats, dec_stats, 'antp_cv')


###################################################################
# Apply univariate pain pattern
###################################################################

unip_p = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                   'painlevel_unviariate_unthresholded_xval.nii.gz')), group_mask)

pvp_stats = pattern_expression_cv(pvp_dat, unip_p, pvp_stats, pvp_stats, 'unip_cv')
mvp_stats = pattern_expression_cv(mvp_dat, unip_p, pvp_stats, mvp_stats, 'unip_cv')
si_stats = pattern_expression_cv(si_dat, unip_p, pvp_stats, si_stats, 'unip_cv')
ant_stats = pattern_expression_cv(ant_dat, unip_p, pvp_stats, ant_stats, 'unip_cv')
dec_stats = pattern_expression_cv(dec_dat, unip_p, pvp_stats, dec_stats, 'unip_cv')


###################################################################
# Apply univariate money pattern
###################################################################

unim_p = apply_mask(load_img(opj(basepath, 'derivatives/mvpa/money_offer_level',
                   'moneylevel_unviariate_unthresholded_xval.nii.gz')), group_mask)

pvp_stats = pattern_expression_cv(pvp_dat, unim_p, mvp_stats, pvp_stats, 'unim_cv')
mvp_stats = pattern_expression_cv(mvp_dat, unim_p, mvp_stats, mvp_stats, 'unim_cv')
si_stats = pattern_expression_cv(si_dat, unim_p, mvp_stats, si_stats, 'unim_cv')
ant_stats = pattern_expression_cv(ant_dat, unim_p, mvp_stats, ant_stats, 'unim_cv')
dec_stats = pattern_expression_cv(dec_dat, unim_p, mvp_stats, dec_stats, 'unim_cv')


# ###################################################################
# # Apply external patterns
# ###################################################################

other_maps = {'nps': opj(basepath, 'external/wager_maps/',
                         'weights_NSF_grouppred_cvpcr.nii.gz'),
              'siips': opj(basepath, 'external/wager_maps/',
                           '2017_Woo_SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii'),
              'pines': opj(basepath, 'external/wager_maps/',
                           '2015_Chang_PLoSBiology_PINES/Rating_Weights_LOSO_2.nii')
              }

for name, path in other_maps.items():
    # Load img, resample to mask, apply mask
    pattern = apply_mask(resample_to_img(load_img(path),
                                         group_mask, interpolation='nearest'),
                         group_mask)
    # Apply to data
    pvp_stats = pattern_expression_nocv(pvp_dat, pattern, pvp_stats, name)
    mvp_stats = pattern_expression_nocv(mvp_dat, pattern, mvp_stats, name)
    si_stats = pattern_expression_nocv(si_dat, pattern, si_stats, name)
    dec_stats = pattern_expression_nocv(dec_dat, pattern, dec_stats, name)
    ant_stats = pattern_expression_nocv(ant_dat, pattern, ant_stats, name)

###################################################################
# Add wide data
###################################################################

pvp_stats = add_wide_data(pvp_stats, widestats, varnames)
mvp_stats = add_wide_data(mvp_stats, widestats, varnames)
si_stats = add_wide_data(si_stats, widestats, varnames)
dec_stats = add_wide_data(dec_stats, widestats, varnames)
ant_stats = add_wide_data(ant_stats, widestats, varnames)

###################################################################
# Add computational estimates
###################################################################

# PVP
comp_data = pd.read_csv(opj(basepath, 'derivatives', 'behavioural',
                        'behavioural_with_compestimates.csv')).groupby(['sub',
                                                                         'pain_rank']).mean().reset_index()
for sub in pvp_stats['subject_id'].unique():
    sub_dat = comp_data[comp_data['sub'] == sub]
    for plev in sub_dat['pain_rank']:
        row = np.where((pvp_stats['subject_id'] == sub) & (pvp_stats['Y_true'] == plev))[0][0]
        pvp_stats.set_value(row, 'sv_pain', np.asarray(sub_dat[sub_dat['pain_level'] == plev]['sv_pain'])[0])
        pvp_stats.set_value(row, 'k_pain', np.asarray(sub_dat[sub_dat['pain_level'] == plev]['k_pain'])[0])
        pvp_stats.set_value(row, 'sv_both', np.asarray(sub_dat[sub_dat['pain_level'] == plev]['sv_both'])[0])


# Decision
comp_data = pd.read_csv(opj(basepath, 'derivatives', 'behavioural',
                        'behavioural_with_compestimates.csv'))

dec_stats['trials'] = dec_stats['trials'] + 20*np.asarray(dec_stats['run']-1)
dec_stats = dec_stats[(dec_stats['duration'] < 5.0)
                            & (dec_stats['duration'] > 0.2)]

new_dec_stats = []
for sub in dec_stats['subject_id'].unique():
    sub_dat = comp_data[comp_data['sub'] == sub].reset_index()
    sub_dec_stats = dec_stats[dec_stats['subject_id'] == sub].reset_index()
    assert (np.asarray(sub_dat['trials']) == np.asarray(sub_dec_stats['trials'])).all()
    sub_dec_stats['k_pain'] = np.asarray(sub_dat['k_pain'])
    sub_dec_stats['sv_pain'] = np.asarray(sub_dat['sv_pain'])
    sub_dec_stats['sv_both'] = np.asarray(sub_dat['sv_both'])
    new_dec_stats.append(sub_dec_stats)
dec_stats = pd.concat(new_dec_stats).reset_index()
dec_stats = dec_stats.loc[:, ~dec_stats.columns.str.contains('^Unnamed')]

###################################################################
# Save all
###################################################################

pvp_stats.to_csv(opj(basepath, 'derivatives/mvpa/pain_offer_level',
                     'painlevel_cvstats.csv'))
mvp_stats.to_csv(opj(basepath, 'derivatives/mvpa/money_offer_level',
                     'moneylevel_cvstats.csv'))
si_stats.to_csv(opj(basepath, 'derivatives/mvpa/shock_intensity_level',
                    'shocklevel_cvstats.csv'))
dec_stats.to_csv(opj(basepath, 'derivatives/mvpa/decision', 'decision_stats_out.csv'))

ant_stats.to_csv(opj(basepath, 'derivatives/mvpa/anticipation_level',
                     'anticipation_cvstats.csv'))
