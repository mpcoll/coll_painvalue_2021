import os
from os.path import join as opj
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from scipy.stats import binom_test, ttest_1samp, ttest_rel, pearsonr

basepath = '/data'

# Outpath
outpath =  opj(basepath, 'derivatives/mvpa/decision')
if not os.path.exists(outpath):
    os.mkdir(outpath)

###################################################################
# Load data
###################################################################
# Decision data
dec_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                            'decision_stats.csv'))

# Remove trials with high vif, unanswered trials (duration == 5)
dec_stats_clean = dec_stats[(dec_stats['vifs'] < 2)
                            & (dec_stats['duration'] < 5.0)
                            & (dec_stats['duration'] > 0.2)]

perc_removed = (len(dec_stats)- len(dec_stats_clean))/len(dec_stats)

###################################################################
# Set up CV prediction
###################################################################

# Classifier
clf = Pipeline(steps=[('scaler', StandardScaler()),
                      ('svm', LinearSVC())])
# CV split
cv = GroupKFold(10)

# Data frame to collect results
out = pd.DataFrame(index=np.unique(dec_stats_clean['subject_id']))

# Make balanced accuracy scorer
bal_acc = make_scorer(balanced_accuracy_score)

###################################################################
# Predict choice using pain
###################################################################
# Get data
X = np.asarray(dec_stats_clean['pvp_cv_cosine']).reshape(-1, 1)
Y = np.asarray(dec_stats_clean['accept']).astype(int)

# Cross val prediction
pvp_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                              scoring=bal_acc, cv=cv)

# Binomial test
pvp_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(pvp_bal_acc)),
                         len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(pvp_bal_acc))] )


# Save data to
out['pvp_bal_acc'] = np.mean(pvp_bal_acc)
out['pvp_pval'] = pvp_pval

###################################################################
# Predict choice using money
###################################################################

X = np.asarray(dec_stats_clean['mvp_cv_cosine']).reshape(-1, 1)
Y = np.asarray(dec_stats_clean['accept']).astype(int)


mvp_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                          scoring=bal_acc, cv=cv)

# Binomial test
mvp_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(mvp_bal_acc)),
              len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(mvp_bal_acc))] )


out['mvp_bal_acc'] = np.mean(mvp_bal_acc)
out['mvp_pval'] = mvp_pval


###################################################################
# Predict choice using both
###################################################################

X = np.asarray([dec_stats_clean['pvp_cv_cosine'], dec_stats_clean['mvp_cv_cosine']]).T
Y = np.asarray(dec_stats_clean['accept']).astype(int)


bal_acc = make_scorer(balanced_accuracy_score)


both_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                          scoring=bal_acc, cv=cv)

# Binomial test
both_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(both_bal_acc)),
              len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(both_bal_acc))] )


out['both_bal_acc'] = np.mean(both_bal_acc)
out['both_pval'] = both_pval


# Refit on all to get SVM boundaries for plot
clf.fit( X, Y)
model_out = dict(clf=clf, x=X, y=Y, name='painmoney')
# Save model for plot
np.save(opj(outpath, 'model_painmoney.npy'), model_out)


# ###################################################################
# # Same but at the participant level
# ###################################################################

pvp_subs, mvp_subs, both_subs = [], [], [], []
for sub in np.unique(dec_stats_clean['subject_id']):

    sub_dat = dec_stats_clean[dec_stats_clean['subject_id'] == sub]
    Y = np.asarray(sub_dat['accept']).astype(int)
    cv = StratifiedKFold(10)

    # PVP
    X = np.asarray(sub_dat['pvp_cv_cosine']).reshape(-1, 1)
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'pvp_bal_acc_sub'] = sub_acc
    pvp_subs.append(sub_acc)

    # MVP
    X = np.asarray(sub_dat['mvp_cv_cosine']).reshape(-1, 1)
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'mvp_bal_acc_sub'] = sub_acc
    mvp_subs.append(sub_acc)

    # Both
    X = np.asarray([sub_dat['mvp_cv_cosine'], sub_dat['pvp_cv_cosine']]).T
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'both_bal_acc_sub'] = sub_acc
    both_subs.append(sub_acc)


# Save data
out['pvp_bal_acc_sub_mean'] = np.mean(pvp_subs)
out['mvp_bal_acc_sub_mean'] = np.mean(mvp_subs)
out['both_bal_acc_sub_mean'] = np.mean(both_subs)

# One sample t-tests against chance accuracy
out['pvp_bal_acc_sub_tval'], out['pvp_bal_acc_sub_pval'] = ttest_1samp(np.asarray(pvp_subs), 0.5)
out['mvp_bal_acc_sub_tval'], out['mvp_bal_acc_sub_pval'] = ttest_1samp(np.asarray(mvp_subs), 0.5)
out['both_bal_acc_sub_tval'], out['both_bal_acc_sub_pval'] = ttest_1samp(np.asarray(both_subs), 0.5)

out['both_vs_mvp_acc_sub_tval'], out['both_vs_mvp_acc_sub_pval'] = ttest_rel(np.asarray(both_subs), np.asarray(mvp_subs))
out['pvp_vs_mvp_acc_sub_tval'], out['pvp_vs_mvp_acc_sub_pval'] = ttest_rel(np.asarray(pvp_subs), np.asarray(mvp_subs))


# Save
out = out.reset_index()
out['subject_id'] = out['index']
out.to_csv(opj(outpath, 'choice_prediction_results.csv'))


# SAME PAINFIRST TRIALS ONLY FOR SUPP MAT

###################################################################
# Load data
###################################################################
# Decision data
dec_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                            'decision_stats.csv'))

# Remove trials with high vif, unanswered trials (duration == 5)
dec_stats_clean = dec_stats[(dec_stats['vifs'] < 2)
                            & (dec_stats['duration'] < 5.0)
                            & (dec_stats['duration'] > 0.2)]

perc_removed = (len(dec_stats)- len(dec_stats_clean))/len(dec_stats)

# KEEP only painfirst
dec_stats_clean = dec_stats_clean[dec_stats_clean['painfirst'] == 1]

###################################################################
# Set up CV prediction
###################################################################

# Classifier
clf = Pipeline(steps=[('scaler', StandardScaler()),
                      ('svm', LinearSVC())])
# CV split
cv = GroupKFold(10)

# Data frame to collect results
out = pd.DataFrame(index=np.unique(dec_stats_clean['subject_id']))

# Make balanced accuracy scorer
bal_acc = make_scorer(balanced_accuracy_score)

###################################################################
# Predict choice using pain
###################################################################
# Get data
X = np.asarray(dec_stats_clean['pvp_cv_cosine']).reshape(-1, 1)
Y = np.asarray(dec_stats_clean['accept']).astype(int)

# Cross val prediction
pvp_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                              scoring=bal_acc, cv=cv)

# Binomial test
pvp_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(pvp_bal_acc)),
                         len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(pvp_bal_acc))] )


# Save data to
out['pvp_bal_acc'] = np.mean(pvp_bal_acc)
out['pvp_pval'] = pvp_pval

###################################################################
# Predict choice using money
###################################################################

X = np.asarray(dec_stats_clean['mvp_cv_cosine']).reshape(-1, 1)
Y = np.asarray(dec_stats_clean['accept']).astype(int)


mvp_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                          scoring=bal_acc, cv=cv)

# Binomial test
mvp_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(mvp_bal_acc)),
              len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(mvp_bal_acc))] )



out['mvp_bal_acc'] = np.mean(mvp_bal_acc)
out['mvp_pval'] = mvp_pval


###################################################################
# Predict choice using both
###################################################################

X = np.asarray([dec_stats_clean['pvp_cv_cosine'], dec_stats_clean['mvp_cv_cosine']]).T
Y = np.asarray(dec_stats_clean['accept']).astype(int)


bal_acc = make_scorer(balanced_accuracy_score)


both_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                          scoring=bal_acc, cv=cv)

# Binomial test
both_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(both_bal_acc)),
              len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(both_bal_acc))] )



out['both_bal_acc'] = np.mean(both_bal_acc)
out['both_pval'] = both_pval


# Refit on all to get SVM boundaries for plot
clf.fit( X, Y)
model_out = dict(clf=clf, x=X, y=Y, name='painmoney_painfirst')
# Save model for plot
np.save(opj(outpath, 'model_painmoney_painfirst.npy'), model_out)


# ###################################################################
# # Same but at the participant level
# ###################################################################

pvp_subs, mvp_subs, both_subs = [], [], [], []
for sub in np.unique(dec_stats_clean['subject_id']):

    sub_dat = dec_stats_clean[dec_stats_clean['subject_id'] == sub]
    Y = np.asarray(sub_dat['accept']).astype(int)
    if sub == 'sub-21':
        cv = StratifiedKFold(3) # Otherwise not enough trial
    else:
        cv = StratifiedKFold(5)

    # PVP
    X = np.asarray(sub_dat['pvp_cv_cosine']).reshape(-1, 1)
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'pvp_bal_acc_sub'] = sub_acc
    pvp_subs.append(sub_acc)

    # MVP
    X = np.asarray(sub_dat['mvp_cv_cosine']).reshape(-1, 1)
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'mvp_bal_acc_sub'] = sub_acc
    mvp_subs.append(sub_acc)

    # Both
    X = np.asarray([sub_dat['mvp_cv_cosine'], sub_dat['pvp_cv_cosine']]).T
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'both_bal_acc_sub'] = sub_acc
    both_subs.append(sub_acc)
# Save data
out['pvp_bal_acc_sub_mean'] = np.mean(pvp_subs)
out['mvp_bal_acc_sub_mean'] = np.mean(mvp_subs)
out['both_bal_acc_sub_mean'] = np.mean(both_subs)

# One sample t-tests against chance accuracy
out['pvp_bal_acc_sub_tval'], out['pvp_bal_acc_sub_pval'] = ttest_1samp(np.asarray(pvp_subs), 0.5)
out['mvp_bal_acc_sub_tval'], out['mvp_bal_acc_sub_pval'] = ttest_1samp(np.asarray(mvp_subs), 0.5)
out['both_bal_acc_sub_tval'], out['both_bal_acc_sub_pval'] = ttest_1samp(np.asarray(both_subs), 0.5)

out['both_vs_mvp_acc_sub_tval'], out['both_vs_mvp_acc_sub_pval'] = ttest_rel(np.asarray(both_subs), np.asarray(mvp_subs))
out['pvp_vs_mvp_acc_sub_tval'], out['pvp_vs_mvp_acc_sub_pval'] = ttest_rel(np.asarray(pvp_subs), np.asarray(mvp_subs))


# Save
out = out.reset_index()
out['subject_id'] = out['index']
out.to_csv(opj(outpath, 'choice_prediction_results_painfirst.csv'))


# SAME MONEY FIRST TRIALS ONLY


###################################################################
# Load data
###################################################################
# Decision data
dec_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                            'decision_stats.csv'))

# Remove trials with high vif, unanswered trials (duration == 5)
dec_stats_clean = dec_stats[(dec_stats['vifs'] < 2)
                            & (dec_stats['duration'] < 5.0)
                            & (dec_stats['duration'] > 0.2)]

perc_removed = (len(dec_stats)- len(dec_stats_clean))/len(dec_stats)

dec_stats_clean = dec_stats_clean[dec_stats_clean['painfirst'] == 0]
###################################################################
# Set up CV prediction
###################################################################

# Classifier
clf = Pipeline(steps=[('scaler', StandardScaler()),
                      ('svm', LinearSVC())])
# CV split
cv = GroupKFold(10)

# Data frame to collect results
out = pd.DataFrame(index=np.unique(dec_stats_clean['subject_id']))

# Make balanced accuracy scorer
bal_acc = make_scorer(balanced_accuracy_score)

###################################################################
# Predict choice using pain
###################################################################
# Get data
X = np.asarray(dec_stats_clean['pvp_cv_cosine']).reshape(-1, 1)
Y = np.asarray(dec_stats_clean['accept']).astype(int)

# Cross val prediction
pvp_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                              scoring=bal_acc, cv=cv)

# Binomial test
pvp_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(pvp_bal_acc)),
                         len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(pvp_bal_acc))] )


# Save data to
out['pvp_bal_acc'] = np.mean(pvp_bal_acc)
out['pvp_pval'] = pvp_pval

###################################################################
# Predict choice using money
###################################################################

X = np.asarray(dec_stats_clean['mvp_cv_cosine']).reshape(-1, 1)
Y = np.asarray(dec_stats_clean['accept']).astype(int)


mvp_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                          scoring=bal_acc, cv=cv)

# Binomial test
mvp_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(mvp_bal_acc)),
              len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(mvp_bal_acc))] )


out['mvp_bal_acc'] = np.mean(mvp_bal_acc)
out['mvp_pval'] = mvp_pval



###################################################################
# Predict choice using both
###################################################################

X = np.asarray([dec_stats_clean['pvp_cv_cosine'], dec_stats_clean['mvp_cv_cosine']]).T
Y = np.asarray(dec_stats_clean['accept']).astype(int)


bal_acc = make_scorer(balanced_accuracy_score)


both_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                          scoring=bal_acc, cv=cv)

# Binomial test
both_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(both_bal_acc)),
              len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(both_bal_acc))] )


out['both_bal_acc'] = np.mean(both_bal_acc)
out['both_pval'] = both_pval


# Refit on all to get SVM boundaries for plot
clf.fit( X, Y)
model_out = dict(clf=clf, x=X, y=Y, name='painmoney_moneyfirst')
# Save model for plot
np.save(opj(outpath, 'model_painmoney_moneyfirst.npy'), model_out)


# ###################################################################
# # Same but at the participant level
# ###################################################################

pvp_subs, mvp_subs, both_subs = [], [], []
for sub in np.unique(dec_stats_clean['subject_id']):

    sub_dat = dec_stats_clean[dec_stats_clean['subject_id'] == sub]
    Y = np.asarray(sub_dat['accept']).astype(int)
    cv = StratifiedKFold(5)

    # PVP
    X = np.asarray(sub_dat['pvp_cv_cosine']).reshape(-1, 1)
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'pvp_bal_acc_sub'] = sub_acc
    pvp_subs.append(sub_acc)

    # MVP
    X = np.asarray(sub_dat['mvp_cv_cosine']).reshape(-1, 1)
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'mvp_bal_acc_sub'] = sub_acc
    mvp_subs.append(sub_acc)

    # Both
    X = np.asarray([sub_dat['mvp_cv_cosine'], sub_dat['pvp_cv_cosine']]).T
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'both_bal_acc_sub'] = sub_acc
    both_subs.append(sub_acc)



# Save data
out['pvp_bal_acc_sub_mean'] = np.mean(pvp_subs)
out['mvp_bal_acc_sub_mean'] = np.mean(mvp_subs)
out['both_bal_acc_sub_mean'] = np.mean(both_subs)

# One sample t-tests against chance accuracy
out['pvp_bal_acc_sub_tval'], out['pvp_bal_acc_sub_pval'] = ttest_1samp(np.asarray(pvp_subs), 0.5)
out['mvp_bal_acc_sub_tval'], out['mvp_bal_acc_sub_pval'] = ttest_1samp(np.asarray(mvp_subs), 0.5)
out['both_bal_acc_sub_tval'], out['both_bal_acc_sub_pval'] = ttest_1samp(np.asarray(both_subs), 0.5)

out['both_vs_mvp_acc_sub_tval'], out['both_vs_mvp_acc_sub_pval'] = ttest_rel(np.asarray(both_subs), np.asarray(mvp_subs))
out['pvp_vs_mvp_acc_sub_tval'], out['pvp_vs_mvp_acc_sub_pval'] = ttest_rel(np.asarray(pvp_subs), np.asarray(mvp_subs))


# Save
out = out.reset_index()
out['subject_id'] = out['index']
out.to_csv(opj(outpath, 'choice_prediction_results_moneyfirst.csv'))



# Predict which is first

###################################################################
# Load data
###################################################################
# Decision data
dec_stats = pd.read_csv(opj(basepath, 'derivatives/mvpa/decision',
                            'decision_stats.csv'))

# Remove trials with high vif, unanswered trials (duration == 5)
dec_stats_clean = dec_stats[(dec_stats['vifs'] < 2)
                            & (dec_stats['duration'] < 5.0)
                            & (dec_stats['duration'] > 0.2)]

###################################################################
# Set up CV prediction
###################################################################

# Classifier
clf = Pipeline(steps=[('scaler', StandardScaler()),
                      ('svm', LinearSVC())])
# CV split
cv = GroupKFold(10)

# Data frame to collect results
out = pd.DataFrame(index=np.unique(dec_stats_clean['subject_id']))

# Make balanced accuracy scorer
bal_acc = make_scorer(balanced_accuracy_score)

###################################################################
# Predict choice using pain
###################################################################
# Get data
X = np.asarray(dec_stats_clean['pvp_cv_cosine']).reshape(-1, 1)
Y = np.asarray(dec_stats_clean['painfirst']).astype(int)

# Cross val prediction
pvp_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                              scoring=bal_acc, cv=cv)

# Binomial test
pvp_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(pvp_bal_acc)),
                         len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(pvp_bal_acc))] )


# Save data to
out['pvp_bal_acc'] = np.mean(pvp_bal_acc)
out['pvp_pval'] = pvp_pval

###################################################################
# Predict choice using money
###################################################################

X = np.asarray(dec_stats_clean['mvp_cv_cosine']).reshape(-1, 1)
Y = np.asarray(dec_stats_clean['painfirst']).astype(int)


mvp_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                          scoring=bal_acc, cv=cv)

# Binomial test
mvp_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(mvp_bal_acc)),
              len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(mvp_bal_acc))] )


out['mvp_bal_acc'] = np.mean(mvp_bal_acc)
out['mvp_pval'] = mvp_pval


###################################################################
# Predict choice using both
###################################################################

X = np.asarray([dec_stats_clean['pvp_cv_cosine'], dec_stats_clean['mvp_cv_cosine']]).T
Y = np.asarray(dec_stats_clean['painfirst']).astype(int)


bal_acc = make_scorer(balanced_accuracy_score)


both_bal_acc = cross_val_score(clf, X, Y, groups=dec_stats_clean['subject_id'],
                          scoring=bal_acc, cv=cv)

# Binomial test
both_pval = binom_test(x=[np.round(len(dec_stats_clean)*np.mean(both_bal_acc)),
              len(dec_stats_clean)-np.round(len(dec_stats_clean)*np.mean(both_bal_acc))] )


out['both_bal_acc'] = np.mean(both_bal_acc)
out['both_pval'] = both_pval


# Refit on all to get SVM boundaries for plot
clf.fit( X, Y)
model_out = dict(clf=clf, x=X, y=Y, name='painmoney_whichfirst')
# Save model for plot
np.save(opj(outpath, 'model_painmoney_whichfirst.npy'), model_out)


# ###################################################################
# # Same but at the participant level
# ###################################################################

pvp_subs, mvp_subs, both_subs = [], [], []
for sub in np.unique(dec_stats_clean['subject_id']):

    sub_dat = dec_stats_clean[dec_stats_clean['subject_id'] == sub]
    Y = np.asarray(sub_dat['painfirst']).astype(int)
    cv = StratifiedKFold(5)

    # PVP
    X = np.asarray(sub_dat['pvp_cv_cosine']).reshape(-1, 1)
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'pvp_bal_acc_sub'] = sub_acc
    pvp_subs.append(sub_acc)

    # MVP
    X = np.asarray(sub_dat['mvp_cv_cosine']).reshape(-1, 1)
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'mvp_bal_acc_sub'] = sub_acc
    mvp_subs.append(sub_acc)

    # Both
    X = np.asarray([sub_dat['mvp_cv_cosine'], sub_dat['pvp_cv_cosine']]).T
    sub_acc = np.mean(cross_val_score(clf, X, Y,
                          scoring=bal_acc, cv=cv))

    out.loc[sub, 'both_bal_acc_sub'] = sub_acc
    both_subs.append(sub_acc)


# Save data
out['pvp_bal_acc_sub_mean'] = np.mean(pvp_subs)
out['mvp_bal_acc_sub_mean'] = np.mean(mvp_subs)
out['both_bal_acc_sub_mean'] = np.mean(both_subs)

# One sample t-tests against chance accuracy
out['pvp_bal_acc_sub_tval'], out['pvp_bal_acc_sub_pval'] = ttest_1samp(np.asarray(pvp_subs), 0.5)
out['mvp_bal_acc_sub_tval'], out['mvp_bal_acc_sub_pval'] = ttest_1samp(np.asarray(mvp_subs), 0.5)
out['both_bal_acc_sub_tval'], out['both_bal_acc_sub_pval'] = ttest_1samp(np.asarray(both_subs), 0.5)

out['both_vs_mvp_acc_sub_tval'], out['both_vs_mvp_acc_sub_pval'] = ttest_rel(np.asarray(both_subs), np.asarray(mvp_subs))
out['pvp_vs_mvp_acc_sub_tval'], out['pvp_vs_mvp_acc_sub_pval'] = ttest_rel(np.asarray(pvp_subs), np.asarray(mvp_subs))


# Save
out = out.reset_index()
out['subject_id'] = out['index']
out.to_csv(opj(outpath, 'choice_prediction_results_whichfirst.csv'))

