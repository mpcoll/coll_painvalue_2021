# Computational models at the participant level
# -*- coding: utf-8 -*-
import copy
import os.path
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import itertools
import os
from os.path import join as opj
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.io import loadmat
import ptitprince as pt
from scipy.stats import pearsonr

###################################################################
# OS specific parameters
###################################################################

basepath = '/data'

###################################################################
# Fixed parameters
###################################################################

# Paths
sourcedir = opj(basepath, 'source')
outdir = opj(basepath, 'derivatives/behavioural')
outfigdir = opj(basepath, 'derivatives/figures/behavioural')
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists(outfigdir):
    os.mkdir(outfigdir)

# Get participants
r = re.compile('.*sub-*')
subs = list(filter(r.match, os.listdir(sourcedir)))

###################################################################
# Plot parameters
###################################################################
# colors
current_palette = sns.color_palette('colorblind', 6)
colp = current_palette[0]
colm = current_palette[2]

cold = current_palette[1]
colc = current_palette[3]

# Label size
labelfontsize = 9
titlefontsize = np.round(labelfontsize*1)
ticksfontsize = np.round(labelfontsize*0.8)
legendfontsize = np.round(labelfontsize*0.8)
colorbarfontsize = legendfontsize

# Font
plt.rcParams['font.family'] = 'Helvetica'

# Despine
plt.rc("axes.spines", top=False, right=False)

####################################################################
# Loop sub and run and concatenante all csv files
data = pd.DataFrame()


for s in subs:
    for run in range(5):
        subdat = pd.read_csv(opj(sourcedir, s,
                                 'func', s + '_task-npng_run-'
                                 + str(run+1) + '_events.tsv'),
                             sep='\t')
        subdat['sub'] = s
        subdat['trials'] = np.arange(1, 21) + 20*run
        data = pd.concat([data, subdat])

# Put in a smaller dataframe to get rid of unused columns
df = pd.DataFrame()
df['sub'] = data['sub']
df['rt'] = data['offer2_duration']*1000
df['accept'] = data['accept']
df['pain_level'] = data['pain_level']
df['pain_rank'] = data['pain_rank']
df['pain_intensity'] = data['intensity_level']
df['money_level'] = data['money_level']
df['money_rank'] = round((data['money_level']+1.11)/1.11)
df['run'] = data['run']
df['trials'] = data['trials']
df = df.reset_index()

# Set up the data as a pandas dataframe
df["TrialChoice"] = df["accept"]
df["money_level"] = df['money_rank'].copy()
df["pain_level"] = df['pain_rank'].copy()
df["SubjectID"] = df['sub'].copy()

# Remove unanswered trials
df = df[df['rt'] < 5000]
df = df[df['rt'] > 200]

## Define functions
def get_AIC(LL, num_params):
    """Returns AIC score (float) for model based on log-likelihood (LL: float)
    and number of parameters (num_params: int)."""
    return -2*LL + 2*num_params

def fit_SV_function(func_type, k, data, levels_var):
    """Fits mathematical function to levels of the good on offer.

    Here the supplied mathematical function `func_type` is fit to each row
        (i.e., trial) in `data` to transform levels of the good on offer
        (e.g., N-back levels). A subject-specific `k` parameter is fit to
        adjust the scaling of the function for each subject.

    Args:
        func_type (string): Name of function to fit to stimulus levels. Options
            are "linear", "hyper", "para", "expo", or "none" for linear,
            hyperbolic, parabolic, exponential, or unchanged functions,
            respectively.
        k (float): Subject-specific scaling parameter for function.
        data (pandas.DataFrame): Dataframe containing `levels_var` as a column.
        levels_var (string): Name of column in `data` that contains the stimulus
            levels of interest (e.g., N-back levels).

    Returns:
        (pandas.Series): The `levels_var` values transformed by `func_type`
            function.
    """
    return {
        #Scale down by 0.1 to prevent overflow error
        "linear" : k * data[levels_var] * 0.1,
        "hyper"  : 1 / (1 - k * data[levels_var] * 0.1),
        "para"   : k * (data[levels_var]**2) * 0.1,
        "expo"   : np.exp(k * data[levels_var] * 0.1),
        "none"   : data[levels_var]
    }[func_type] 

def get_model_LL(params, data, money_func, pain_func):
    """Returns log-likelihood of model after fitting with mathematical function(s).

    This function returns the log-likelihood of a model that is fit using a
        specified mathematical function (e.g., parabolic) with the parameters
        supplied by the `scipy.optimize.fmin` function.

    Args:
        params (numpy.array): The free parameters to be estimated. A "choice bias"
            and "beta" parameter influence the softmax decision function. A "k"
            scaling parameter is used to adjust the mathematical functions fit
            to the good(s) levels.
        data (pandas.DataFrame): Dataframe passed to `fit_SV_function` containing
            `levels_var` as a column.
        money_func (string): Name of function to fit to money levels. Options
            are "linear", "hyper", "para", "expo", or "none" for linear, hyperbolic,
            parabolic, exponential, or unchanged functions, respectively.
        pain_func (string): Name of function to fit to pain levels. Options
            are "linear", "hyper", "para", "expo", or "none" for linear, hyperbolic,
            parabolic, exponential, or unchanged functions, respectively.

    Returns:
        (float): The sum of the log likelihoods of the model fit to each row
            (i.e., trial) using the supplied mathematical functions. The sign
            is flipped to return a positive number for minimization using `fmin`.
    """
    if len(params) < 4:
        if money_func != "none":
            [choice_bias, beta_param, k_money] = params
            k_pain = 1
        elif pain_func != "none":
            [choice_bias, beta_param, k_pain] = params
            k_money = 1
        else:
            [choice_bias, beta_param] = params
            k_money = 1
            k_pain = 1
    else:
        [choice_bias, beta_param, k_money, k_pain] = params

    # Restrict k scaling parameters to be positive
    if((k_money < 0) or (k_pain < 0)):
        return 10000
    LL = 0

    # Fit the mathematical functions
    sv_money = fit_SV_function(money_func, k_money, data, "money_level")
    sv_pain = fit_SV_function(pain_func, k_pain, data, "pain_level")
    sv_both = sv_pain - sv_money

    # Input data to softmax decision function and return log-likelihood
    p_pain = 1 / (1 + np.exp(beta_param * (choice_bias + sv_both)))
    data = data.assign(p_pain=p_pain)
    LL = np.where(data["TrialChoice"] == 1, np.log(p_pain), np.log(1-p_pain))
    return -1*sum(LL) #return log-likelihood (for minimizing the LL value) (LL is negative, so flip the sign)


def fit_all_parts(money_models, pain_models, df, mean_params=None):

    # Initialize dictionaries to write to
    LL_dict = {}
    AIC_sum_dict = {}
    fit_models = {}
    fit_models_summaries = {}
    summary_df = pd.DataFrame(index=df["SubjectID"].unique())


    # Loop through all possible combinations of models
    for model_type_combo in itertools.product(money_models, pain_models):
        # Set the correct function for money and pain
        money_func = model_type_combo[0] #e.g., "linear"
        pain_func = model_type_combo[1]   #e.g., "expo"
        print(model_type_combo)

        # Reset log-likelihood (LL), AIC, and summary values for each combination of models
        total_LL = 0
        total_AIC = 0
        AIC_list = []

        # Loop through each subject to fit model functions
        for subject in df["SubjectID"].unique():
            subj_data = df.loc[df["SubjectID"] == subject].reset_index()


            # Set all parameters to 1 for initialization
            choice_bias_start = 1
            beta_param_start = 1
            k_money_start = 1
            k_pain_start = 1

            # Choice bias and starting point will always be included in arguments
            arg_array = [choice_bias_start, beta_param_start]

            # Add in the k parameters for money and pain functions if necessary
            bounds = [(None, None), (-10, 10)]
            if money_func != "none":
                arg_array.append(k_money_start)
                bounds = [(None, None), (-10, 10), (None, None)]

            if pain_func != "none":
                arg_array.append(k_pain_start)
                bounds = [(None, None), (-10, 10), (None, None)]

            if pain_func != "none" and money_func != "none":
                bounds = [(None, None), (-10, 10), (-100, 100), (-100, 100)]


            # Minimize log-likelihood and estimate parameters
            res = minimize(fun=get_model_LL, x0=np.array(arg_array),
                           args=(subj_data, money_func, pain_func),
                           bounds=bounds, method='SLSQP',
                           options={'maxiter': 1000000})

            # Make sure convergence is reched
            if not res['success']:
                print(subject)
                print(model_type_combo)
                print(res)
                raise ValueError

            # Grab estimated parameters
            print(res)
            res_params = res['x']

            if money_func == "none":
                res_params = np.insert(res_params, 2, 1)
            if pain_func == "none":
                res_params = np.insert(res_params, 3, 1)

            print("\t", subj_data.SubjectID.unique())
            print("\t", res['fun']) #Prints subject LL

            total_LL += res['fun']
            total_AIC += get_AIC(-1*res['fun'], len(res['x']))

            # Make a summary dataframe with information of model fit
            mod_name = str(model_type_combo[1])
            summary_df.loc[subject, mod_name + '_choice_bias'] = res_params[0]
            summary_df.loc[subject, mod_name + '_beta'] = res_params[1]
            summary_df.loc[subject, mod_name + '_k_money'] = res_params[2]
            summary_df.loc[subject, mod_name + '_k_pain'] = res_params[3]
            summary_df.loc[subject, mod_name + '_LL'] = -1*res['fun']
            summary_df.loc[subject, mod_name + '_AIC'] = get_AIC(-1*res['fun'], len(res['x']))

            # Calculate SVs for subject based on fmin fit values
            subj_sv_money = fit_SV_function(money_func, res_params[2], subj_data, "money_level")
            subj_sv_pain = fit_SV_function(pain_func, res_params[3], subj_data, "pain_level")
            subj_sv_both = subj_sv_pain - subj_sv_money
            subj_p_pain = 1 / (1 + np.exp(res_params[1] * (res_params[0] + subj_sv_both)))

            # Add the new SV fits to the original dataframe
            df.loc[df["SubjectID"] == subject, "sv_money"] = np.asarray(subj_sv_money)
            df.loc[df["SubjectID"] == subject, "sv_pain"] = np.asarray(subj_sv_pain)
            df.loc[df["SubjectID"] == subject, "sv_both"] = np.asarray(subj_sv_both)
            df.loc[df["SubjectID"] == subject, "p_pain_all"] = np.asarray(subj_p_pain)
            df.loc[df["SubjectID"] == subject, "choice_bias"] = res_params[0]
            df.loc[df["SubjectID"] == subject, "beta"] = res_params[1]
            df.loc[df["SubjectID"] == subject, "k_pain"] = res_params[3]
            df.loc[df["SubjectID"] == subject, "k_money"] = res_params[2]

        # Add information and data to dictionaries containing all model combinations
        LL_dict[model_type_combo] = total_LL
        AIC_sum_dict[model_type_combo] = total_AIC
        fit_models[model_type_combo] = copy.deepcopy(df)
        fit_models_summaries[model_type_combo] = summary_df


    return LL_dict, AIC_sum_dict, summary_df, df


# Compare models of pain value scaling
money_models = ["none"]
pain_models = ["none", "linear", "expo", "para"]

LL_dict, AIC_dict, summary_df, df  = fit_all_parts(money_models, pain_models,
                                         df)


# # Save to perform model comparison in Matlab
summary_df.to_csv(opj(outdir, 'matlab_model_comparison.csv'))


# Refit with winning model
money_models = ["none"]
pain_models = ['expo']

_, _, summary_df, df  = fit_all_parts(money_models, pain_models, df)

df.to_csv(opj(outdir, 'behavioural_with_compestimates.csv'))


if not os.path.exists(opj(basepath, 'derivatives/behavioural',
                      'VBA_model_comp.mat')):
    raise FileNotFoundError('Please run vba_model_comparison.m and rerun this code')

##############################################################################
# Supplementary figures
###############################################################################

labelfontsize = 7
titlefontsize = np.round(labelfontsize*1.5)
ticksfontsize = np.round(labelfontsize*0.8)
legendfontsize = np.round(labelfontsize*0.8)

vbacomp = loadmat(opj(basepath, 'derivatives/behavioural',
                      'VBA_model_comp.mat'))

modnames = ['None', 'Linear', 'Expo', 'Para']


ep = list(vbacomp['out']['pep'][0][0][0])
ef = [float(ef)*100 for ef in vbacomp['out']['Ef'][0][0]]

ep = np.asarray(ep)
ef = np.asarray(ef)
fig, host = plt.subplots(figsize=(2.5, 2))

par1 = host.twinx()
color1 = '#7293cb'
color2 = '#e1974c'

x = np.arange(0.5, (len(ep))*0.75, 0.75)
x2 = [c + 0.25 for c in x]
p1 = host.bar(x, ep, width=0.25, color=color1, linewidth=1, edgecolor='k')
p2 = par1.bar(x2, ef, width=0.25, color=color2, linewidth=1, edgecolor='k')

host.set_ylim(0, 1)
par1.set_ylim(0, 100)


# host.set_xlabel("Distance")
host.set_ylabel("Protected exceedance probability",
                fontsize=labelfontsize)
par1.set_ylabel("Model Frequency (%)",  fontsize=labelfontsize)


for ax in [par1]:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    plt.setp(ax.spines.values(), visible=False)
    ax.spines["right"].set_visible(True)

host.yaxis.label.set_color(color1)
par1.yaxis.label.set_color(color2)

host.spines["left"].set_edgecolor(color1)
par1.spines["right"].set_edgecolor(color2)

host.set_xticks([i+0.125 for i in x])
host.set_xticklabels(modnames, size=labelfontsize, rotation=30)

host.tick_params(axis='x', labelsize=ticksfontsize)

host.tick_params(axis='y', colors=color1, labelsize=ticksfontsize)
par1.tick_params(axis='y', colors=color2, labelsize=ticksfontsize)
fig.tight_layout()
fig.savefig(opj(outfigdir, 'model_comparison.svg'), dpi=600)



fig, ax = plt.subplots(1, 3, figsize=(3, 2))
pal = sns.color_palette("deep", 5)
labels = [r'$Bias$', r'$\beta$', r'$k$']
for idx, var in enumerate(['expo_choice_bias', 'expo_beta', 'expo_k_pain']):
    dplot = summary_df[[var]].melt()
    pt.half_violinplot(x='variable', y="value", data=dplot, inner=None,
                       jitter=True, color=pal[idx], lwidth=0, width=0.6,
                       offset=0.17, cut=1, ax=ax[idx],
                       linewidth=1, alpha=0.6, zorder=19)
    sns.stripplot(x='variable', y="value", data=dplot,
                  jitter=0.08, ax=ax[idx],
                  linewidth=1, alpha=0.6, color=pal[idx], zorder=1)
    sns.boxplot(x='variable', y="value", data=dplot,
                color=pal[idx], whis=np.inf, linewidth=1, ax=ax[idx],
                width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                whiskerprops={'zorder': 10, 'alpha': 1},
                medianprops={'zorder': 11, 'alpha': 0.5})

    # if var == 'expo_beta':
    #     ax[idx].set_ylim(0, 5)
    ax[idx].set_xticklabels([labels[idx]], fontsize=labelfontsize)
    if idx == 0:
        ax[idx].set_ylabel('Value', fontsize=labelfontsize)
    else:
        ax[idx].set_ylabel('')
    ax[idx].set_xlabel('')
    ax[idx].tick_params('y', labelsize=ticksfontsize)
    ax[idx].tick_params('x', labelsize=ticksfontsize)
fig.tight_layout()

fig.savefig(opj(outfigdir, 'model_parameters.svg'), dpi=600)


from scipy.stats import pearsonr, zscore

corr = []
df['sv_pain_z'] = 99999.0
for sub in df['sub'].unique():
    df['sv_pain_z'][df['sub'] == sub] = zscore(df['sv_pain'][df['sub'] == sub])


fig, ax = plt.subplots(1,1, figsize=(1.5, 1.5))
df['pain_rank'] = df['pain_rank'].astype(int)
sns.pointplot('pain_rank', 'sv_pain_z', hue='sub', data=df,
              scale=0.1, ax=ax, ci=None, legend=None,
              palette='viridis')
ax.set_ylabel('Pain subjective value\n(z scored)', fontsize=labelfontsize)
ax.set_xlabel('Pain level', fontsize=labelfontsize)
ax.tick_params('y', labelsize=ticksfontsize)
ax.tick_params('x', labelsize=ticksfontsize-1)
ax.get_legend().remove()
fig.tight_layout()
fig.savefig(opj(outfigdir, 'subjective_value.svg'), dpi=600)



