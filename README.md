
# Code for Coll et al., The neural signature of the decision value of future pain

You can execute the line below to reproduce all analyses/figures in the manuscript. The BIDS formatted data root folder is the $DATAPATH folder and this repository is the $CODEPATH folder.

The system parameters (e.g. number of cpus) should be changed in each script. Bootstrapping/permuting can take a long time.

The emotion dataset (need to download data at http://neurovault.org/collections/503) and the pain risk dataset (need to download data at https://openneuro.org/datasets/ds001814) should be in the $DATAPATH/external/pines and the $DATAPATH/external/ds_001814 respectively.

The NPS signature can be obtained on request from Tor Wager. The SIIPS signature and the Canlab parcellation are available at https://github.com/canlab/Neuroimaging_Pattern_Masks.

The striatum mask is availalbe at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases.

The templates used for plotting can be obtained from https://www.templateflow.org/.


# Running the analyses

## Preprocessing with fmriprep
You need to sort out paths and add freesurfer license.txt
```
docker run -it --rm -v $DATAPATH:/data -v $DATAPATH/derivatives:/out  poldracklab/fmriprep:latest /data /out participant --output-spaces MNI152NLin2009cAsym:res-2 --fs-license-file /out/license.txt --fs-no-reconall
```

## Pull the container
```
docker pull mpcoll2/coll_painvalue_2021:latest
```

## Behavioural analyses
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python /code/behavioural/behavioural.py
```
Note that there are no R kernels in the container so the R script should be executed locally.
```
Rscript $CODEPATH/behavioural/behavioural_linearmodels.R
```

## Create the group mask
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python /code/preprocessing/make_group_mask.py
```


## GLMs first level
Pain and money first part offer
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python /code/glms/model_offer1.py
```
Shocks
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python /code/glms/model_shocks.py 
```
Anticipation
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python /code/glms/model_anticipation.py
```
Single trials decisions
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python /code/glms/model_offer2_strials.py
```

## MVPA + univariate analyses
Offer 1 pain level
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/mvpa/pain_level.py
```

Offer 1 money level
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/mvpa/money_level.py 
```
Shock level
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/mvpa/shock_level.py
```
Collect data for single trial deicsions
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/mvpa/decision.py
```
Anticipation level
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/mvpa/anticipation_level.py
```

## Apply patterns to data with cross validation
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/mvpa/cv_pattern_expressions.py
```

## Predict participants' choices
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/mvpa/predict_choice.py
```

## Run whole brain cross-prediction searchlight
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/mvpa/sl_crosspainmoney.py
```

## Run whole brain cross-prediction searchlight in striatum ROI
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/mvpa/sl_crosspainmoney_striatumonly.py
```

## Run analyses on emotion dataset
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/external_datasets/pines/rating_level.py
```

## Run analyses on pain risk dataset
Preprocessing
```
docker run -it --rm -v $DS001814PATH:/data -v $DS001814PATH/derivatives:/out  poldracklab/fmriprep:latest /data /out participant --output-spaces MNI152NLin2009cAsym:res-2 --fs-license-file /out/license.txt --fs-no-reconall
```
GLM
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python /code/external/ds_001814/glms/model_risk_strials.py
```
Apply pattern and figure for risk dataset
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python /code/external/ds_001814/mvpa/risk_level_strials.py
```

## Make all plots (panels assembled manually for manuscript)

```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/figures/figures_pvp.py
```
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/figures/figures_mvp.py
```
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/figures/figures_sip.py
```
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/figures/figures_crosspred.py
```
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/figures/figures_crosspred.py
```
```
docker run -it --rm -v $DATAPATH:/data -v $CODEPATH:/code mpcoll2/coll_painvalue_2021:latest python  /code/figures/figures_posneg_weights.py
```