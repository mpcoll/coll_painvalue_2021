docker run -it --rm -v $DATAPATH/source/external/ds_001814:/data -v $DATAPATH/source/external/ds_001814/derivatives:/out  poldracklab/fmriprep:latest /data /out participant --output-spaces MNI152NLin2009cAsym:res-2 --fs-license-file /out/license.txt --fs-no-reconall
