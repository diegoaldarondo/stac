#!/bin/bash
# DATAFOLDER=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/snippets/JDM25_v5/reformatted/*.mat
# DATAPATHS=($(find $DATAFOLDER))
# FILES=($(find $DATAFOLDER -printf "%f\n"))
#
# VERBOSE=True
# OFFSETPATH=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/offsets/JDM25_m_9_NoHip.p
# VISUALIZE=False
# SAVEPATH=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/results/JDM25_v14
# PARAMPATH=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/june3/JDM25.yaml
#
# cnt=${#FILES[@]}
# cnt=$(($cnt - 1))
# sbatch --array=0-$cnt --partition=shared,general,olveczky cluster/submit_compute_stac.sh $PARAMPATH $SAVEPATH $OFFSETPATH ${DATAPATHS[*]}


DATAFOLDER=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/snippets/JDM33/reformatted/*.mat
DATAPATHS=($(find $DATAFOLDER))
FILES=($(find $DATAFOLDER -printf "%f\n"))

VERBOSE=True
OFFSETPATH=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/offsets/JDM33_june14.p
VISUALIZE=False
SAVEPATH=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/results/JDM33_v5
PARAMPATH=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/june3/JDM33.yaml

cnt=${#FILES[@]}
cnt=$(($cnt - 1))
sbatch --array=0-$cnt --partition=shared,general,olveczky cluster/submit_compute_stac.sh $PARAMPATH $SAVEPATH $OFFSETPATH ${DATAPATHS[*]}


DATAFOLDER=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/snippets/JDM31/reformatted/*.mat
DATAPATHS=($(find $DATAFOLDER))
FILES=($(find $DATAFOLDER -printf "%f\n"))

VERBOSE=True
OFFSETPATH=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/offsets/JDM31_june14.p
VISUALIZE=False
SAVEPATH=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/results/JDM31_v2
PARAMPATH=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/june3/JDM31.yaml

cnt=${#FILES[@]}
cnt=$(($cnt - 1))
sbatch --array=0-$cnt --partition=shared,general,olveczky cluster/submit_compute_stac.sh $PARAMPATH $SAVEPATH $OFFSETPATH ${DATAPATHS[*]}
