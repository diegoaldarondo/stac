#!/bin/bash
DATAFOLDER="/n/home02/daldarondo/LabDir/Diego/data/dm/stac/snippets/JDM33/reformatted/*.mat"
DATAPATHS=($(find $DATAFOLDER))
FILES=($(find $DATAFOLDER -printf "%f\n"))

VERBOSE="True"
OFFSETPATH="/n/home02/daldarondo/LabDir/Diego/data/dm/stac/offsets/JDM33_june12.p"
VISUALIZE="False"
NSNIPPETS=380
SAVEPATH="/n/home02/daldarondo/LabDir/Diego/data/dm/stac/results/JDM33_v3"
PARAMPATH="/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/june3/JDM33.yaml"

cnt=${#FILES[@]}
cnt=$(($cnt - 1))
sbatch --array=0-$cnt cluster/submit_compute_stac.sh $PARAMPATH $SAVEPATH $OFFSETPATH ${DATAPATHS[*]}
