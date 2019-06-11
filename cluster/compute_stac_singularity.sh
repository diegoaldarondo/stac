#!/bin/bash
DATAFOLDER="/n/home02/daldarondo/LabDir/Diego/data/dm/stac/snippets/JDM25_v5/reformatted/*.mat"
DATAPATHS=($(find $DATAFOLDER))
FILES=($(find $DATAFOLDER -printf "%f\n"))

VERBOSE="True"
OFFSETPATH="/n/home02/daldarondo/LabDir/Diego/data/dm/stac/offsets/JDM25_m_9_NoHip.p"
VISUALIZE="False"
NSNIPPETS=380
SAVEPATH="/n/home02/daldarondo/LabDir/Diego/data/dm/stac/results/JDM25_v11"
PARAMPATH="/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/june3/JDM25.yaml"

cnt=${#FILES[@]}
for i in $(seq 0 $[$cnt-1]);
  do sbatch submit_compute_stac.sh ${DATAPATHS[$i]} $PARAMPATH $SAVEPATH $OFFSETPATH
done
