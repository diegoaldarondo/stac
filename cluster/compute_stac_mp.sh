#!/bin/bash
DATAFOLDER="/home/diego/data/dm/stac/snippets/JDM25_v5/reformatted/*.mat"
DATAPATHS=($(find $DATAFOLDER))
FILES=($(find $DATAFOLDER -printf "%f\n"))

VERBOSE="True"
OFFSETPATH="/home/diego/data/dm/stac/offsets/JDM25_m_9_NoHip.p"
VISUALIZE="False"
NSNIPPETS=380
SAVEPATH="/home/diego/data/dm/stac/results/JDM25_v11"
PARAMPATH="/home/diego/code/olveczky/dm/stac/params/june3/JDM25.yaml"
echo ${DATAPATHS[0]} $PARAMPATH $SAVEPATH $OFFSETPATH

cnt=${#FILES[@]}
for i in $(seq 0 $[$cnt-1]);
  do echo ${DATAPATHS[$i]} $PARAMPATH $SAVEPATH $OFFSETPATH;
done | xargs -l --max-procs=1 bash -c 'python compute_stac.py $0 $1 --verbose="True" --save-path=$2 --offset-path=$3'

# cnt=${#FILES[@]}
# for i in $(seq 0 $[$cnt-1]);
#   do echo ${DATAPATHS[$i]} $PARAMPATH $SAVEPATH $OFFSETPATH;
# done | xargs -l --max-procs=1 bash -c 'echo $0 $1 --verbose="True" --save-path=$2 --offset-path=$3'
