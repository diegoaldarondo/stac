#!/bin/bash
DATAFOLDER="/home/diego/data/dm/stac/results/JDM25_v9/*.p"
SAVEFOLDER="/home/diego/data/dm/stac/clips/JDM25_v9/"
PARAMPATH="/home/diego/code/olveczky/dm/stac/params/june3/JDM25.yaml"

DATAPATHS=($(find $DATAFOLDER))
FILES=($(find $DATAFOLDER -printf "%f\n"))

SAVEPATHS=()
cnt=${#FILES[@]}
for ((i=0;i<cnt;i++)); do
  SAVEPATHS[i]="$SAVEFOLDER${FILES[i]%.p}.mp4"
done

cnt=${#FILES[@]}
for i in $(seq 0 $[$cnt-1]);
  do echo ${DATAPATHS[$i]} $PARAMPATH ${SAVEPATHS[$i]};
done | xargs -l --max-procs=7 bash -c 'python view_stac.py $0 $1 --render-video="True" --save-path=$2 --headless="True"'
