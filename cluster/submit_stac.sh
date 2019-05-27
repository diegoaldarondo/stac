#!/bin/bash
DATAPATH="/home/diego/data/dm/stac/snippets/JDM25_v5/reformatted"
VERBOSE="True"
OFFSETPATH="/home/diego/data/dm/stac/offsets/JDM25_offset.p"
VISUALIZE="False"
NSNIPPETS=380
SAVEPATH="/home/diego/data/dm/stac/results/JDM25_v5"
PARAMPATH="/home/diego/code/olveczky/dm/stac/params/JDM25.yaml"
sbatch --array=1-$NSNIPPETS --wait cluster/stac.sh $DATAPATH $PARAMPATH $VERBOSE $VISUALIZE $SAVEPATH $OFFSETPATH

###### Rendering videos ######
DATAFOLDER="$SAVEPATH/*.p"
SAVEFOLDER="/home/diego/data/dm/stac/clips/JDM33_v2/"

DATAPATHS=($(find $DATAFOLDER))
FILES=($(find $DATAFOLDER -printf "%f\n"))

SAVEPATHS=()
cnt=${#FILES[@]}
for ((i=0;i<cnt;i++)); do
  SAVEPATHS[i]="$SAVEFOLDER${FILES[i]%.p}.mp4"
done

cnt=${#FILES[@]}
for i in $(seq 0 $[$cnt-1]);
  do
  echo ${DATAPATHS[$i]} $PARAMPATH ${SAVEPATHS[$i]};
  srun -l cluster/render.sh ${DATAPATHS[$i]} $PARAMPATH ${SAVEPATHS[$i]}
done
