#!/bin/bash
DATAPATH="/home/diego/code/olveczky/dm/stac/results/JDM25_v3/snippet{}.p"
SAVEPATH="/home/diego/code/olveczky/dm/stac/clips/JDM25_v3_2/snippet{}.mp4"
NSNIPPETS=186

seq 186 $NSNIPPETS | xargs -i --max-procs=8 python view_stac.py $DATAPATH --render-video="True" --save-path=$SAVEPATH --headless="True"
