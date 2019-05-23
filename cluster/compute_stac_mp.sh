#!/bin/bash
DATAPATH="/home/diego/data/dm/stac/snippets/JDM33/reformatted"
VERBOSE="True"
OFFSETPATH="/home/diego/data/dm/stac/offsets/JDM33_offset_2.p"
VISUALIZE="False"
NSNIPPETS=611
SAVEPATH="/home/diego/data/dm/stac/results/JDM33_v2"
PARAMPATH="/home/diego/code/olveczky/dm/stac/params/JDM33.yaml"
seq $NSNIPPETS | xargs -i --max-procs=7 python compute_stac.py $DATAPATH $PARAMPATH --n-snip={} --verbose=$VERBOSE --visualize=$VISUALIZE --save-path=$SAVEPATH --offset-path=$OFFSETPATH
