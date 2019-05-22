#!/bin/bash
DATAPATH="/home/diego/data/dm/stac/snippets/JDM33"
VERBOSE="True"
OFFSETPATH="/home/diego/data/dm/stac/offsets/JDM33_offset.p"
VISUALIZE="False"
NSNIPPETS=15
SAVEPATH="/home/diego/data/dm/stac/results/JDM33"
PARAMPATH="/home/diego/code/olveczky/dm/stac/params/JDM33.yaml"
seq 7 $NSNIPPETS | xargs -i --max-procs=7 python compute_stac.py $DATAPATH $PARAMPATH --n-snip={} --verbose=$VERBOSE --visualize=$VISUALIZE --save-path=$SAVEPATH --offset-path=$OFFSETPATH
