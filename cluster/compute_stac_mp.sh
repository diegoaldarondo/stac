#!/bin/bash
DATAPATH="/home/diego/data/dm/stac/snippets/JDM25_v5/reformatted"
VERBOSE="True"
OFFSETPATH="/home/diego/data/dm/stac/offsets/test.p"
VISUALIZE="False"
NSNIPPETS=380
SAVEPATH="/home/diego/data/dm/stac/results/JDM25_v7"
PARAMPATH="/home/diego/code/olveczky/dm/stac/params/baseParams.yaml"
seq $NSNIPPETS | xargs -i --max-procs=7 python compute_stac.py $DATAPATH $PARAMPATH --n-snip={} --verbose=$VERBOSE --visualize=$VISUALIZE --save-path=$SAVEPATH --offset-path=$OFFSETPATH
