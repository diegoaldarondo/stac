#!/bin/bash
DATAPATH="/home/diego/data/dm/stac/snippets/JDM25_v5/reformatted"
VERBOSE="True"
OFFSETPATH="/home/diego/data/dm/stac/offsets/JDM25_m_9_NoHip.p"
VISUALIZE="False"
NSNIPPETS=380
SAVEPATH="/home/diego/data/dm/stac/results/JDM25_v10"
PARAMPATH="/home/diego/code/olveczky/dm/stac/params/june3/JDM25.yaml"
seq $NSNIPPETS | xargs -i --max-procs=8 python compute_stac.py $DATAPATH $PARAMPATH --n-snip={} --verbose=$VERBOSE --visualize=$VISUALIZE --save-path=$SAVEPATH --offset-path=$OFFSETPATH
