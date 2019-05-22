#!/bin/bash
DATAPATH="/home/diego/data/dm/stac/snippets/JDM25"
VERBOSE="True"
# OFFSETPATH="./results/out2000_e4.p"
OFFSETPATH="/home/diego/data/dm/stac/results/JDM25_offset_new_offset_m_reg25_5_17.p"
VISUALIZE="False"
NSNIPPETS=1
SAVEPATH="/home/diego/data/dm/stac/results/JDM25_v4"
PARAMPATH="/home/diego/code/olveczky/dm/stac/params/JDM25.yaml"
seq $NSNIPPETS | xargs -i --max-procs=1 python compute_stac.py $DATAPATH $PARAMPATH --n-snip={} --verbose=$VERBOSE --visualize=$VISUALIZE --save-path=$SAVEPATH --offset-path=$OFFSETPATH
# seq $NSNIPPETS | xargs -i --max-procs=1 python compute_stac.py $DATAPATH $PARAMPATH --n-snip={} --verbose=$VERBOSE --visualize=$VISUALIZE --save-path=$SAVEPATH
