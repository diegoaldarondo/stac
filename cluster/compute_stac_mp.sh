#!/bin/bash
DATAPATH="/home/diego/data/dm/stac/snippets/JDM25"
VERBOSE="True"
# OFFSETPATH="./results/out2000_e4.p"
OFFSETPATH="./results/JDM25_offset_new_offset_m_reg25_5_17.p"
VISUALIZE="False"
NSNIPPETS=292

seq $NSNIPPETS | xargs -i --max-procs=7 python compute_stac.py $DATAPATH --n-snip={} --offset-path=$OFFSETPATH --verbose=$VERBOSE --visualize=$VISUALIZE
