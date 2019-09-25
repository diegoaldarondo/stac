#!/bin/bash
set -e
base_folder=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/august15
mkdir $base_folder


# data_folder=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/snippets/snippets_snippet_v7_JDM25/reformatted/*.mat
# data_paths=($(find $data_folder))
# files=($(find $data_folder -printf "%f\n"))
# offset_path=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM25.p
# save_path="$base_folder"/JDM25
# mkdir $save_path
# param_path=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/july15/JDM25.yaml
# cnt=${#files[@]}
# cnt=$(($cnt - 1))
# sbatch --array=0-$cnt --partition=shared cluster/submit_compute_stac.sh $param_path $save_path $offset_path ${data_paths[*]}


# data_folder=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/snippets/snippets_snippet_v6_JDM33/reformatted/*.mat
# data_paths=($(find $data_folder))
# files=($(find $data_folder -printf "%f\n"))
# offset_path=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM33.p
# save_path="$base_folder"/JDM33
# mkdir $save_path
# param_path=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/july15/JDM33.yaml
# cnt=${#files[@]}
# cnt=$(($cnt - 1))
# sbatch --array=0-$cnt --partition=shared cluster/submit_compute_stac.sh $param_path $save_path $offset_path ${data_paths[*]}
#
#

data_folder=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/snippets/snippets_snippet_v7_JDM31/reformatted/*.mat
data_paths=($(find $data_folder))
files=($(find $data_folder -printf "%f\n"))
offset_path=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM31.p
save_path="$base_folder"/JDM31
mkdir $save_path
param_path=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/july15/JDM31.yaml
cnt=${#files[@]}
cnt=$(($cnt - 1))
sbatch --array=0-$cnt --partition=shared cluster/submit_compute_stac.sh $param_path $save_path $offset_path ${data_paths[*]}
