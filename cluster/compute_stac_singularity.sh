#!/bin/bash
set -e
data_folder=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/snippets/JDM25_v5/reformatted/*.mat
data_paths=($(find $data_folder))
files=($(find $data_folder -printf "%f\n"))

offset_path=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/offsets/JDM25_m_9_NoHip.p
save_path=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/results/JDM25_v16
mkdir $save_path
param_path=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/june3/JDM25.yaml

cnt=${#files[@]}
cnt=$(($cnt - 1))
sbatch --array=0-$cnt --partition=shared,general,olveczky cluster/submit_compute_stac.sh $param_path $save_path $offset_path ${data_paths[*]}


data_folder=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/snippets/JDM33/reformatted/*.mat
data_paths=($(find $data_folder))
files=($(find $data_folder -printf "%f\n"))

offset_path=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/offsets/JDM33_june18.p
save_path=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/results/JDM33_v8
mkdir $save_path
param_path=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/june3/JDM33.yaml

cnt=${#files[@]}
cnt=$(($cnt - 1))
sbatch --array=0-$cnt --partition=shared,general,olveczky cluster/submit_compute_stac.sh $param_path $save_path $offset_path ${data_paths[*]}


data_folder=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/snippets/JDM31/reformatted/*.mat
data_paths=($(find $data_folder))
files=($(find $data_folder -printf "%f\n"))

offset_path=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/offsets/JDM31_june18.p
save_path=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/results/JDM31_v5
mkdir $save_path
param_path=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/june3/JDM31.yaml

cnt=${#files[@]}
cnt=$(($cnt - 1))
sbatch --array=0-$cnt --partition=shared,general,olveczky cluster/submit_compute_stac.sh $param_path $save_path $offset_path ${data_paths[*]}
