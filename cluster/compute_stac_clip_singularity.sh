#!/bin/bash
set -e
base_folder=/n/home02/daldarondo/LabDir/Diego/data/dm/stac/results/long_clips/JDM31
mkdir $base_folder

data_path=/n/home02/daldarondo/LabDir/Jesse/Data/Motionanalysis_captures/DANNCE_animals/DANNCE_mujoco_test/predictions_preprocessed_J31.mat
offset_path=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM31.p
param_path=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/july15/JDM31_DANNCE.yaml
snippet_duration=500
clip_duration=100000
start_frame=($(seq 0 $snippet_duration $clip_duration))
cnt=${#start_frame[@]}
cnt=$(($cnt - 1))
save_path=$base_folder
sbatch --array=0-$cnt --wait --partition=shared cluster/submit_compute_stac_clip.sh $data_path $param_path $save_path $offset_path ${start_frame[*]}
wait
sbatch --partition=shared cluster/submit_merge_stac_clips.sh $base_folder
