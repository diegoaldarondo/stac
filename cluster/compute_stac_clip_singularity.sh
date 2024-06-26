#!/bin/bash
set -e
base_folder=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/JDM31_full_day
# mkdir $base_folder

# data_path=/n/home02/daldarondo/LabDir/Jesse/Data/Motionanalysis_captures/DANNCE_animals/DANNCE_mujoco_test/predictions_preprocessed_J31.mat
data_path=/n/home02/daldarondo/LabDir/Jesse/Data/Dropbox_curated_sharefolders/mocap_data_diego/diego_mocap_files_rat_JDM31_day_8.mat
offset_path=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM31.p
param_path=/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/july15/JDM31_DANNCE.yaml
snippet_duration=3600
clip_duration=28080000
# clip_duration=3600
start_frame=($(seq 0 $snippet_duration $clip_duration))
cnt=${#start_frame[@]}
cnt=$(($cnt - 1))
save_path=$base_folder
sbatch --array=0-$cnt --wait --partition=shared,olveczky,serial_requeue cluster/submit_compute_stac_clip.sh $data_path $param_path $save_path $offset_path ${start_frame[*]}
wait
sbatch --partition=shared,olveczky,serial_requeue cluster/submit_merge_stac_clips.sh $base_folder
