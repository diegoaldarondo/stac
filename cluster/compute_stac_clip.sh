#!/bin/bash
source /home/.envs/mujoco200_3.7/bin/activate
export img_path="/n/home02/daldarondo/LabDir/Diego/.images/mj_stac.sif"
export MUJOCO_GL="osmesa"
export MJLIB_PATH="/home/.mujoco/mujoco200_linux/bin/libmujoco200.so"
export MJKEY_PATH="/home/.mujoco/mjkey.txt"
<<<<<<< HEAD
stac_path="/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/stac"
python3 $stac_path/batch_compute_stac_clip_singularity.py
=======
cd /home/code/stac
data_path=$1
param_path=$2
save_path=$3
offset_path=$4
start_frame=$5
python3 compute_stac.py $data_path $param_path --save-path=$save_path --offset-path=$offset_path --start-frame=$start_frame --verbose="True" --process-snippet="False" --skip=1
>>>>>>> 493409a8c72119d325f865113f7962d6ca15e888
