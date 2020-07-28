#!/bin/bash
source /home/.envs/mujoco200_3.7/bin/activate
export img_path="/n/home02/daldarondo/LabDir/Diego/.images/mj_stac.sif"
export MUJOCO_GL="osmesa"
export MJLIB_PATH="/home/.mujoco/mujoco200_linux/bin/libmujoco200.so"
export MJKEY_PATH="/home/.mujoco/mjkey.txt"
stac_path="/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/stac"
python3 $stac_path/batch_compute_stac_clip_singularity.py $1
