#!/bin/bash
# Script to run all steps in converting dannce predictions to npmp embeddings in a single job. 
# Must be launched from the project folder. 
#
# Several files need to be configured prior to running this script.
# 
# 1. Stac model params file in ./stac_params/params.yaml
# 2. Stac submission params file in ./stac_params/submission_params.yaml
# 3. You should have made an offset.p file in ./stac . Use initial_offsets notebook in stac repo. 
# 4. The trajectory datasets need to be registered in dm_control/locomotion/tasks/reference_pose/rat_subsets.py
#    - Name the dataset with a name that does not already exist. (Ex. dannce_rig_1)
#    - Construct a ClipCollection with the names, as in other examples in the code.
#    - Add the clip collection to the RAT_SUBSETS dict
#    - Rebuild dm_control (python setup.py install)
# 
# Inputs: dataset - Name of trajectory dataset registered in dm_control.
# Example: sbatch dannce2npmp.sh dataset_name

# Ask for a single core on the olveczky partition to serve as a dispatcher and merger. 
#SBATCH --job-name=dannce2npmp
#SBATCH --mem=20000
#SBATCH -t 2-00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky

set -e
# Load the environment setup functions
source ~/.bashrc
setup_mujoco200_3.7

# Make the log directory if it doesn't exist
# mkdir -p logs

# Stac data and merge if successful
# stac-submit ./stac_params/submission_params.yaml && stac-merge ./stac

# Preprocess npmp and merge if successful
# parallel-npmp-preprocessing ./stac/total.p ./npmp_preprocessing && merge-npmp-preprocessing ./npmp_preprocessing

# Embed npmp
mkdir -p npmp
mkdir -p npmp/model_3_no_noise
dispatch-npmp-embed ./npmp_preprocessing/total.hdf5 ./npmp/model_3_no_noise_segmented $1 --stac-params=./stac_params/params.yaml --offset-path=./stac/offset.p
