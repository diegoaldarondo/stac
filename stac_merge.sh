#!/bin/bash
#SBATCH --job-name=stac_merge
#SBATCH --mem=40000
#SBATCH -t 0-01:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky
#SBATCH --constraint="intel&avx2"
# Ask for a single core on the olveczky partition to serve as a dispatcher and merger. 
# Script to run all steps in converting dannce predictions to npmp embeddings in a single job. 
# Must be launched from the project folder. 
#
# Several files need to be configured prior to running this script.
# 
# 1. Stac model params file in ./stac_params/params.yaml
# 2. Stac submission params file in ./stac_params/submission_params.yaml
# 3. You should have made an offset.p file in ./stac . Use initial_offsets notebook in stac repo. 
# 
# Example: sbatch stac_and_preprocess.sh 
set -e

# Load the environment setup functions
source ~/.bashrc
setup_mujoco200_3.7

stac-merge ./stac

