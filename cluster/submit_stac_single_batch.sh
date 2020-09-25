#!/bin/bash
#SBATCH -J compute_stac
#SBATCH -N 1                # number of nodes
#SBATCH -c 1               # Number of threads (cores)
#SBATCH --mem 15000        # memory for all cores
#SBATCH -t 0-05:00          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o logs/Job.compute_stac.%N.%j.out    # STDOUT
#SBATCH -e logs/Job.compute_stac.%N.%j.err    # STDERR
setup_mujoco200_3.7
stac_path="/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/stac"
stac-compute-single-batch $1