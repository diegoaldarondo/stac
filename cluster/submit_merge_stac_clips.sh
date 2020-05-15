#!/bin/bash
#SBATCH -J merge_stac
#SBATCH -p olveczky
#SBATCH -N 1                # number of nodes
#SBATCH -n 1                # number of tasks
#SBATCH -c 1               # Number of threads (cores)
#SBATCH --mem 60000        # memory for all cores
#SBATCH -t 0-00:10          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o logs/Job.merge_stac.%N.%j.out    # STDOUT
#SBATCH -e logs/Job.merge_stac.%N.%j.err    # STDERR
img_path="/n/home02/daldarondo/LabDir/Diego/.images/mj_stac.sif"
data_path=$1; shift
stac_path="/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/stac"
singularity exec $img_path bash $stac_path/cluster/merge_stac.sh $data_path
