#!/bin/bash
#SBATCH -J compute_stac
#SBATCH -N 1                # number of nodes
#SBATCH -n 1                # number of tasks
#SBATCH -c 8               # Number of threads (cores)
#SBATCH --mem 3000        # memory for all cores
#SBATCH -t 0-00:45          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o logs/Job.compute_stac.%N.%j.out    # STDOUT
#SBATCH -e logs/Job.compute_stac.%N.%j.err    # STDERR
img_path="/n/home02/daldarondo/LabDir/Diego/.images/mj_stac.sif"
param_path=$1; shift
save_path=$1; shift
offset_path=$1; shift
data_path=( "$@" )
echo singularity exec $img_path bash /home/compute_stac.sh ${data_path[$SLURM_ARRAY_TASK_ID]} $param_path $save_path $offset_path
singularity exec $img_path bash /home/compute_stac.sh ${data_path[$SLURM_ARRAY_TASK_ID]} $param_path $save_path $offset_path
