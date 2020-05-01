#!/bin/bash
#SBATCH -J compute_stac
#SBATCH -N 1                # number of nodes
#SBATCH -n 1                # number of tasks
#SBATCH -c 8               # Number of threads (cores)
#SBATCH --mem 20000        # memory for all cores
#SBATCH -t 0-05:00          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o logs/Job.compute_stac.%N.%j.out    # STDOUT
#SBATCH -e logs/Job.compute_stac.%N.%j.err    # STDERR
img_path="/n/home02/daldarondo/LabDir/Diego/.images/mj_stac.sif"
data_path=$1; shift
param_path=$1; shift
save_path=$1; shift
offset_path=$1; shift
start_frame=( "$@" )
start_frame=${start_frame[$SLURM_ARRAY_TASK_ID]}
save_path=$save_path"/$start_frame.p"
echo singularity exec $img_path bash /home/compute_stac_clip.sh $data_path $param_path $save_path $offset_path $start_frame
singularity exec $img_path bash /home/compute_stac_clip.sh $data_path $param_path $save_path $offset_path $start_frame
