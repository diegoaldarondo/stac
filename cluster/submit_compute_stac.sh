#!/bin/bash
#SBATCH -J compute_stac
#SBATCH -p shared     # partition (queue)
#SBATCH -N 1                # number of nodes
#SBATCH -n 1                # number of tasks
#SBATCH --mem 10000        # memory for all cores
#SBATCH -t 0-01:00          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o logs/Job.compute_stac.%N.%j.out    # STDOUT
#SBATCH -e logs/Job.compute_stac.%N.%j.err    # STDERR

singularity exec mj_stac.sif bash /home/compute_stac.sh $1 $2 $3 $4
