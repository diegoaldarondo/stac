#!/bin/bash
#SBATCH -J stac
#SBATCH -p olveczky     # partition (queue)
#SBATCH -N 1                # number of nodes
#SBATCH -n 1                # number of tasks
#SBATCH --mem 20000        # memory for all cores
#SBATCH -t 0-02:00          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o logs/Job.stac.%N.%j.out    # STDOUT
#SBATCH -e logs/Job.stac.%N.%j.err    # STDERR

FUNC=compute_stac.py
DATAPATH=$1
SAVEPATH=$2
FRAMESTART=$3
DURATION=$4

srun -l cluster/py.sh $FUNC $DATAPATH
