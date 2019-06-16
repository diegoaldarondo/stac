#!/bin/bash
#SBATCH -J render
#SBATCH -p olveczky     # partition (queue)
#SBATCH -N 1                # number of nodes
#SBATCH -n 1                # number of tasks
#SBATCH --mem 20000        # memory for all cores
#SBATCH -t 0-01:00          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o logs/Job.render.%N.%j.out    # STDOUT
#SBATCH -e logs/Job.render.%N.%j.err    # STDERR

FUNC="view_stac.py"
DATAPATH=$1
PARAMPATH=$2
SAVEPATH=$3
srun -l cluster/py.sh $FUNC $DATAPATH $PARAMPATH  --render-video="True" --save-path=$SAVEPATH --headless="True"
