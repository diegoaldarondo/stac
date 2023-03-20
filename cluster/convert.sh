#!/bin/bash
#SBATCH -J convert_mocap
#SBATCH -N 1                # number of nodes
#SBATCH -c 1               # Number of threads (cores)
#SBATCH -p olveczky,shared,serial_requeue # Number of threads (cores)
#SBATCH --mem 5000        # memory for all cores
#SBATCH -t 1-00:00          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o /dev/null    # STDOUT
#SBATCH -e /dev/null    # STDERR
#SBATCH --constraint="intel&avx2"
source ~/.bashrc
setup_mujoco210_3.7
python -c "import convert; convert.submit()"
