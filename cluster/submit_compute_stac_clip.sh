#!/bin/bash
#SBATCH -J compute_stac
#SBATCH -N 1                # number of nodes
#SBATCH -n 1                # number of tasks
<<<<<<< HEAD
#SBATCH -c 1               # Number of threads (cores)
#SBATCH --mem 15000        # memory for all cores
=======
#SBATCH -c 8               # Number of threads (cores)
#SBATCH --mem 20000        # memory for all cores
>>>>>>> 493409a8c72119d325f865113f7962d6ca15e888
#SBATCH -t 0-05:00          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o logs/Job.compute_stac.%N.%j.out    # STDOUT
#SBATCH -e logs/Job.compute_stac.%N.%j.err    # STDERR
img_path="/n/home02/daldarondo/LabDir/Diego/.images/mj_stac.sif"
stac_path="/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/stac"
singularity exec $img_path bash $stac_path/cluster/compute_stac_clip.sh
