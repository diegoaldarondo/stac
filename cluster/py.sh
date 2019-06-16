echo "$@"
module load python/3.6.3-fasrc02
module load bazel/0.13.0-fasrc01 gcc/4.9.3-fasrc01 hdf5/1.8.12-fasrc08 cmake
source ~/.envs/mujoco200_python3.6/bin/activate
python "$@"
