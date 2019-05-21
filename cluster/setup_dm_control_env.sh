#!/bin/bash
module load python/3.6.3-fasrc02
python -m virtualenv "~/.envs/mujoco200_3.6"
source ./mujoco200_3.6/bin/activate
pip3 install six absl-py enum34 future futures glfw lxml numpy pyopengl pyparsing
pip3 install h5py scipy
pip3 install git+git://github.com/deepmind/dm_control.git

# Test installation_
python -c "from dm_control import mujoco"
