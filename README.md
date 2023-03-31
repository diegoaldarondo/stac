# STAC
Implementation of simultaneous tracking and calibration ([STAC](https://ieeexplore.ieee.org/abstract/document/7030016)) for calibrating mujoco models with keypoint data using the [dm_control suite](https://github.com/deepmind/dm_control).


## Requirements and Installation
Stac has been tested on Windows 10, Windows 11, Ubuntu 18.04, and CentOS7 with python 3.9. In Linux, one can install stac in a conda virtual environment using the following bash script (Windows can use a bash emulator).

```
git clone https://github.com/diegoaldarondo/stac
cd stac
conda create -n stac python=3.9
conda activate stac
pip install -e .
```

## Authors
* [Diego Aldarondo](https://github.com/diegoaldarondo)
* [Josh Merel](https://github.com/jsmerel)
* [Jesse Marshall](https://github.com/jessedmarshall)
* [Bence Olveczky](https://olveczkylab.oeb.harvard.edu/)
