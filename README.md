# stac
Implementation of simultaneous tracking and calibration (STAC) for calibrating mujoco models with keypoint data using the [dm_control suite](https://github.com/deepmind/dm_control).

[Original paper](https://ieeexplore.ieee.org/abstract/document/7030016)

## Requirements and Installation
As `stac` relies on `mujoco` and `dm_control`, we recommend building within a `virtualenv`.

`stac` has been tested with python 3.6 and Ubuntu 18.04, CentoOS7, and Windows 10. 

* Install prerequisites using the included setup scripts.
```
python setup.py install
```

* Follow the instructions [here](https://github.com/deepmind/dm_control) to install `dm_control`. If environment is correctly configured, this is as simple as:

## Authors
* [Diego Aldarondo](https://github.com/diegoaldarondo)
* [Josh Merel](https://github.com/jsmerel)
* [Jesse Marshall](https://github.com/jessedmarshall)
* [Bence Olveczky](https://olveczkylab.oeb.harvard.edu/)
