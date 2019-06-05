"""Utility functions to convert from mocap positions to mujoco sensors."""
import numpy as np
import h5py
import os
import yaml


def load_params(param_path):
    """Load parameters for the animal.

    :param param_path: Path to .yaml file specifying animal parameters.
    """
    with open(param_path, 'r') as infile:
        try:
            params = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def load_kp_data_from_file(filename, struct_name='markers_preproc'):
    """Format kp_data files from matlab to python through hdf5.

    :param filename: Path to v7.3 mat file containing
    :param struct_name: Name of the struct to load.
    """
    with h5py.File(filename, 'r') as f:
        data = f[struct_name]
        kp_names = [k for k in data.keys()]

        # Concatenate the data for each keypoint, and format to (t x n_dims)
        kp_data = \
            np.concatenate([data[name][:] for name in kp_names]).T
        return kp_data, kp_names


def load_snippets_from_file(folder):
    """Load snippets from file and return list of kp_data."""
    files = [f for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f)) and 'params' not in f]
    filenames = [os.path.join(folder, f) for f in files]
    snippets = [None] * len(filenames)
    behaviors = [None] * len(filenames)
    com_vels = [None] * len(filenames)
    for i, filename in enumerate(filenames):
        with h5py.File(filename, 'r') as f:
            try:
                data = f['data']
            except KeyError:
                data = f['preproc_mocap']
            kp_names = [k for k in data.keys()]
            # Concatenate the data for each keypoint,
            # and format to (t x n_dims)
            snippets[i] = \
                np.concatenate([data[name][:] for name in kp_names]).T
    return snippets, kp_names, files, behaviors, com_vels
