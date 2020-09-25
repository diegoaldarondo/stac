"""Utility functions to load data from .mat .yaml and .h5 files."""
import numpy as np
import h5py
import os
import yaml
from scipy.io import loadmat


def load_params(param_path):
    """Load parameters for the animal.

    :param param_path: Path to .yaml file specifying animal parameters.
    """
    with open(param_path, "r") as infile:
        try:
            params = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def load_kp_data_from_file(
    filename, struct_name="markers_preproc", start_frame=None, end_frame=None
):
    """Format kp_data files from matlab to python through hdf5.

    :param filename: Path to v7.3 mat file containing
    :param struct_name: Name of the struct to load.
    """
    try:
        with h5py.File(filename, "r") as f:
            data = f["mocapstruct_here"][struct_name]
            kp_names = [k for k in data.keys()]

        # Concatenate the data for each keypoint, and format to (t x n_dims)
        if start_frame is None:
            kp_data = np.concatenate([data[name][:] for name in kp_names]).T
        else:
            markers = []
            for name in kp_names:
                print(name, flush=True)
                m = data[name][:]
                markers.append(m[:, start_frame:end_frame])
            kp_data = np.concatenate(markers).T
    except OSError:
        data = loadmat(filename)
        data = data["predictions"]
        kp_names = [k for k in data.dtype.names if not any([n in k for n in ["Shin"]])]
        kp_names.sort()
        if start_frame is None:
            kp_data = np.concatenate([data[name][0, 0] for name in kp_names], axis=1)
        else:
            kp_data = np.concatenate(
                [data[name][0, 0][start_frame:end_frame, :] for name in kp_names],
                axis=1,
            )
    return kp_data, kp_names


def load_snippets_from_file(filename):
    """Load snippet."""
    with h5py.File(filename, "r") as f:
        try:
            data = f["data"]
        except KeyError:
            data = f["preproc_mocap"]
        kp_names = [k for k in data.keys()]
        # Concatenate the data for each keypoint,
        # and format to (t x n_dims)
        snippet = np.concatenate([data[name][:] for name in kp_names]).T
        behavior = "".join(chr(x) for x in f["clustername"][:])
        # com_vel = f['snippet_com_vel'][:][0]
        com_vel = 0
    return snippet, kp_names, behavior, com_vel


def load_snippets_from_folder(folder):
    """Load snippets from file and return list of kp_data."""
    files = [
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and "params" not in f
    ]
    filenames = [os.path.join(folder, f) for f in files]
    snippets = [None] * len(filenames)
    behaviors = [None] * len(filenames)
    com_vels = [None] * len(filenames)
    for i, filename in enumerate(filenames):
        snippets[i], kp_names, behaviors[i], com_vels[i] = load_snippets_from_file(
            filename
        )
    return snippets, kp_names, files, behaviors, com_vels
