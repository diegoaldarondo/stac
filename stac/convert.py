"""Module to convert qpos files to sites for arbitrary marker sets.
"""
import pickle
from stac import util
from stac import rodent_environments
from dm_control import viewer
from scipy.io import savemat
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco.wrapper.mjbindings import mjlib
from typing import Text, Dict
import os
import numpy as np


def load_data(file_path: Text) -> Dict:
    """Load data from a chunk

    Args:
        file_path (Text): Path to a qpos pickle file.

    Returns:
        Dict: Model registration dictionary.
    """
    with open(file_path, "rb") as f:
        in_dict = pickle.load(f)
    return in_dict


def convert(file_path, offset_path, params_path, save_path):
    """Summary

    Args:
        file_path (TYPE): Path to the qpos pickle file.
        offset_path (TYPE): Path to the offset pickle file.
        params_path (TYPE): Path the the stac params .yaml file.
        save_path (TYPE): Path the the output .mat file.
    """
    # import pdb
    # pdb.set_trace()
    # Load the parameters and data
    params = util.load_params(params_path)
    data = load_data(file_path)

    
    params["n_frames"] = data["kp_data"].shape[0]
    params["n_frames"] = int(params["n_frames"])

    # Build the environment
    env = rodent_environments.rodent_mocap(data["kp_data"], params)
    rescale.rescale_subtree(
        env.task._walker._mjcf_root,
        params["scale_factor"],
        params["scale_factor"],
    )
    mjlib.mj_kinematics(env.physics.model.ptr, env.physics.data.ptr)
    # Center of mass position
    mjlib.mj_comPos(env.physics.model.ptr, env.physics.data.ptr)
    env.reset()

    # Load the offsets and set the sites to those positions.
    with open(params["offset_path"], "rb") as f:
        in_dict = pickle.load(f)

    sites = env.task._walker.body_sites
    env.physics.bind(sites).pos[:] = in_dict["offsets"]
    for n_site, p in enumerate(env.physics.bind(sites).pos):
        sites[n_site].pos = p

    # Set the environment with the qpos.
    env.task.precomp_qpos = data["qpos"]

    env.task.initialize_episode(env.physics, 0)
    prev_time = env.physics.time()

    sites = np.copy(env.physics.bind(env.task._walker.body_sites).xpos[:])

    # Loop through the clip, saving the sites on each frame.
    n_frame = 0
    walker_body_sites = np.zeros(
        (params["n_frames"], sites.shape[0], sites.shape[1], sites.shape[2])
    )
    while prev_time < env._time_limit:
        while (np.round(env.physics.time() - prev_time, decimals=5)) < params[
            "_TIME_BINS"
        ]:
            env.physics.step()
        env.task.after_step(env.physics, None)
        walker_body_sites[n_frame, ...] = np.copy(
            env.physics.bind(env.task._walker.body_sites).xpos[:]
        )
        n_frame += 1
        print(n_frame)
        prev_time = np.round(env.physics.time(), decimals=2)

    # Save the results
    if not os.exists(os.dirname(save_path)):
        os.makedirs(os.dirname(save_path))
    savemat(save_path, {"walker_body_sites", walker_body_sites})
    # with open(save_path, "wb") as f:
    #     pickle.dump(walker_body_sites, f)


if __name__ == "__main__":
    file_path = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac/total.p"
    offset_path = (
        "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM25.p"
    )
    params_path = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac_params/params.yaml"
    save_path = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/mocap_conversion/data.mat"
    convert(file_path, offset_path, params_path, save_path)
