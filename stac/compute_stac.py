"""Compute stac optimization on data.
"""
from scipy.io import savemat
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco.wrapper.mjbindings import mjlib
import stac.stac_base as stac_base
import stac.rodent_environments as rodent_environments
import numpy as np
import stac.util as util
import pickle
import os
from typing import List, Dict, Tuple, Text
from scipy.io import loadmat

_MM_TO_METERS = 1000


def initial_optimization(env, offsets: np.ndarray, params: Dict, maxiter: int = 100):
    """Optimize the first frame with alternating q and m phase.

    Args:
        env (TYPE): Environment
        offsets (np.ndarray): Vector of starting offsets for initial q_phase
        params (Dict): parameter dictionary
        maxiter (int, optional): Maximum number of iterations for m-phase optimization
    """
    root_optimization(env, params)

    # Initial q-phase optimization to get joints into approximate position.
    q, _, _ = pose_optimization(env, params)

    # Initial m-phase optimization to calibrate offsets
    stac_base.m_phase(
        env.physics,
        env.task.kp_data,
        env.task._walker.body_sites,
        np.arange(params["n_frames"]),
        q,
        offsets,
        params,
        reg_coef=params["M_REG_COEF"],
        maxiter=maxiter,
    )


def root_optimization(env, params: Dict, frame: int = 0):
    """Optimize only the root.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary
        frame (int, optional): Frame to optimize
    """
    stac_base.q_phase(
        env.physics,
        env.task.kp_data[frame, :],
        env.task._walker.body_sites,
        params,
        root_only=True,
    )

    # First optimize over the trunk
    trunk_kps = [
        any([n in kp_name for n in ["Spine", "Hip", "Shoulder", "Offset"]])
        for kp_name in params["kp_names"]
    ]
    trunk_kps = np.repeat(np.array(trunk_kps), 3)
    stac_base.q_phase(
        env.physics,
        env.task.kp_data[frame, :],
        env.task._walker.body_sites,
        params,
        root_only=True,
        kps_to_opt=trunk_kps,
    )


def get_part_ids(env, parts: List) -> np.ndarray:
    """Get the part ids given a list of parts.

    Args:
        env (TYPE): Environment
        parts (List): List of part names

    Returns:
        np.ndarray: Array of part identifiers
    """
    part_names = env.physics.named.data.qpos.axes.row.names
    return np.array([any(part in name for part in parts) for name in part_names])


def offset_optimization(env, offsets, q, params: Dict):
    time_indices = np.random.randint(0, params["n_frames"], params["N_SAMPLE_FRAMES"])
    stac_base.m_phase(
        env.physics,
        env.task.kp_data,
        env.task._walker.body_sites,
        time_indices,
        q,
        offsets,
        params,
        reg_coef=params["M_REG_COEF"],
    )


def pose_optimization(env, params: Dict) -> Tuple:
    """Perform q_phase over the entire clip.

    Optimizes limbs and head independently.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary.

    Returns:
        Tuple: qpos, walker body sites, xpos
    """
    q = []
    x = []
    walker_body_sites = []
    if params["INDIVIDUAL_PART_OPTIMIZATION"] is None:
        indiv_parts = []
    else:
        indiv_parts = [
            get_part_ids(env, parts)
            for parts in params["INDIVIDUAL_PART_OPTIMIZATION"].values()
        ]

    # Iterate through all of the frames in the clip
    for n_frame in range(params["n_frames"]):
        # Optimize over all points
        stac_base.q_phase(
            env.physics,
            env.task.kp_data[n_frame, :],
            env.task._walker.body_sites,
            params,
        )

        # Next optimize over parts individually to improve time and accur.
        for part in indiv_parts:
            stac_base.q_phase(
                env.physics,
                env.task.kp_data[n_frame, :],
                env.task._walker.body_sites,
                params,
                qs_to_opt=part,
            )
        q.append(np.copy(env.physics.named.data.qpos[:]))
        x.append(np.copy(env.physics.named.data.xpos[:]))
        walker_body_sites.append(
            np.copy(env.physics.bind(env.task._walker.body_sites).xpos[:])
        )

    return q, walker_body_sites, x


def build_env(kp_data: np.ndarray, params: Dict):
    """Builds the environment for the keypoint data.

    Args:
        kp_data (np.ndarray): Key point data.
        params (Dict): Parameters for the environment.

    Returns:
        : The environment
    """
    env = rodent_environments.rodent_mocap(kp_data, params)
    rescale.rescale_subtree(
        env.task._walker._mjcf_root,
        params["SCALE_FACTOR"],
        params["SCALE_FACTOR"],
    )
    mjlib.mj_kinematics(env.physics.model.ptr, env.physics.data.ptr)
    mjlib.mj_comPos(env.physics.model.ptr, env.physics.data.ptr)
    env.reset()
    return env


def initialize_part_names(env):
    # Get the ids of the limbs, accounting for quaternion and pos
    part_names = env.physics.named.data.qpos.axes.row.names
    for _ in range(6):
        part_names.insert(0, part_names[0])
    return part_names


def package_data(env, q, x, walker_body_sites, part_names, kp_data, params):
    # Extract pose, offsets, data, and all parameters
    offsets = env.physics.bind(env.task._walker.body_sites).pos[:].copy()
    names_xpos = env.physics.named.data.xpos.axes.row.names
    data = {
        "qpos": q,
        "xpos": x,
        "walker_body_sites": walker_body_sites,
        "offsets": offsets,
        "names_qpos": part_names,
        "names_xpos": names_xpos,
        "kp_data": np.copy(kp_data),
    }
    for k, v in params.items():
        data[k] = v
    return data


class STAC:
    def __init__(
        self,
        param_path: Text,
    ):
        """Initialize STAC

        Args:
            param_path (Text): Path to parameters .yaml file.
        """
        self._properties = util.load_params(param_path)
        self._properties["data"] = None
        self._properties["n_frames"] = None

        # Default ordering of mj sites is alphabetical, so we reorder to match
        kp_names = loadmat(self._properties["SKELETON_PATH"])["joint_names"]
        self._properties["kp_names"] = [name[0] for name in kp_names[0]]
        self._properties["stac_keypoint_order"] = np.argsort(
            self._properties["kp_names"]
        )
        for property_name in self._properties.keys():

            def getter(self, name=property_name):
                return self._properties[name]

            def setter(self, value, name=property_name):
                self._properties[name] = value

            setattr(STAC, property_name, property(fget=getter, fset=setter))

    def _prepare_data(self, kp_data: np.ndarray) -> np.ndarray:
        """Prepare the data for STAC.

        Args:
            kp_data (np.ndarray): Keypoint data in meters (n_frames, 3, n_keypoints).

        Returns:
            np.ndarray: Keypoint data in meters (n_frames, n_keypoints * 3).
        """
        kp_data = kp_data[:, :, self.stac_keypoint_order]
        kp_data = np.transpose(kp_data, (0, 2, 1))
        kp_data = np.reshape(kp_data, (kp_data.shape[0], -1))
        return kp_data

    def fit(self, kp_data: np.ndarray) -> "STAC":
        """Calibrate and fit the model to keypoints.

        Performs three rounds of alternating marker and quaternion optimization. Optimal
        results with greater than 200 frames of data in which the subject is moving.

        Args:
            keypoints (np.ndarray): Keypoint data in meters (n_frames, 3, n_keypoints).

        Example:
            st = st.fit(keypoints)

        Returns:

        """
        kp_data = self._prepare_data(kp_data)
        self.n_frames = kp_data.shape[0]
        env = build_env(kp_data, self._properties)
        part_names = initialize_part_names(env)

        # Get and set the offsets of the markers
        offsets = np.copy(env.physics.bind(env.task._walker.body_sites).pos[:])
        offsets *= self.SCALE_FACTOR
        env.physics.bind(env.task._walker.body_sites).pos[:] = offsets
        mjlib.mj_kinematics(env.physics.model.ptr, env.physics.data.ptr)
        mjlib.mj_comPos(env.physics.model.ptr, env.physics.data.ptr)
        for n_site, p in enumerate(env.physics.bind(env.task._walker.body_sites).pos):
            env.task._walker.body_sites[n_site].pos = p

        # Optimize the pose and offsets for the first frame
        initial_optimization(env, offsets, self._properties)

        # Optimize the pose for the whole sequence
        q, walker_body_sites, x = pose_optimization(env, self._properties)

        # Optimize the offsets
        offset_optimization(env, offsets, q, self._properties)

        # Optimize the pose for the whole sequence
        q, walker_body_sites, x = pose_optimization(env, self._properties)
        self.data = package_data(
            env, q, x, walker_body_sites, part_names, kp_data, self._properties
        )
        return self

    def transform(self, kp_data: np.ndarray, offset_path: Text) -> Dict:
        """Register skeleton to keypoint data

        Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

        Example:
            data = stac.transform(keypoints, offset_path)

        Args:
            keypoints (np.ndarray): Keypoint data in meters (n_frames, 3, n_keypoints).
            offset_path (Text): Path to offset file saved after .fit()

        Returns:
            Dict: Registered data dictionary
        """
        kp_data = self._prepare_data(kp_data)
        self.n_frames = kp_data.shape[0]
        self.offset_path = offset_path
        env = build_env(kp_data, self._properties)
        part_names = initialize_part_names(env)

        # If preloading offsets, set them now.
        with open(self.offset_path, "rb") as f:
            in_dict = pickle.load(f)
        sites = env.task._walker.body_sites
        env.physics.bind(sites).pos[:] = in_dict["offsets"]
        for n_site, p in enumerate(env.physics.bind(sites).pos):
            sites[n_site].pos = p

        # Optimize the root position
        root_optimization(env, self._properties)

        # Optimize the pose for the whole sequence
        q, walker_body_sites, x = pose_optimization(env, self._properties)

        # Extract pose, offsets, data, and all parameters
        self.data = package_data(
            env, q, x, walker_body_sites, part_names, kp_data, self._properties
        )
        return self.data

    def save(self, save_path: Text):
        """Save data.

        Args:
            save_path (Text): Path to save data. Defaults to None.
        """
        if os.path.dirname(save_path) != "":
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _, file_extension = os.path.splitext(save_path)
        if file_extension == ".p":
            with open(save_path, "wb") as output_file:
                pickle.dump(self.data, output_file, protocol=2)
        elif file_extension == ".mat":
            savemat(save_path, self.data)
