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

_MM_TO_METERS = 1000


def preprocess_data(
    data_path: Text,
    start_frame: int,
    end_frame: int,
    skip: int,
    params: Dict,
) -> Tuple[np.ndarray, List]:
    """Preprocess mocap data for stac fitting.

    Args:
        data_path (Text): Path to .mat mocap file
        start_frame (int): Frame to start stac tracking
        end_frame (int): Frame to end stac tracking
        skip (int): Subsampling rate for the frames
        params (Dict): Parameters dictionary
        struct_name (Text, optional): Field name of .mat file to load. DEPRECATED

    Returns:
        Tuple: kp_data (np.ndarray): Keypoint data
               kp_names (List): List of keypoint names
    """
    kp_data, kp_names = util.load_dannce_data(
        data_path,
        params["skeleton_path"],
        start_frame=start_frame,
        end_frame=end_frame,
    )
    kp_data = kp_data[::skip, :]
    kp_data = kp_data / _MM_TO_METERS
    kp_data[:, 2::3] -= params["Z_OFFSET"]
    return kp_data, kp_names


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
    time_indices = np.random.randint(0, params["n_frames"], params["n_sample_frames"])
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
    Perform bidirectional temporal regularization.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary.

    Returns:
        Tuple: qpos, walker body sites, xpos
    """
    q = []
    x = []
    walker_body_sites = []

    r_leg = get_part_ids(
        env,
        [
            "vertebra_1",
            "vertebra_2",
            "vertebra_3",
            "vertebra_4",
            "vertebra_5",
            "vertebra_6",
            "hip_R",
            "knee_R",
            "ankle_R",
            "foot_R",
        ],
    )
    l_leg = get_part_ids(
        env,
        [
            "vertebra_1",
            "vertebra_2",
            "vertebra_3",
            "vertebra_4",
            "vertebra_5",
            "vertebra_6",
            "hip_L",
            "knee_L",
            "ankle_L",
            "foot_L",
        ],
    )
    r_arm = get_part_ids(
        env,
        [
            "scapula_R",
            "shoulder_R",
            "shoulder_s",
            "elbow_R",
            "hand_R",
            "finger_R",
        ],
    )
    l_arm = get_part_ids(
        env,
        [
            "scapula_L",
            "shoulder_L",
            "shoulder_s",
            "elbow_L",
            "hand_L",
            "finger_L",
        ],
    )
    head = get_part_ids(env, ["atlas", "cervical", "atlant_extend"])
    indiv_parts = [head, r_leg, l_leg, r_arm, l_arm]

    # Iterate through all of the frames in the clip
    for i in range(params["n_frames"]):
        # Optimize over all points
        stac_base.q_phase(
            env.physics,
            env.task.kp_data[i, :],
            env.task._walker.body_sites,
            params,
        )

        # Next optimize over the limbs individually to improve time and accur.
        for part in indiv_parts:
            stac_base.q_phase(
                env.physics,
                env.task.kp_data[i, :],
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
        "kp_data": np.copy(kp_data[: params["n_frames"], :]),
    }
    for k, v in params.items():
        data[k] = v
    return data


class STAC:
    def __init__(
        self,
        data_path: Text,
        param_path: Text,
        start_frame: int = 0,
        end_frame: int = 0,
        n_frames: int = None,
        n_sample_frames: int = 50,
        skip: int = 1,
        skeleton_path: Text = "/n/holylfs02/LABS/olveczky_lab/Diego/code/Label3D/skeletons/rat23.mat",
    ):
        """Initialize STAC

        Args:
            data_path (Text): Path to dannce .mat file
            param_path (Text): Path to parameters .yaml file.
            save_path (Text, optional): Path to save data. Defaults to None.
            start_frame (int, optional): Starting frame. Defaults to 0.
            end_frame (int, optional): Ending frame. Defaults to 0.
            n_frames (int, optional): Number of frames to evaluate. Defaults to None.
            n_sample_frames (int, optional): Number of frames to evaluate for m-phase estimation. Defaults to 50.
            skip (int, optional): Frame skip. Defaults to 1.
            verbose (bool, optional): If True, print status messages. Defaults to False.
            skeleton_path (Text, optional): Path to skeleton file. Defaults to "/n/holylfs02/LABS/olveczky_lab/Diego/code/Label3D/skeletons/rat23.mat".
        """
        # Aggregate optional cl arguments into params dict
        self._properties = util.load_params(param_path)
        kw = {
            "data_path": data_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": n_frames,
            "n_sample_frames": n_sample_frames,
            "skip": skip,
            "skeleton_path": skeleton_path,
            "kp_names": None,
            "data": None,
        }
        for key, v in kw.items():
            self._properties[key] = v

        # Create properties dynamically
        for property_name in self._properties.keys():

            def getter(self, name=property_name):
                return self._properties[name]

            def setter(self, value, name=property_name):
                self._properties[name] = value

            setattr(STAC, property_name, property(fget=getter, fset=setter))

    def fit(self) -> Dict:
        """Calibrate and fit the model to keypoints.

        Performs three rounds of alternating marker and quaternion optimization. Optimal
        results with greater than 200 frames of data in which the subject is moving.

        Example:
            stac = STAC(data_path, param_path, **kwargs)
            offset_data = stac.fit()
            stac.save(offset_data, offset_save_path)

        Returns:
            Dict: Data dictionary
        """
        kp_data = self._prepare_data()
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

    def _prepare_data(self) -> np.ndarray:
        """Preprocess data and keypoint names.

        Returns:
            np.ndarray: Keypoint data (nSamples, nKeypoints*3)
        """
        kp_data, kp_names = preprocess_data(
            self.data_path,
            self.start_frame,
            self.end_frame,
            self.skip,
            self._properties,
        )
        self.kp_names = kp_names
        if self.n_frames is None:
            self.n_frames = kp_data.shape[0]
        self.n_frames = int(self.n_frames)
        return kp_data

    def transform(self, offset_path: Text) -> Dict:
        """Register skeleton to keypoint data

        Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

        Example:
            stac = STAC(data_path, param_path, **kwargs)
            offset_data = stac.fit()
            stac.save(offset_data, offset_save_path)
            data = stac.transform(offset_save_path)
            stac.save(data, save_path)

        Args:
            offset_path (Text): Path to offset file saved after .fit()

        Returns:
            Dict: Registered data dictionary
        """
        kp_data = self._prepare_data()
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
