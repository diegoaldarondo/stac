"""Implementation of stac for animal motion capture in dm_control suite."""
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np
import scipy.optimize
from typing import List, Dict, Text, Union, Tuple


class _TestNoneArgs(BaseException):
    """Simple base exception"""

    pass


def q_loss(
    q: np.ndarray,
    physics,
    kp_data: np.ndarray,
    sites: np.ndarray,
    qs_to_opt: np.ndarray = None,
    q_copy: np.ndarray = None,
    kps_to_opt: np.ndarray = None,
) -> float:
    """Compute the marker loss for q_phase optimization.

    Args:
        q (np.ndarray): Qpos for current frame.
        physics (TYPE): Physics of current environment.
        kp_data (np.ndarray): Reference keypoint data.
        sites (np.ndarray): sites of keypoints at frame_index
        qs_to_opt (List, optional): Binary vector of qposes to optimize.
        q_copy (np.ndarray, optional): Copy of current qpos, for use in optimization of subsets
        kps_to_opt (List, optional): Vector denoting which keypoints to use in loss.

    Returns:
        float: loss value
    """
    # If optimizing arbitrary sets of qpos, add the optimizer qpos to the copy.
    if qs_to_opt is not None:
        q_copy[qs_to_opt] = q.copy()
        q = np.copy(q_copy)

    residual = kp_data - q_joints_to_markers(q, physics, sites)
    if kps_to_opt is not None:
        residual = residual[kps_to_opt]
    return residual


def q_joints_to_markers(q: np.ndarray, physics, sites: np.ndarray) -> np.ndarray:
    """Convert site information to marker information.

    Args:
        q (np.ndarray): Postural state
        physics (TYPE): Physics of current environment
        sites (np.ndarray): Sites of keypoint data.

    Returns:
        np.ndarray: Array of marker positions.
    """
    physics.named.data.qpos[:] = q.copy()

    # Forward kinematics
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

    # Center of mass position
    mjlib.mj_comPos(physics.model.ptr, physics.data.ptr)

    return np.array(physics.bind(sites).xpos).flatten()


# TODO: Refactor
def q_phase(
    physics,
    marker_ref_arr: np.ndarray,
    sites: np.ndarray,
    params: Dict,
    qs_to_opt: np.ndarray = None,
    kps_to_opt: np.ndarray = None,
    root_only: bool = False,
    trunk_only: bool = False,
    ftol: float = None,
):
    """Update q_pose using estimated marker parameters.

    Args:
        physics (TYPE): Physics of current environment.
        marker_ref_arr (np.ndarray): Keypoint data reference
        sites (np.ndarray): sites of keypoints at frame_index
        params (Dict): Animal parameters dictionary
        qs_to_opt (np.ndarray, optional): Description
        kps_to_opt (np.ndarray, optional): Description
        root_only (bool, optional): Description
        trunk_only (bool, optional): Description
        temporal_regularization (bool, optional): If True, regularize arm joints over time.
        ftol (float, optional): Description
        qs_to_opt (None, optional Binary vector of qs to optimize.
        kps_to_opt (None, optional Logical vector of keypoints to use in q loss function.
        root_only (bool, optional If True, only optimize the root.
        trunk_only (bool, optional If True, only optimize the trunk.
    """
    lb = np.concatenate([-np.inf * np.ones(7), physics.named.model.jnt_range[1:][:, 0]])
    lb = np.minimum(lb, 0.0)
    ub = np.concatenate([np.inf * np.ones(7), physics.named.model.jnt_range[1:][:, 1]])
    # Define initial position of the optimization
    q0 = np.copy(physics.named.data.qpos[:])
    q_copy = np.copy(q0)

    # Set the center to help with finding the optima (does not need to be exact)
    if root_only or trunk_only:
        q0[:3] = marker_ref_arr[12:15]
        diff_step = params["_ROOT_DIFF_STEP"]
    else:
        diff_step = params["_DIFF_STEP"]
    if root_only:
        qs_to_opt = np.zeros_like(q0, dtype=np.bool)
        qs_to_opt[:7] = True

    # Limit the optimizer to a subset of qpos
    if qs_to_opt is not None:
        q0 = q0[qs_to_opt]
        lb = lb[qs_to_opt]
        ub = ub[qs_to_opt]

    # Use different tolerances for root vs normal optimization
    if ftol is None:
        if root_only:
            ftol = params["_ROOT_FTOL"]
        elif qs_to_opt is not None:
            ftol = params["_LIMB_FTOL"]
        else:
            ftol = params["_FTOL"]
    try:
        q_opt_param = scipy.optimize.least_squares(
            lambda q: q_loss(
                q,
                physics,
                marker_ref_arr.T,
                sites,
                qs_to_opt=qs_to_opt,
                q_copy=q_copy,
                kps_to_opt=kps_to_opt,
            ),
            q0,
            bounds=(lb, ub),
            ftol=ftol,
            diff_step=diff_step,
            verbose=0,
        )

        # Set pose to the optimized q and step forward.
        if qs_to_opt is None:
            physics.named.data.qpos[:] = q_opt_param.x
        else:
            q_copy[qs_to_opt] = q_opt_param.x
            physics.named.data.qpos[:] = q_copy.copy()

        mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

    except ValueError:
        print("Warning: optimization failed.", flush=True)
        q_copy[np.isnan(q_copy)] = 0.0
        physics.named.data.qpos[:] = q_copy.copy()
        mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)


def m_loss(
    offset: np.ndarray,
    physics,
    kp_data: np.ndarray,
    time_indices: List,
    sites: np.ndarray,
    q: np.ndarray,
    initial_offsets: np.ndarray,
    is_regularized: bool = None,
    reg_coef: float = 0.0,
):
    """Compute the marker loss for optimization.

    Args:
        offset (np.ndarray): vector of offsets to inferred mocap sites
        physics (TYPE): Physics of current environment.
        kp_data (np.ndarray): Mocap data in global coordinates
        time_indices (List): time_indices used for offset estimation
        sites (np.ndarray): sites of keypoints at frame_index
        q (np.ndarray): qpos values for the frames in time_indices
        initial_offsets (np.ndarray): Initial offset values for offset regularization
        is_regularized (bool, optional): binary vector of offsets to regularize.
        reg_coef (float, optional): L1 regularization coefficient during marker loss.
    """
    residual = 0
    reg_term = 0
    for i, frame in enumerate(time_indices):
        physics.named.data.qpos[:] = q[frame].copy()

        # Get the offset relative to the initial position, only for
        # markers you wish to regularize
        reg_term += ((offset - initial_offsets.flatten()) ** 2) * is_regularized
        residual += (kp_data[:, i] - m_joints_to_markers(offset, physics, sites)) ** 2
    return np.sum(residual) + reg_coef * np.sum(reg_term)


def m_joints_to_markers(offset, physics, sites) -> np.ndarray:
    """Convert site information to marker information.

    Args:
        offset (TYPE):  Current offset.
        physics (TYPE):  Physics of current environment
        sites (TYPE):  Sites of keypoint data.

    Returns:
        TYPE: Array of marker positions
    """
    physics.bind(sites).pos[:] = np.reshape(offset.copy(), (-1, 3))

    # Forward kinematics
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

    # Center of mass position
    mjlib.mj_comPos(physics.model.ptr, physics.data.ptr)

    return np.array(physics.bind(sites).xpos).flatten()


def m_phase(
    physics,
    kp_data: np.ndarray,
    sites: np.ndarray,
    time_indices: List,
    q: np.ndarray,
    initial_offsets: np.ndarray,
    params: Dict,
    reg_coef: float = 0.0,
    maxiter: int = 50,
):
    """Estimate marker offset, keeping qpos fixed.

    Args:
        physics (TYPE): Physics of current environment
        kp_data (np.ndarray): Keypoint data.
        sites (np.ndarray): sites of keypoints at frame_index.
        time_indices (List): time_indices used for offset estimation.
        q (np.ndarray): qpos values for the frames in time_indices.
        initial_offsets (np.ndarray): Initial offset values for offset regularization.
        params (Dict): Animal parameters dictionary
        reg_coef (float, optional): L1 regularization coefficient during marker loss.
        maxiter (int, optional): Maximum number of iterations to use in the minimization.
    """
    # Define initial position of the optimization
    offset0 = np.copy(physics.bind(sites).pos[:]).flatten()

    # Define which offsets to regularize
    is_regularized = []
    for site in sites:
        if any(n in site.name for n in params["_SITES_TO_REGULARIZE"]):
            is_regularized.append(np.array([1.0, 1.0, 1.0]))
        else:
            is_regularized.append(np.array([0.0, 0.0, 0.0]))
    is_regularized = np.stack(is_regularized).flatten()

    # Optimize dm
    offset_opt_param = scipy.optimize.minimize(
        lambda offset: m_loss(
            offset,
            physics,
            kp_data[time_indices, :].T,
            time_indices,
            sites,
            q,
            initial_offsets,
            is_regularized=is_regularized,
            reg_coef=reg_coef,
        ),
        offset0,
        options={"maxiter": maxiter},
    )

    # Set pose to the optimized m and step forward.
    physics.bind(sites).pos[:] = np.reshape(offset_opt_param.x, (-1, 3))

    # Forward kinematics, and save the results to the walker sites as well
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
    for n_site, p in enumerate(physics.bind(sites).pos):
        sites[n_site].pos = p
