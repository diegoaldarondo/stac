"""Implementation of stac for animal motion capture in dm_control suite."""
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np
import scipy.optimize


class _TestNoneArgs(BaseException):
    """Simple base exception"""

    pass


def q_loss(
    q,
    physics,
    kp_data,
    sites,
    params,
    qs_to_opt=None,
    q_copy=None,
    reg_coef=0.0,
    root_only=False,
    temporal_regularization=False,
    q_prev=None,
    q_next=None,
    kps_to_opt=None,
) -> float:
    """Compute the marker loss for q_phase optimization.

    Args:
        q (TYPE): Qpos for current frame.
        physics (TYPE): Physics of current environment.
        kp_data (TYPE): Reference keypoint data.
        sites (TYPE): sites of keypoints at frame_index
        params (TYPE): Animal parameters dictionary
        qs_to_opt (None, optional): Binary vector of qposes to optimize.
        q_copy (None, optional): Copy of current qpos, for use in optimization of subsets
        reg_coef (float, optional): L1 regularization coefficient during marker loss.
        root_only (bool, optional): If True, only regularize the root.
        temporal_regularization (bool, optional): If True, regularize joints over time.
        q_prev (None, optional): Copy of previous qpos frame
        q_next (None, optional): Copy of next qpos frame
        kps_to_opt (None, optional): Vector denoting which keypoints to use in loss.

    Returns:
        float: loss value
    """
    if temporal_regularization:
        error_msg = " cannot be None if using temporal regularization"
        if qs_to_opt is None:
            raise _TestNoneArgs("qs_to_opt" + error_msg)
        if q_prev is None:
            raise _TestNoneArgs("q_prev" + error_msg)
        # if q_next is None:
        #     raise _TestNoneArgs('q_next' + error_msg)

    # If optimizing arbitrary sets of qpos, add the optimizer qpos to the copy.
    if qs_to_opt is not None:
        q_copy[qs_to_opt] = q.copy()
        q = np.copy(q_copy)

    # Add temporal regularization for arms.
    temp_reg_term = 0.0
    if temporal_regularization:
        temp_reg_term += (q[qs_to_opt] - q_prev[qs_to_opt]) ** 2
        if q_next is not None:
            temp_reg_term += (q[qs_to_opt] - q_next[qs_to_opt]) ** 2

    residual = kp_data.T - q_joints_to_markers(q, physics, sites)
    if kps_to_opt is not None:
        residual = residual[kps_to_opt]

    loss = residual
    return loss


def q_joints_to_markers(q, physics, sites):
    """Convert site information to marker information.

    :param q: Postural state
    :param physics: Physics of current environment
    :param sites: Sites of keypoint data.

    Args:
        q (TYPE): Description
        physics (TYPE): Description
        sites (TYPE): Description

    Returns:
        TYPE: Description
    """
    physics.named.data.qpos[:] = q.copy()

    # Forward kinematics
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

    # Center of mass position
    mjlib.mj_comPos(physics.model.ptr, physics.data.ptr)

    return_value = np.array(physics.bind(sites).xpos)
    return return_value.flatten()


def q_phase(
    physics,
    marker_ref_arr,
    sites,
    params,
    reg_coef=0.0,
    qs_to_opt=None,
    kps_to_opt=None,
    root_only=False,
    trunk_only=False,
    temporal_regularization=False,
    q_prev=None,
    q_next=None,
    ftol=None,
):
    """Update q_pose using estimated marker parameters.

    :param physics: Physics of current environment.
    :param marker_ref_arr: Keypoint data reference
    :param sites: sites of keypoints at frame_index
    :param params: Animal parameters dictionary
    :param reg_coef: L1 regularization coefficient during marker loss.
    :param qs_to_opt: Binary vector of qs to optimize.
    :param kps_to_opt: Logical vector of keypoints to use in q loss function.
    :param root_only: If True, only optimize the root.
    :param temporal_regularization: If True, regularize arm joints over time.

    Args:
        physics (TYPE): Description
        marker_ref_arr (TYPE): Description
        sites (TYPE): Description
        params (TYPE): Description
        reg_coef (float, optional): Description
        qs_to_opt (None, optional): Description
        kps_to_opt (None, optional): Description
        root_only (bool, optional): Description
        trunk_only (bool, optional): Description
        temporal_regularization (bool, optional): Description
        q_prev (None, optional): Description
        q_next (None, optional): Description
        ftol (None, optional): Description
    """
    lb = np.concatenate([-np.inf * np.ones(7), physics.named.model.jnt_range[1:][:, 0]])
    lb = np.minimum(lb, 0.0)
    ub = np.concatenate([np.inf * np.ones(7), physics.named.model.jnt_range[1:][:, 1]])
    # Define initial position of the optimization
    q0 = np.copy(physics.named.data.qpos[:])
    q_copy = np.copy(q0)

    # Set the center to help with finding the optima
    # TODO(centering_bug):
    # The center is not necessarily from 12:15 depending on struct ordering.
    # This probably won't be a problem, as it is just an ititialization for the
    # optimizer, but keep it in mind.
    if root_only or trunk_only:
        q0[:3] = marker_ref_arr[12:15]
        diff_step = params["_ROOT_DIFF_STEP"]
    else:
        diff_step = params["_DIFF_STEP"]
    if root_only:
        qs_to_opt = np.zeros_like(q0, dtype=np.bool)
        qs_to_opt[:7] = True

    # If you only want to optimize a subset of qposes,
    # limit the optimizer to that
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
                marker_ref_arr,
                sites,
                params,
                qs_to_opt=qs_to_opt,
                q_copy=q_copy,
                reg_coef=reg_coef,
                root_only=root_only,
                temporal_regularization=temporal_regularization,
                q_prev=q_prev,
                q_next=q_next,
                kps_to_opt=kps_to_opt,
            ),
            q0,
            bounds=(lb, ub),
            ftol=ftol,
            diff_step=diff_step,
            # loss='soft_l1',
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
    offset,
    physics,
    kp_data,
    time_indices,
    sites,
    q,
    initial_offsets,
    is_regularized=None,
    reg_coef=0.0,
):
    """Compute the marker loss for optimization.

    :param offset: vector of offsets from rat bodies to inferred
                   mocap sites
    :param physics: Physics of current environment.
    :param kp_data: Mocap data in global coordinates
    :param time_indices: time_indices used for offset estimation
    :param sites: sites of keypoints at frame_index
    :param q: qpos values for the frames in time_indices
    :param initial_offsets: Initial offset values for offset regularization
    :param is_regularized: binary vector of offsets to regularize.
    :param reg_coef: L1 regularization coefficient during marker loss.

    Args:
        offset (TYPE): Description
        physics (TYPE): Description
        kp_data (TYPE): Description
        time_indices (TYPE): Description
        sites (TYPE): Description
        q (TYPE): Description
        initial_offsets (TYPE): Description
        is_regularized (None, optional): Description
        reg_coef (float, optional): Description
    """
    residual = 0
    reg_term = 0
    for i, frame in enumerate(time_indices):
        physics.named.data.qpos[:] = q[frame].copy()

        # Get the offset relative to the initial position, only for
        # markers you wish to regularize
        reg_term += ((offset - initial_offsets.flatten()) ** 2) * is_regularized
        residual += (kp_data[i, :].T - m_joints_to_markers(offset, physics, sites)) ** 2
    return np.sum(residual) + reg_coef * np.sum(reg_term)


def m_joints_to_markers(offset, physics, sites):
    """Convert site information to marker information.

    :param offset: Postural state
    :param physics: Physics of current environment
    :param sites: Sites of keypoint data.

    Args:
        offset (TYPE): Description
        physics (TYPE): Description
        sites (TYPE): Description

    Returns:
        TYPE: Description
    """
    physics.bind(sites).pos[:] = np.reshape(offset.copy(), (-1, 3))

    # Forward kinematics
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

    # Center of mass position
    mjlib.mj_comPos(physics.model.ptr, physics.data.ptr)

    return_value = np.array(physics.bind(sites).xpos)
    return return_value.flatten()


def m_phase(
    physics,
    kp_data,
    sites,
    time_indices,
    q,
    initial_offsets,
    params,
    reg_coef=0.0,
    maxiter=50,
):
    """Estimate marker offset, keeping qpos fixed.

    Args:
        physics (TYPE): Physics of current environment
        kp_data (TYPE): Keypoint data.
        sites (TYPE): sites of keypoints at frame_index.
        time_indices (TYPE): time_indices used for offset estimation.
        q (TYPE): qpos values for the frames in time_indices.
        initial_offsets (TYPE): Initial offset values for offset regularization.
        params (TYPE): Animal parameters dictionary
        reg_coef (float, optional): L1 regularization coefficient during marker loss.
        maxiter (int, optional): Maximum number of iterations to use in the minimization.
    """
    # Define initial position of the optimization
    offset0 = np.copy(physics.bind(sites).pos[:]).flatten()

    # Build a matrix of ones and zeros denoting whether that component of
    # offsets will be regularized or not.
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
            kp_data[time_indices, :],
            time_indices,
            sites,
            q,
            initial_offsets,
            is_regularized=is_regularized,
            reg_coef=reg_coef,
        ),
        offset0,
        # tol=params['_ROOT_FTOL'],
        options={"maxiter": maxiter},
    )

    # Set pose to the optimized m and step forward.
    physics.bind(sites).pos[:] = np.reshape(offset_opt_param.x, (-1, 3))

    # Forward kinematics, and save the results to the walker sites as well
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
    for n_site, p in enumerate(physics.bind(sites).pos):
        sites[n_site].pos = p
