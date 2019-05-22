"""Implementation of stac for animal motion capture in dm_control suite."""
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np
import scipy.optimize


def q_loss(q, physics, kp_data, sites, params, qs_to_opt=None, q_copy=None,
           reg_coef=0., temporal_reg_coef=.2,
           root_only=False, temporal_regularization=False):
    """Compute the marker loss for q_phase optimization.

    :param physics: Physics of current environment.
    :param kp_data: Reference
    :param sites: sites of keypoints at frame_index
    :param qs_to_opt: Binary vector of qposes to optimize.
    :param q_copy: Copy of current qpos, for use in optimization of subsets
                   of qpos.
    :param temporal_reg_coef: Regularization coefficient for temporal reg.
    :param reg_coef: L1 regularization coefficient during marker loss.
    :param root_only: If True, only regularize the root.
    :param temporal_regularization: If True, regularize arm joints over time.
    """
    # Make copy of previous frame for temporal regularization
    # TODO(refactor): Make more readable implementation of qpos copy
    q_prev = q_copy.copy()
    # print('q_copy: ', q_copy.shape)
    temporal_arm_regularizer = 0.

    # Optional regularization.
    reg_term = reg_coef*np.sum(q[7:]**2)

    # If only optimizing the root, set everything else to 0.
    if root_only:
        q[7:] = 0.

    # If optimizing arbitrary sets of qpos, add the optimized qpos to the copy.
    if qs_to_opt is not None:
        q_copy[qs_to_opt] = q
        q = np.copy(q_copy)

    # Add temporal regularization for arms.
    if temporal_regularization:
        part_names = physics.named.data.qpos.axes.row.names
        for id, name in enumerate(part_names):
            if any(part in name for part in params['_ARM_JOINTS']):
                temporal_arm_regularizer += (q[id] - q_prev[id])**2

    residual = (kp_data.T - q_joints_to_markers(q, physics, sites))
    return (.5*np.sum(residual**2) + reg_term
            + temporal_reg_coef*temporal_arm_regularizer)


def q_joints_to_markers(q, physics, sites):
    """Convert site information to marker information.

    :param q: Postural state
    :param physics: Physics of current environment
    :param sites: Sites of keypoint data.
    """
    physics.named.data.qpos[:] = q.copy()

    # Forward kinematics
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

    # Center of mass position
    mjlib.mj_comPos(physics.model.ptr, physics.data.ptr)

    return_value = np.array(physics.bind(sites).xpos)
    return return_value.flatten()


def q_phase(physics, marker_ref_arr, sites, params, reg_coef=0.,
            qs_to_opt=None, root_only=False, temporal_regularization=False):
    """Update q_pose using estimated marker parameters.

    :param physics: Physics of current environment.
    :param marker_ref_arr: Keypoint data reference
    :param sites: sites of keypoints at frame_index
    :param reg_coef: L1 regularization coefficient during marker loss.
    :param qs_to_opt: Binary vector of qs to optimize.
    :param root_only: If True, only optimize the root.
    """
    # Define initial position of the optimization
    q0 = np.copy(physics.named.data.qpos[:])

    q_copy = np.copy(q0)

    # Set the center to help with finding the optima
    if root_only:
        q0[:3] = marker_ref_arr[12:15]

    # If you only want to optimize a subset of qposes,
    # limit the optimizer to that
    if qs_to_opt is not None:
        q0 = q0[qs_to_opt]

    # Use different tolerances for root vs normal optimization
    if root_only:
        ftol = params['_ROOT_FTOL']
    else:
        ftol = params['_FTOL']
    q_opt_param = scipy.optimize.least_squares(
            lambda q: q_loss(q, physics, marker_ref_arr, sites, params,
                             qs_to_opt=qs_to_opt,
                             q_copy=q_copy,
                             reg_coef=reg_coef,
                             root_only=root_only,
                             temporal_regularization=temporal_regularization),
            q0, ftol=ftol, diff_step=params['_DIFF_STEP'],
            verbose=0)

    # Set pose to the optimized q and step forward.
    if qs_to_opt is None:
        physics.named.data.qpos[:] = q_opt_param.x
    else:
        q_copy[qs_to_opt] = q_opt_param.x
        physics.named.data.qpos[:] = q_copy.copy()

    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)


def m_loss(offset, physics, kp_data, time_indices, sites, q, initial_offsets,
           is_regularized=None, reg_coef=0.):
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
    """
    residual = 0
    reg_term = 0
    for i, frame in enumerate(time_indices):
        physics.named.data.qpos[:] = q[frame].copy()

        # Get the offset relative to the initial position, only for
        # markers you wish to regularize
        reg_term += ((offset - initial_offsets.flatten())**2)*is_regularized
        residual += (kp_data[i, :].T
                     - m_joints_to_markers(offset, physics, sites))**2
    return .5*np.sum(residual) + reg_coef*np.sum(reg_term)


def m_joints_to_markers(offset, physics, sites):
    """Convert site information to marker information.

    :param offset: Postural state
    :param physics: Physics of current environment
    :param sites: Sites of keypoint data.
    """
    physics.bind(sites).pos[:] = np.reshape(offset.copy(), (-1, 3))

    # Forward kinematics
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

    # Center of mass position
    mjlib.mj_comPos(physics.model.ptr, physics.data.ptr)

    return_value = np.array(physics.bind(sites).xpos)
    return return_value.flatten()


def m_phase(physics, kp_data, sites, time_indices, q, initial_offsets, params,
            reg_coef=0., maxiter=5):
    """Estimate marker offset, keeping qpose fixed.

    :param physics: Physics of current environment.
    :param kp_data: Keypoint data.
    :param sites: sites of keypoints at frame_index.
    :param q: qpos values for the frames in time_indices.
    :param time_indices: time_indices used for offset estimation.
    :param initial_offsets: Initial offset values for offset regularization.
    :param reg_coef: L1 regularization coefficient during marker loss.
    :param maxiter: Maximum number of iterations to use in the minimization.
    """
    # Define initial position of the optimization
    offset0 = np.copy(physics.bind(sites).pos[:]).flatten()

    # Build a matrix of ones and zeros denoting whether that component of
    # offsets will be regularized or not.
    is_regularized = []
    for site in sites:
        if any(n in site.name for n in params['_SITES_TO_REGULARIZE']):
            is_regularized.append(np.array([1., 1., 1.]))
        else:
            is_regularized.append(np.array([0., 0., 0.]))
    is_regularized = np.stack(is_regularized).flatten()

    # Optimize dm
    offset_opt_param = scipy.optimize.minimize(
            lambda offset: m_loss(offset, physics, kp_data[time_indices, :],
                                  time_indices, sites, q, initial_offsets,
                                  is_regularized=is_regularized,
                                  reg_coef=reg_coef),
            offset0, options={'maxiter': maxiter})

    # Set pose to the optimized m and step forward.
    physics.bind(sites).pos[:] = np.reshape(offset_opt_param.x, (-1, 3))

    # Forward kinematics, and save the results to the walker sites as well
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
    for id, p in enumerate(physics.bind(sites).pos):
        sites[id].pos = p
