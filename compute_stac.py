"""Compute stac optimization on data."""
from dm_control import viewer
from scipy.io import savemat
import scipy.ndimage
import clize
import stac
import rodent_environments
import numpy as np
import util
import pickle
import os
import tasks
_MM_TO_METERS = 1000


def _downsample(kp_data, params, orig_freq=300.):
    n_samples = kp_data.shape[0]
    n_upsamples = \
        int(np.round(n_samples / orig_freq * (1 / params['_TIME_BINS'])))
    interp_x_vals = np.linspace(0, n_samples, n_upsamples)
    kp_data_downsampled = np.zeros((interp_x_vals.size, kp_data.shape[1]))

    # Linearly interpolate to downsample for each marker
    for marker in range(kp_data.shape[1]):
        kp_data_downsampled[:, marker] = \
            np.interp(interp_x_vals, range(n_samples), kp_data[:, marker])
    return kp_data_downsampled


def _smooth(kp_data, kp_names, sigma=1):
    parts_to_smooth = ['Arm', 'Elbow']
    ids = np.argwhere([any(part in name for name in kp_names)
                       for part in parts_to_smooth])
    for id in ids:
        kp_data[:, id] = scipy.ndimage.gaussian_filter1d(kp_data[:, id],
                                                         sigma, axis=0)
    return kp_data


def preprocess_snippet(kp_data, kp_names, params):
    """Preprocess snippet data."""
    kp_data = kp_data / _MM_TO_METERS
    spineM_id = np.argwhere([name == 'SpineM' for name in kp_names])
    spineM = np.squeeze(kp_data[:, spineM_id * 3 + np.array([0, 1, 2])])

    # Rescale by centering at spineM, scaling, and decentering
    for dim in range(np.round(kp_data.shape[1] / 3).astype('int32')):
        kp_data[:, dim * 3 + np.array([0, 1, 2])] -= spineM
    kp_data *= params['scale_factor']
    spineM *= params['scale_factor']
    for dim in range(np.round(kp_data.shape[1] / 3).astype('int32')):
        kp_data[:, dim * 3 + np.array([0, 1, 2])] += spineM

    # Downsample
    kp_data = _downsample(kp_data, params)

    # Smooth
    kp_data = _smooth(kp_data, kp_names, sigma=.1)

    # Handle z-offset conditions
    if params['adaptive_z_offset']:
        kp_data[:, 2::3] -= \
            np.nanpercentile(kp_data[:, 2::3].flatten(), params['z_perc'])
        kp_data[:, 2::3] += params['adaptive_z_offset_value']
    else:
        kp_data[:, 2::3] -= params['z_offset']
    return kp_data


def preprocess_data(data_path, start_frame, end_frame, skip, params,
                    struct_name='markers_preproc',):
    """Preprocess mocap data for stac fitting.

    :param data_path: Path to .mat mocap file
    :param start_frame: Frame to start stac tracing
    :param skip: Subsampling rate for the frames
    :param scale_factor: Multiplier for mocap data
    :param struct_name: Field name of .mat file to load
    """
    kp_data, kp_names = util.load_kp_data_from_file(data_path,
                                                    struct_name=struct_name)
    kp_data = kp_data[::skip, :]
    kp_data = preprocess_snippet(
        kp_data[start_frame:end_frame, :], kp_names, params)
    return kp_data, kp_names


def initial_optimization(env, initial_offsets, params, maxiter=100):
    """Optimize the first frame with alternating q and m phase."""
    stac.q_phase(env.physics, env.task.kp_data[0, :],
                 env.task._walker.body_sites, params, root_only=True)
    root_q = [env.physics.named.data.qpos[:].copy()]
    stac.m_phase(env.physics, env.task.kp_data, env.task._walker.body_sites,
                 [0], root_q, initial_offsets, params,
                 reg_coef=params['m_reg_coef'], maxiter=maxiter)


def root_optimization(env, params):
    """Optimize only the root."""
    stac.q_phase(env.physics, env.task.kp_data[0, :],
                 env.task._walker.body_sites, params, root_only=True)


def q_clip(env, qs_to_opt, params):
    """Q-phase across the clip: optimize joint angles."""
    q = []
    walker_body_sites = []
    for i in range(params['n_frames']):
        print(i)
        stac.q_phase(env.physics, env.task.kp_data[i, :],
                     env.task._walker.body_sites, params,
                     reg_coef=params['q_reg_coef'])
        if i == 0:
            temp_reg = False
        else:
            temp_reg = True
        stac.q_phase(env.physics, env.task.kp_data[i, :],
                     env.task._walker.body_sites, params,
                     reg_coef=params['q_reg_coef'],
                     qs_to_opt=qs_to_opt, temporal_regularization=temp_reg)
        q.append(np.copy(env.physics.named.data.qpos[:]))
        walker_body_sites.append(
            np.copy(env.physics.bind(env.task._walker.body_sites).xpos[:])
        )
    return q, walker_body_sites


def _get_part_ids(env, parts):
    part_names = env.physics.named.data.qpos.axes.row.names
    return np.array([any(part in name for part in parts)
                     for name in part_names])


def q_clip_iso(env, params):
    """Perform q_phase over the entire clip.

    Optimizes limbs and head independently.
    Perform bidirectional temporal regularization.
    :param env: Rodent environment.
    :param params: Rodent parameters.
    """
    q = []
    x = []
    walker_body_sites = []
    r_leg = _get_part_ids(env, ['hip_R', 'knee_R'])
    l_leg = _get_part_ids(env, ['hip_L', 'knee_L'])
    r_arm = _get_part_ids(env, ['scapula_R', 'shoulder_R', 'elbow_R'])
    l_arm = _get_part_ids(env, ['scapula_L', 'shoulder_L', 'elbow_L'])
    head = _get_part_ids(env, ['atlas', 'cervical', 'atlant_extend', ])
    if params['LIMBS_TO_TEMPORALLY_REGULARIZE'] == "arms":
        temp_reg_indiv_parts = [r_arm, l_arm]
        non_temp_reg_indiv_parts = [r_leg, l_leg, head]
    elif params['LIMBS_TO_TEMPORALLY_REGULARIZE'] == "arms and legs":
        temp_reg_indiv_parts = [r_leg, l_leg, r_arm, l_arm]
        non_temp_reg_indiv_parts = [head]

    # Iterate through all of the frames in the clip
    for i in range(params['n_frames']):
        print(i)
        # First optimize over all points to get gross estimate and trunk
        stac.q_phase(env.physics, env.task.kp_data[i, :],
                     env.task._walker.body_sites, params,
                     reg_coef=params['q_reg_coef'])

        # Make sure to only use forward temporal regularization on frames 1...n
        if i == 0:
            temp_reg = False
            q_prev = 0
        else:
            temp_reg = True
            q_prev = q[i - 1]

        # Next optimize over the limbs individually to improve time and accur.
        for part in non_temp_reg_indiv_parts:
            stac.q_phase(env.physics, env.task.kp_data[i, :],
                         env.task._walker.body_sites, params,
                         reg_coef=params['q_reg_coef'],
                         qs_to_opt=part)
        for part in temp_reg_indiv_parts:
            stac.q_phase(env.physics, env.task.kp_data[i, :],
                         env.task._walker.body_sites, params,
                         reg_coef=params['q_reg_coef'],
                         qs_to_opt=part,
                         q_prev=q_prev,
                         temporal_regularization=temp_reg)
        q.append(np.copy(env.physics.named.data.qpos[:]))
        x.append(np.copy(env.physics.named.data.xpos[:]))
        walker_body_sites.append(
            np.copy(env.physics.bind(env.task._walker.body_sites).xpos[:])
        )

    # Bidirectional temporal regularization
    for i in range(1, params['n_frames'] - 1):
        # Set model state to current frame
        env.physics.named.data.qpos[:] = q[i]

        # Recompute position of select parts with bidirectional
        # temporal regularizer.
        for part in [r_arm, l_arm, r_leg, l_leg]:
            stac.q_phase(env.physics, env.task.kp_data[i, :],
                         env.task._walker.body_sites, params,
                         reg_coef=params['q_reg_coef'],
                         qs_to_opt=part, temporal_regularization=True,
                         q_prev=q[i - 1],
                         q_next=q[i + 1])

            # Update the parts for the current frame
            q[i][part] = np.copy(env.physics.named.data.qpos[:][part])
            x[i] = np.copy(env.physics.named.data.xpos[:])
        walker_body_sites[i] = \
            np.copy(env.physics.bind(env.task._walker.body_sites).xpos[:])
    return q, walker_body_sites, x


def qpos_z_offset(env, q, x):
    """Add a z offset to the qpos root equal to the minimum of the hands/ankles.

    :param env: Rat mocap environment.
    :param q: List of qposes over a clip.
    :param x: List of xposes over a clip.
    """
    # Find the indices of hands and feet xpositions
    ground_parts = ['foot', 'hand']
    part_names = env.physics.named.data.xpos.axes.row.names
    ground_ids = [any(part in name for part in ground_parts)
                  for name in part_names]

    # Estimate the ground position by the 2nd percentile of the hands/feet
    ground_part_pos = np.zeros((len(x), np.sum(ground_ids)))

    for i, xpos in enumerate(x):
        ground_part_pos[i, :] = xpos[ground_ids, 2]

    # Set the minimum position over the clip to be the ground_pos
    ground_pos = np.min(np.nanpercentile(ground_part_pos, .05, axis=0))
    # ground_pos = ground_pos - .013
    for i, qpos in enumerate(q):
        qpos[2] -= ground_pos
        q[i] = qpos
    return q, ground_pos


def compute_stac(kp_data, save_path, params):
    """Perform stac on rat mocap data.

    :param kp_data: mocap_data
    :param save_path: File to save optimized qposes
    :param params: Dictionary of rat parameters
    """
    if params['n_frames'] is None:
        params['n_frames'] = kp_data.shape[0]
    params['n_frames'] = int(params['n_frames'])
    # Build the environment
    env = rodent_environments.rodent_mocap(
        kp_data, params, use_hfield=params['_USE_HFIELD'])

    # Get the ids of the limbs
    # TODO(partnames): This currently changes the list everywhere.
    # Technically this is what we always want, but consider changing.
    part_names = env.physics.named.data.qpos.axes.row.names
    for i in range(6):
        part_names.insert(0, part_names[0])

    # If preloading offsets, set them now.
    if params['offset_path'] is not None:
        with open(params['offset_path'], 'rb') as f:
            in_dict = pickle.load(f)

        sites = env.task._walker.body_sites
        env.physics.bind(sites).pos[:] = in_dict['offsets']

        for id, p in enumerate(env.physics.bind(sites).pos):
            sites[id].pos = p

        if params['verbose']:
            print('Root Optimization', flush=True)
        root_optimization(env, params)
    else:
        # Get the initial offsets of the markers
        initial_offsets = \
            np.copy(env.physics.bind(env.task._walker.body_sites).pos[:])

        # First optimize the first frame to get an approximation
        # of the m and q phases
        if params['verbose']:
            print('Initial Optimization', flush=True)
        initial_optimization(env, initial_offsets, params)

        # Find the frames to use in the m-phase optimization.
        time_indices = np.random.randint(0, params['n_frames'],
                                         params['n_sample_frames'])

    # Q_phase optimization
    if params['verbose']:
        print('q-phase', flush=True)
    # q, walker_body_sites = q_clip(env, limbs, params)
    q, walker_body_sites, x = q_clip_iso(env, params)

    # If you've precomputed the offsets, stop here.
    # Otherwise do another m and q phase.
    if params['offset_path'] is None:
        # M-phase across the subsampling: optimize offsets
        if params['verbose']:
            print('m-phase', flush=True)
        stac.m_phase(env.physics, env.task.kp_data,
                     env.task._walker.body_sites, time_indices, q,
                     initial_offsets, params, reg_coef=params['m_reg_coef'])
        if params['verbose']:
            print('q-phase', flush=True)
        q, walker_body_sites, x = q_clip_iso(env, params)

    # Fix z offsets using the model positions
    # q, ground_pos = qpos_z_offset(env, q, x)
    # kp_data[:, 2::3] -= ground_pos

    # Optional visualization
    if params['visualize']:
        env.task.precomp_qpos = q
        env.task.render_video = params['render_video']
        viewer.launch(env)
        if env.task.V is not None:
            env.task.V.release()

    # Save the pose, offsets, data, and all parameters
    filename, file_extension = os.path.splitext(save_path)
    offsets = env.physics.bind(env.task._walker.body_sites).pos[:].copy()
    out_dict = {'qpos': q,
                'walker_body_sites': walker_body_sites,
                'offsets': offsets,
                # 'ground_pos': ground_pos,
                'kp_data': np.copy(kp_data[:params['n_frames'], :])}

    if params['_USE_HFIELD'] and isinstance(
            env.task, tasks.ViewMocap_Hfield):
        # env.task.get_heightfield(env.physics)
        out_dict['pedestal_radius'] = env.task._arena.pedestal_radius
        out_dict['pedestal_center'] = env.task._arena.pedestal_center
        out_dict['pedestal_height'] = env.task._arena.pedestal_height
        out_dict['hfield_image'] = env.task._arena.hfield
        out_dict['scaled_arena_diameter'] = env.task._arena.arena_diameter

    for k, v in params.items():
        out_dict[k] = v
    if file_extension == '.p':
        with open(save_path, "wb") as output_file:
            pickle.dump(out_dict, output_file, protocol=2)
    elif file_extension == '.mat':
        savemat(save_path, out_dict)


def handle_args(data_path, param_path, *,
                save_path=None,
                offset_path=None,
                start_frame=0,
                end_frame=None,
                n_snip=None,
                n_frames=None,
                n_sample_frames=50,
                skip=5,
                adaptive_z_offset=False,
                verbose=False,
                visualize=False,
                render_video=False,
                process_snippet=True):
    """Wrap compute_stac to perform appropriate processing.

    :param data_path: List of paths to .mat mocap data files
    :param save_path: File to save optimized qposes
    :param offset_path: Path to precomputed marker offsets
    :param start_frame: Frame within mocap file to start optimization
    :param n_frames: Number of frames to optimize
    :param n_sample_frames: Number of frames to use in m-phase optimization
    :param skip: Subsampling rate for the frames
    :param verbose: If True, display messages during optimization.
    :param visualize: If True, launch viewer
    :param render_video: If True, render_video
    :param process_snippet: If True, process snippet,
                            otherwise assume mocap struct.
    """
    # Aggregate optional cl arguments into params dict
    kw = {"data_path": data_path,
          "offset_path": offset_path,
          "start_frame": start_frame,
          "n_frames": n_frames,
          "n_sample_frames": n_sample_frames,
          "verbose": verbose,
          "skip": skip,
          "adaptive_z_offset": adaptive_z_offset,
          "visualize": visualize,
          "render_video": render_video}
    params = util.load_params(param_path)
    for key, v in kw.items():
        params[key] = v

    if process_snippet:
        data, kp_names, behavior, com_vel = \
            util.load_snippets_from_file(data_path)

        # Put useful statistics into params dict for now. Consider stats dict
        params['behavior'] = behavior
        params['com_vel'] = com_vel

        kp_data = \
            preprocess_snippet(data, kp_names, params)

        # Save the file as a pickle with the same name as the input file in the
        # folder specified by save_path
        file, ext = os.path.splitext(os.path.basename(data_path))
        save_path = os.path.join(save_path, file + '.p')
        print(save_path, flush=True)
        compute_stac(kp_data, save_path, params)
        print('Finished %s' % (save_path))

    # Support file-based processing
    else:
        if end_frame is None:
            end_frame = start_frame + 500
        kp_data, kp_names = \
            preprocess_data(data_path, start_frame, end_frame, skip, params)
        if save_path is None:
            save_path = os.path.join(os.getcwd(),
                                     'results',
                                     'snippet' + data_path[:-4] + '.p')
        compute_stac(kp_data, save_path, params)


if __name__ == '__main__':
    clize.run(handle_args)
