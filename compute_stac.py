"""Compute stac optimization on data."""
from dm_control import viewer
from scipy.io import savemat
import clize
import stac
import rodent_environments
import numpy as np
import util
import pickle
import os
_MM_TO_METERS = 1000


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
    kp_data = kp_data[::params['skip'], :]

    # Handle z-offset conditions
    if params['adaptive_z_offset']:
        kp_data[:, 2::3] -= \
            np.nanpercentile(kp_data[:, 2::3].flatten(), params['z_perc'])
        kp_data[:, 2::3] += params['adaptive_z_offset_value']
    else:
        kp_data[:, 2::3] -= params['z_offset']
    return kp_data


def preprocess_data(data_path, start_frame, params,
                    struct_name='markers_preproc'):
    """Preprocess mocap data for stac fitting.

    :param data_path: Path to .mat mocap file
    :param start_frame: Frame to start stac tracing
    :param skip: Subsampling rate for the frames
    :param scale_factor: Multiplier for mocap data
    :param z_offset: Subtractive offset for mocap_data
    :param struct_name: Field name of .mat file to load
    """
    kp_data, kp_names = util.load_kp_data_from_file(data_path,
                                                    struct_name=struct_name)
    kp_data = preprocess_snippet(kp_data[start_frame:], kp_names, params)
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
    env = rodent_environments.rodent_mocap(kp_data, params)

    # Get the ids of the limbs
    # TODO(partnames): This currently changes the list everywhere.
    # Technically this is what we always want, but consider changing.
    part_names = env.physics.named.data.qpos.axes.row.names
    for i in range(6):
        part_names.insert(0, part_names[0])

    limbs = np.array([any(part in name for part in params['_IS_LIMB'])
                      for name in part_names])

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
    q, walker_body_sites = q_clip(env, limbs, params)

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
        q, walker_body_sites = q_clip(env, limbs, params)

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
                'kp_data': np.copy(kp_data[:params['n_frames'], :])}
    for k, v in params.items():
        out_dict[k] = v
    if file_extension == '.p':
        with open(save_path, "wb") as output_file:
            pickle.dump(out_dict, output_file)
    elif file_extension == '.mat':
        savemat(save_path, out_dict)


def handle_args(data_path, param_path, *,
                save_path=None,
                offset_path=None,
                start_frame=0,
                n_snip=None,
                n_frames=None,
                n_sample_frames=50,
                skip=2,
                adaptive_z_offset=True,
                verbose=False,
                visualize=False,
                render_video=False):
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
    """
    # Aggregate optional cl arguments into params dict
    kw = {"offset_path": offset_path,
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

    # Support snippet-based processing
    if n_snip is not None:
        data, kp_names, files, behaviors, com_vels = \
            util.load_snippets_from_file(data_path)

        # Handle seq offset
        n_snip = int(n_snip) - 1
        print(n_snip)

        # Put useful statistics into params dict for now. Consider stats dict
        params['behavior'] = behaviors[n_snip]
        params['com_vel'] = com_vels[n_snip]

        kp_data = \
            preprocess_snippet(data[n_snip], kp_names, params)

        # Save the file as a pickle with the same name as the input file in the
        # folder specified by save_path
        if save_path is None:
            save_path = os.path.join(os.getcwd(),
                                     'results', 'JDM25_v4',
                                     'snippet%d.p' % (n_snip))
        else:
            save_path = os.path.join(save_path, files[n_snip][:-4] + '.p')
        print(save_path, flush=True)
        compute_stac(kp_data, save_path, params)
        print('Finished %d' % (n_snip))

    # Support file-based processing
    else:
        kp_data, kp_names = \
            preprocess_data(data_path, start_frame, params)
        if save_path is None:
            save_path = os.path.join(os.getcwd(),
                                     'results',
                                     'snippet' + data_path[:-4] + '.p')
        compute_stac(kp_data, save_path, params)


if __name__ == '__main__':
    clize.run(handle_args)
