"""Compute stac optimization on data."""
from dm_control import viewer
from scipy.io import savemat
# from scipy.ndimage.filters import gaussian_filter1d
import clize
import stac
import rodent_environments
import numpy as np
import util
import pickle
import os

_IS_LIMB = ['scapula', 'hip', 'knee',
            'shoulder', 'elbow']
_MM_TO_METERS = 1000


def preprocess_snippet(kp_data, kp_names, skip=2, scale_factor=1.5,
                       smooth_arms=True, z_offset=.02, adaptive_z_offset=False,
                       adaptive_z_offset_value=.02):
    """Preprocess snippet data."""
    # # Smooth arms, if desired
    # if smooth_arms:
    #     for id, name in enumerate(kp_names):
    #         if any(part in name for part in ["Arm", "Elbow"]):
    #             trace = kp_data[:, id*3 + np.array([0, 1, 2])]
    #             kp_data[:, id*3 + np.array([0, 1, 2])] = \
    #                 gaussian_filter1d(trace, 1, axis=0)
    kp_data = kp_data/_MM_TO_METERS
    spineM_id = np.argwhere([name == 'SpineM' for name in kp_names])
    spineM = np.squeeze(kp_data[:, spineM_id*3 + np.array([0, 1, 2])])
    # Resize
    for dim in range(np.round(kp_data.shape[1]/3).astype('int32')):
        kp_data[:, dim*3] -= spineM[:, 0]
        kp_data[:, dim*3 + 1] -= spineM[:, 1]
        kp_data[:, dim*3 + 2] -= spineM[:, 2]
    kp_data = kp_data*scale_factor
    spineM *= scale_factor
    for dim in range(np.round(kp_data.shape[1]/3).astype('int32')):
        kp_data[:, dim*3] += spineM[:, 0]
        kp_data[:, dim*3 + 1] += spineM[:, 1]
        kp_data[:, dim*3 + 2] += spineM[:, 2]
    kp_data = kp_data[::skip, :]
    # Handle z-offset ambiguity
    if adaptive_z_offset:
        kp_data[:, 2::3] -= \
            np.nanpercentile(kp_data[:, 2::3].flatten(), 5)
        kp_data[:, 2::3] += adaptive_z_offset_value
    else:
        kp_data[:, 2::3] -= z_offset
    return kp_data


def preprocess_data(data_path, start_frame,
                    skip=10, scale_factor=1.5, z_offset=.02,
                    adaptive_z_offset=False,
                    adaptive_z_offset_value=.02,
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
    kp_data = kp_data[start_frame::skip, :]/_MM_TO_METERS*scale_factor
    if adaptive_z_offset:
        kp_data[:, 2::3] -= \
            np.nanpercentile(kp_data[:, 2::3].flatten(), 5)
        kp_data[:, 2::3] += adaptive_z_offset_value
    else:
        kp_data[:, 2::3] -= z_offset
    return kp_data


def initial_optimization(env, initial_offsets, m_reg_coef=.1, maxiter=100):
    """Optimize the first frame with alternating q and m phase."""
    stac.q_phase(env.physics, env.task.kp_data[0, :],
                 env.task._walker.body_sites, root_only=True)
    root_q = [env.physics.named.data.qpos[:].copy()]
    stac.m_phase(env.physics, env.task.kp_data, env.task._walker.body_sites,
                 [0], root_q, initial_offsets,
                 reg_coef=m_reg_coef, maxiter=maxiter)


def root_optimization(env):
    """Optimize only the root."""
    stac.q_phase(env.physics, env.task.kp_data[0, :],
                 env.task._walker.body_sites, root_only=True)


def q_clip(env, n_frames, q_reg_coef, qs_to_opt):
    """Q-phase across the clip: optimize joint angles."""
    q = []
    for i in range(n_frames):
        stac.q_phase(env.physics, env.task.kp_data[i, :],
                     env.task._walker.body_sites, reg_coef=q_reg_coef)
        if i == 0:
            temp_reg = False
        else:
            temp_reg = True
        stac.q_phase(env.physics, env.task.kp_data[i, :],
                     env.task._walker.body_sites, reg_coef=q_reg_coef,
                     qs_to_opt=qs_to_opt, temporal_regularization=temp_reg)
        retval = env.physics.named.data.qpos[:]
        q.append(retval.copy())
    return q


def compute_stac(kp_data, save_path, *,
                 offset_path=None,
                 start_frame=0,
                 n_frames=None,
                 n_sample_frames=50,
                 q_reg_coef=0.,
                 m_reg_coef=.1,
                 skip=10,
                 scale_factor=1.5,
                 z_offset=.02,
                 verbose=False,
                 visualize=False,
                 render_video=False):
    """Perform stac on rat mocap data.

    :param kp_data: mocap_data
    :param save_path: File to save optimized qposes
    :param offset_path: Path to precomputed marker offsets
    :param start_frame: Frame within mocap file to start optimization
    :param n_frames: Number of frames to optimize
    :param n_sample_frames: Number of frames to use in m-phase optimization
    :param q_reg_coef: Regularization coefficient for q-phase
    :param m_reg_coef: Regularization coefficient for m-phase
    :param skip: Subsampling rate for the frames
    :param scale_factor: Multiplier for mocap data
    :param z_offset: Subtractive offset for mocap_data
    :param verbose: If True, display messages during optimization.
    :param visualize: If True, launch viewer
    :param render_video: If True, render_video
    """
    if n_frames is None:
        n_frames = kp_data.shape[0]
    n_frames = int(n_frames)
    # Build the environment
    env = rodent_environments.rodent_mocap(kp_data, int(n_frames))

    # Get the ids of the limbs
    part_names = env.physics.named.data.qpos.axes.row.names
    for i in range(6):
        part_names.insert(0, part_names[0])

    limbs = np.array([any(part in name for part in _IS_LIMB)
                      for name in part_names])

    # If preloading offsets, set them now.
    if offset_path is not None:
        with open(offset_path, 'rb') as f:
            in_dict = pickle.load(f)
        sites = env.task._walker.body_sites
        env.physics.bind(sites).pos[:] = \
            in_dict['offsets']
        for id, p in enumerate(env.physics.bind(sites).pos):
            sites[id].pos = p
        if verbose:
            print('Root Optimization', flush=True)
        root_optimization(env)
    else:
        # Get the initial offsets of the markers
        initial_offsets = \
            np.copy(env.physics.bind(env.task._walker.body_sites).pos[:])

        # First optimize the first frame to get an approximation
        # of the m and q phases
        if verbose:
            print('Initial Optimization', flush=True)
        initial_optimization(env, initial_offsets, m_reg_coef=m_reg_coef)

        # Find the frames to use in the m-phase optimization.
        time_indices = np.random.randint(0, n_frames, n_sample_frames)

    # Q_phase optimization
    if verbose:
        print('q-phase', flush=True)
    q = q_clip(env, n_frames, q_reg_coef, limbs)

    # If you've precomputed the offsets, stop here.
    # Otherwise do another m and q phase.
    if offset_path is None:
        # M-phase across the subsampling: optimize offsets
        if verbose:
            print('m-phase', flush=True)
        stac.m_phase(env.physics, env.task.kp_data,
                     env.task._walker.body_sites, time_indices, q,
                     initial_offsets, reg_coef=m_reg_coef)
        if verbose:
            print('q-phase', flush=True)
        q = q_clip(env, n_frames, q_reg_coef, limbs)

    # Optional visualization
    if visualize:
        env.task.precomp_qpos = q
        env.task.render_video = render_video
        viewer.launch(env)
        if env.task.V is not None:
            env.task.V.release()

    # Save the pose, offsets, data, and all parameters
    filename, file_extension = os.path.splitext(save_path)
    offsets = env.physics.bind(env.task._walker.body_sites).pos[:].copy()
    out_dict = {'qpos': q,
                'offsets': offsets,
                'kp_data': np.copy(kp_data[:n_frames, :]),
                'start_frame': start_frame,
                'offset_path': offset_path,
                'n_frames': n_frames,
                'n_sample_frames': n_sample_frames,
                'q_reg_coef': q_reg_coef,
                'm_reg_coef': m_reg_coef,
                'skip': skip,
                'scale_factor': scale_factor,
                'z_offset': z_offset,
                'verbose': verbose,
                'visualize': visualize,
                'render_video': render_video}
    if file_extension == '.p':
        with open(save_path, "wb") as output_file:
            pickle.dump(out_dict, output_file)
    elif file_extension == '.mat':
        savemat(save_path, out_dict)


def handle_args(data_path, *,
                save_path=None,
                offset_path=None,
                start_frame=0,
                n_snip=None,
                n_frames=None,
                n_sample_frames=50,
                q_reg_coef=0.,
                m_reg_coef=.25,
                skip=10,
                scale_factor=1.2,
                z_offset=.02,
                verbose=False,
                visualize=False,
                render_video=False,
                adaptive_z_offset_value=.02):
    """Wrap compute_stac to perform appropriate processing.

    :param data_path: List of paths to .mat mocap data files
    :param save_path: File to save optimized qposes
    :param offset_path: Path to precomputed marker offsets
    :param start_frame: Frame within mocap file to start optimization
    :param n_frames: Number of frames to optimize
    :param n_sample_frames: Number of frames to use in m-phase optimization
    :param q_reg_coef: Regularization coefficient for q-phase
    :param m_reg_coef: Regularization coefficient for m-phase
    :param skip: Subsampling rate for the frames
    :param scale_factor: Multiplier for mocap data
    :param z_offset: Subtractive offset for mocap_data
    :param verbose: If True, display messages during optimization.
    :param visualize: If True, launch viewer
    :param render_video: If True, render_video
    """
    kw = {"offset_path": offset_path,
          "start_frame": start_frame,
          "n_frames": n_frames,
          "n_sample_frames": n_sample_frames,
          "q_reg_coef": q_reg_coef,
          "m_reg_coef": m_reg_coef,
          "skip": skip,
          "scale_factor": scale_factor,
          "z_offset": z_offset,
          "verbose": verbose,
          "visualize": visualize,
          "render_video": render_video}

    if n_snip is not None:
        data, kp_names = util.load_snippets_from_file(data_path)
        n_snip = int(n_snip) - 1  # Handle seq offset
        print(n_snip)
        kp_data = \
            preprocess_snippet(data[n_snip], kp_names,
                               scale_factor=scale_factor,
                               adaptive_z_offset=True,
                               adaptive_z_offset_value=adaptive_z_offset_value)
        if save_path is None:
            save_path = os.path.join(os.getcwd(),
                                     'results', 'JDM25_v3',
                                     'snippet%d.p' % (n_snip))
        print(save_path)
        compute_stac(kp_data, save_path, **kw)
        print('Finished %d' % (n_snip))

    else:
        kp_data = \
            preprocess_data(data_path, start_frame, skip=skip,
                            scale_factor=scale_factor,
                            adaptive_z_offset=True,
                            z_offset=z_offset,
                            adaptive_z_offset_value=adaptive_z_offset_value)
        if save_path is None:
            save_path = os.path.join(os.getcwd(),
                                     'results',
                                     'snippet' + data_path[:-4] + '.p')
        compute_stac(kp_data, save_path, **kw)


if __name__ == '__main__':
    clize.run(handle_args)
