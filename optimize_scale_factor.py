from dm_control import viewer
from scipy.io import savemat
import clize
import stac
import rodent_environments
import numpy as np
import util
import scipy.optimize
import pickle
import os
import compute_stac
_MM_TO_METERS = 1000


def scale_loss(scale_factor, data_path, params, kp_data, kp_names):
    kp_data = preprocess_data(scale_factor, data_path, params,
                              kp_data, kp_names)
    env = rodent_environments.rodent_mocap(kp_data, params)
    compute_stac.root_optimization(env, params)
    site_pos = np.copy(env.physics.bind(env.task._walker.body_sites).xpos[:])
    print(scale_factor)
    return (site_pos.flatten() - kp_data[0, :].squeeze())**2


def preprocess_data(scale_factor, data_path, params, kp_data, kp_names,
                    struct_name='markers_preproc'):
    """Preprocess mocap data for stac fitting.

    :param data_path: Path to .mat mocap file
    :param skip: Subsampling rate for the frames
    :param scale_factor: Multiplier for mocap data
    :param z_offset: Subtractive offset for mocap_data
    :param struct_name: Field name of .mat file to load
    """
    kp_data = kp_data[params['start_frame']:, :]
    kp_data = kp_data/_MM_TO_METERS

    spineM_id = np.argwhere([name == 'SpineM' for name in kp_names])
    spineM = np.squeeze(kp_data[:, spineM_id*3 + np.array([0, 1, 2])])

    # Rescale by centering at spineM, scaling, and decentering
    for dim in range(np.round(kp_data.shape[1]/3).astype('int32')):
        kp_data[:, dim*3 + np.array([0, 1, 2])] -= spineM
    kp_data *= scale_factor
    spineM *= scale_factor
    for dim in range(np.round(kp_data.shape[1]/3).astype('int32')):
        kp_data[:, dim*3 + np.array([0, 1, 2])] += spineM

    # Downsample
    kp_data = kp_data[::params['skip'], :]
    kp_data = kp_data[:params['n_frames'], :]
    # Handle z-offset conditions
    if params['adaptive_z_offset']:
        kp_data[:, 2::3] -= \
            np.nanpercentile(kp_data[:, 2::3].flatten(), params['z_perc'])
        kp_data[:, 2::3] += params['adaptive_z_offset_value']
    else:
        kp_data[:, 2::3] -= params['z_offset']
    return kp_data


def optimize_scale_factor(data_path, param_path,
                          start_frame, *, n_frames=50, skip=10):
    kw = {"start_frame": int(start_frame),
          "n_frames": n_frames,
          "skip": skip,
          "adaptive_z_offset": True,
          "visualize": True}
    params = util.load_params(param_path)
    for key, v in kw.items():
        params[key] = v
    kp_data, kp_names = util.load_kp_data_from_file(
        data_path,
        struct_name='markers_preproc')
    # scale_factor0 = params['scale_factor']
    scale_factor0 = 1.3
    scale_opt = scipy.optimize.least_squares(
        lambda scale: scale_loss(scale, data_path, params, kp_data, kp_names),
        scale_factor0
        )
    print(scale_opt.x)


if __name__ == '__main__':
    clize.run(optimize_scale_factor)
