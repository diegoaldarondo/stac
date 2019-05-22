"""View stac results."""
from dm_control import viewer
import clize
import rodent_environments
import numpy as np
import pickle
import util


def view_stac(data_path, param_path, *,
              render_video=False,
              save_path=None,
              headless=False):
    """View the output of stac.

    :param data_path: Path to .p file containing qpos, offsets,
                      and optionally kp_data.
    :param render_video: If True, make a video and put it in clips.
    :param save_path: Save any rendered videos to this path.
    :param headless: If True, make a video in headless mode.
    """
    # Load in the qpos, offsets, and markers
    with open(data_path, 'rb') as f:
        in_dict = pickle.load(f)
        offsets = in_dict['offsets']
        q = in_dict['qpos']
        n_frames = len(q)
        if 'kp_data' in in_dict.keys():
            kp_data = in_dict['kp_data']
        else:
            kp_data = np.zeros((n_frames, offsets.size))

    params = util.load_params(param_path)
    params['n_frames'] = n_frames
    # Build the environment, and set the offsets, and params
    env = rodent_environments.rodent_mocap(kp_data, params)
    sites = env.task._walker.body_sites
    env.physics.bind(sites).pos[:] = offsets
    for id, site in enumerate(sites):
        site.pos = offsets[id, :]
    env.task.precomp_qpos = q
    env.task.render_video = render_video
    if save_path is not None:
        env.task.video_name = save_path
        print('Rendering: ', env.task.video_name)

    # Render a video in headless mode
    prev_time = env.physics.time()
    if headless & render_video:
        while prev_time < env._time_limit:
            while (env.physics.time() - prev_time) < params['_TIME_BINS']:
                env.physics.step()
            env.task.after_step(env.physics, None)
            prev_time = env.physics.time()

    # Otherwise, use the viewer
    else:
        viewer.launch(env)
    if env.task.V is not None:
        env.task.V.release()


if __name__ == "__main__":
    clize.run(view_stac)
