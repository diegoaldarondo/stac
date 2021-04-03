"""View stac results.

Attributes:
    FPS (int): Default frames per second of the video. 
"""
from dm_control import viewer
from dm_control.locomotion.walkers import rescale
import clize
import stac.rodent_environments as rodent_environments
import numpy as np
import pickle
import stac.util as util
import os
import imageio
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums
from typing import Text, List, Dict, Union, Tuple

FPS = 50


def load_data(data_path: Text, start_frame: int = 0, end_frame: int = -1):
    """Load data from .pickle file

    Args:
        data_path (TYPE): Path to .pickle file contatining qpos
        start_frame (int, optional): First frame
        end_frame (TYPE, optional): Last frame

    Returns:
        TYPE: Qpos List, site offsets, kp_data np.ndarray, number of frames

    Raises:
        ValueError: Description
    """
    with open(data_path, "rb") as f:
        in_dict = pickle.load(f)
        q = in_dict["qpos"]
        if isinstance(start_frame, int) and isinstance(end_frame, int):
            q = q[start_frame:end_frame]
        else:
            raise ValueError("start_frame and end_frame must be an integer")
        n_frames = len(q)
        if "offsets" not in in_dict.keys():
            offsets = np.zeros((20, 3))
        else:
            offsets = in_dict["offsets"]
        if "kp_data" in in_dict.keys():
            kp_data = in_dict["kp_data"]
        else:
            kp_data = np.zeros((n_frames, offsets.size))
    return q, offsets, kp_data, n_frames


def view_stac(
    data_path: Text,
    param_path: Text,
    *,
    render_video: bool = False,
    save_path: Text = None,
    headless: bool = False,
    start_frame: int = 0,
    end_frame: int = -1
):
    """View the output of stac.

    Args:
        data_path (Text): Path to .p file containing qpos, offsets, and optionally kp_data.
        param_path (Text): Path to stac parameters file.
        render_video (bool, optional): If True, make a video and put it in clips.
        save_path (Text, optional): Save any rendered videos to this path.
        headless (bool, optional): If True, make a video in headless mode.
        start_frame (int, optional): Starting frame for rendering.
        end_frame (int, optional): Ending frame for rendering.
    """
    # Load in the qpos, offsets, and markers
    q, offsets, kp_data, n_frames = load_data(
        data_path, start_frame=start_frame, end_frame=end_frame
    )
    print(len(q[0]))
    setup_visualization(
        param_path,
        q,
        offsets,
        kp_data,
        n_frames,
        render_video=render_video,
        save_path=save_path,
        headless=headless,
    )


def setup_visualization(
    param_path: str,
    q,
    offsets,
    kp_data,
    n_frames: int,
    render_video: bool = False,
    save_path: Text = None,
    headless: bool = False,
):
    """Sets up stac visualization.

    Args:
        param_path (str): Path to stac parameters file.
        q (TYPE): Array of qpos for each frames.
        offsets (TYPE): Site offsets
        kp_data (TYPE): Description
        n_frames (TYPE): Number of frames
        render_video (bool, optional): If True, make a video and put it in clips.
        save_path (Text, optional): Save any rendered videos to this path.
        headless (bool, optional): If True, make a video in headless mode.
    """
    params = util.load_params(param_path)
    params["n_frames"] = n_frames - 1

    # Build the environment, and set the offsets, and params
    if params["ARENA_TYPE"] == "HField":
        params[
            "data_path"
        ] = "/home/diego/data/dm/stac/snippets/snippets_snippet_v8_JDM31_Day_8/reformatted/snippet_3_AdjustPosture.mat"
        env = rodent_environments.rodent_mocap(
            kp_data,
            params,
            # hfield_image=in_dict['hfield_image'],
            hfield_image=None,
            pedestal_center=in_dict["pedestal_center"],
            pedestal_radius=in_dict["pedestal_radius"],
            pedestal_height=in_dict["pedestal_height"],
            arena_diameter=in_dict["_ARENA_DIAMETER"],
            arena_center=in_dict["_ARENA_CENTER"],
        )
    elif params["ARENA_TYPE"] == "Standard":
        env = rodent_environments.rodent_mocap(kp_data, params)
    elif params["ARENA_TYPE"] == "DannceArena":
        env = rodent_environments.rodent_mocap(
            kp_data,
            params,
            arena_diameter=params["_ARENA_DIAMETER"],
            arena_center=params["_ARENA_CENTER"],
        )

    rescale.rescale_subtree(
        env.task._walker._mjcf_root,
        params["scale_factor"],
        params["scale_factor"],
    )
    env.task._arena._mjcf_root.worldbody.add(
        "camera",
        name="CameraE",
        pos=[-0.0452, 1.5151, 0.3174],
        fovy=50,
        quat="0.0010 -0.0202 -0.7422 -0.6699",
    )
    env.reset()
    sites = env.task._walker.body_sites
    env.physics.bind(sites).pos[:] = offsets
    for n_offset, site in enumerate(sites):
        site.pos = offsets[n_offset, :]
    env.task.precomp_qpos = q
    env.task.render_video = render_video

    # Render a video in headless mode
    env.task.initialize_episode(env.physics, 0)
    prev_time = env.physics.time()
    scene_option = wrapper.MjvOption()
    scene_option.geomgroup[2] = 1
    scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.geomgroup[3] = 1
    n_frame = 0
    frames = []
    if headless & render_video:
        while prev_time < env._time_limit:
            while (np.round(env.physics.time() - prev_time, decimals=5)) < params[
                "_TIME_BINS"
            ]:
                env.physics.step()
            env.task.after_step(env.physics, None)
            rgbArr = env.physics.render(
                368,
                368,
                camera_id="walker/close_profile",
                scene_option=scene_option,
            )
            frames.append(rgbArr)
            n_frame += 1
            print(n_frame)
            prev_time = np.round(env.physics.time(), decimals=2)
    # Otherwise, use the viewer
    else:
        viewer.launch(env)
    if env.task.V is not None:
        env.task.V.release()
    print(n_frame, flush=True)
    if save_path is not None and render_video:
        env.task.video_name = save_path
        print("Rendering: ", env.task.video_name)
        write_video(save_path, frames)


def write_video(filename: Text, frames: List, fps: int = FPS):
    """Write a video to disk

    Args:
        filename (Text): File name in which to save video
        frames (List): List of frames to include in video
        fps (int, optional): Frames per second.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    print("Writing to %s" % (filename))
    with imageio.get_writer(filename, fps=fps) as video:
        for frame in frames:
            video.append_data(frame)


if __name__ == "__main__":
    clize.run(view_stac)
