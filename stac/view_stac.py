"""View stac results.

Attributes:
    FPS (int): Default frames per second of the video. 
"""
# from dm_control import viewer
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
from scipy.ndimage import median_filter, uniform_filter1d
import re
import scipy.io as spio
from scipy.ndimage import gaussian_filter
import cv2

ALPHA_BASE_VALUE = 0.5
FPS = 50


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_data(
    data_path: Text, start_frame: int = 0, end_frame: int = -1, return_q_names=False
):
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

        q_names = in_dict["names_qpos"]
        # print(q_names)
        q = fix_tail(q, q_names)

    if return_q_names:
        return q, offsets, kp_data, n_frames, q_names
    else:
        return q, offsets, kp_data, n_frames


def fix_tail(q: np.ndarray, q_names: List):
    """Fix the tail so that it is in the base position.

    Args:
        q (np.ndarray): qpos
        q_names (List): names of each dimension of qpos

    Returns:
        [type]: [description]
    """
    for i, name in enumerate(q_names):
        if re.match("walker/vertebra_C.*bend", name):
            # print(name, i, flush=True)
            q[:, i] = 0.0
        if re.match("walker/vertebra_C.*extend", name):
            q[:, i] = 0.0
    return q


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
    segmented=False,
    camera_kwargs: List = None,
    registration_xml: bool = False,
):
    """Sets up stac visualization.

    Args:
        param_path (str): Path to stac parameters file.
        q (TYPE): Array of qpos for each frames.
        offsets (TYPE): Site offsets
        kp_data (TYPE): Description
        n_frames (TYPE): Number of frames
        render_video (bool, optional): If True, make a video and put it in clips.
    """
    params = util.load_params(param_path)
    params["n_frames"] = n_frames - 1

    if segmented:
        alpha = 0.0
        params["_ARENA_DIAMETER"] = None
    else:
        alpha = 1.0
        params["_ARENA_DIAMETER"] = None
    if registration_xml:
        params[
            "_XML_PATH"
        ] = "/n/home02/daldarondo/LabDir/Diego/code/dm/stac/models/rodent_stac.xml"

    env = setup_arena(kp_data, params, alpha=alpha)

    rescale.rescale_subtree(
        env.task._walker._mjcf_root,
        params["scale_factor"],
        params["scale_factor"],
    )

    # Setup cameras
    # fovy = 2*atan(height/(2*params{1}.K(2,2)))/2*pi*360
    if camera_kwargs is not None:
        for kwargs in camera_kwargs:
            env.task._arena._mjcf_root.worldbody.add("camera", **kwargs)
    else:
        env.task._arena._mjcf_root.worldbody.add(
            "camera",
            name="Camera1",
            pos=[-0.8781364, 0.3775911, 0.4291190],
            fovy=28.8255,
            quat="0.5353    0.3435   -0.4623   -0.6178",
        )

    env.task._arena._mjcf_root.worldbody.add(
        "camera",
        name="RearCamera",
        pos=[-1.5461, 0.1636, 0.2243],
        fovy=50,
        quat="0.5353    0.3435   -0.4623   -0.6178",
    )

    # env.physics.contexts.mujoco._ptr.contents.fogRGBA[0] = 1.0
    # env.physics.contexts.mujoco._ptr.contents.fogRGBA[1] = 1.0
    # env.physics.contexts.mujoco._ptr.contents.fogRGBA[2] = 1.0
    env.reset()
    setup_sites(q, offsets, env)
    env.task.render_video = render_video
    env.task.initialize_episode(env.physics, 0)
    scene_option = setup_hires_scene()
    return params, env, scene_option


def setup_lores_scene():
    """Return scene_option for lores scene.

    Returns:
        [type]: MjvOption
    """
    scene_option = wrapper.MjvOption()
    scene_option.geomgroup[2] = 1
    scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True

    scene_option.geomgroup[3] = 1
    return scene_option


def setup_hires_scene():
    """Return scene_option for hires scene.

    Returns:
        [type]: MjvOption
    """
    scene_option = wrapper.MjvOption()
    # scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    # scene_option.geomgroup[3] = 0
    # scene_option.sitegroup[0] = 0
    # scene_option.sitegroup[1] = 0
    scene_option.sitegroup[2] = 0
    scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_LIGHT] = False
    scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
    scene_option._ptr.contents.flags[enums.mjtRndFlag.mjRND_SHADOW] = False
    scene_option._ptr.contents.flags[enums.mjtRndFlag.mjRND_REFLECTION] = False
    scene_option._ptr.contents.flags[enums.mjtRndFlag.mjRND_SKYBOX] = False
    scene_option._ptr.contents.flags[enums.mjtRndFlag.mjRND_FOG] = False
    return scene_option


def setup_variability_scene():
    """Return scene_option for hires scene.

    Returns:
        [type]: MjvOption
    """
    scene_option = wrapper.MjvOption()
    # scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    # scene_option.geomgroup[3] = 0
    # scene_option.sitegroup[0] = 0
    # scene_option.sitegroup[1] = 0
    scene_option.sitegroup[2] = 0
    scene_option.sitegroup[5] = 1
    scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_LIGHT] = False
    scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
    scene_option._ptr.contents.flags[enums.mjtRndFlag.mjRND_SHADOW] = False
    scene_option._ptr.contents.flags[enums.mjtRndFlag.mjRND_REFLECTION] = False
    scene_option._ptr.contents.flags[enums.mjtRndFlag.mjRND_SKYBOX] = False
    scene_option._ptr.contents.flags[enums.mjtRndFlag.mjRND_FOG] = False
    return scene_option


def setup_sites(q: np.ndarray, offsets: np.ndarray, env):
    """Setup the walker sites.

    Args:
        q (np.ndarray): qpos
        offsets (np.ndarray): model stac offsets
        env ([type]): environment
    """
    sites = env.task._walker.body_sites
    env.physics.bind(sites).pos[:] = offsets
    for n_offset, site in enumerate(sites):
        site.pos = offsets[n_offset, :]
    print(q.shape)
    # q[:, 7:] = uniform_filter1d(q[:, 7:], size=5, axis=0)
    env.task.precomp_qpos = q


def load_calibration(dannce_file: Text) -> Tuple:
    """Load camera1 calibration from dannce.mat file

    Args:
        dannce_file (Text): Path to dannce.mat file

    Returns:
        Tuple: Intrinsic matrix, radial distortion, tangential distortion, R, t
    """
    M = loadmat(dannce_file)
    return M["params"]


def overlay_frame(
    rgb_frame: np.ndarray,
    params: List,
    recon_frame: np.ndarray,
    seg_frame: np.ndarray,
    cam_id: int = 0,
) -> np.ndarray:
    """Make a single overlain frame of reconstruction + rgb.

    Args:
        rgb_frame (np.ndarray): Original undistorted video frame.
        K (np.ndarray): Intrinsic matrix
        rdistort (np.ndarray): Radial distortion
        tdistort (np.ndarray): Tangential distortion
        recon_frame (np.ndarray): Reconstructed frame.
        seg_frame (np.ndarray): Segmented frame.

    Returns:
        np.ndarray: overlain frame
    """
    # Load and undistort the rgb frame
    rgb_frame = cv2.undistort(
        rgb_frame,
        params[cam_id].K.T,
        np.concatenate(
            [params[cam_id].RDistort, params[cam_id].TDistort], axis=0
        ).T.squeeze(),
        params[cam_id].K.T,
    )

    # Calculate the alpha mask using the segmented video
    alpha = (seg_frame[:, :, 0] >= 0.0) * ALPHA_BASE_VALUE
    alpha = gaussian_filter(alpha, 2)
    alpha = gaussian_filter(alpha, 2)
    alpha = gaussian_filter(alpha, 2)
    frame = np.zeros_like(recon_frame)

    # Blend the two videos
    for n_chan in range(recon_frame.shape[2]):
        frame[:, :, n_chan] = (
            alpha * recon_frame[:, :, n_chan] + (1 - alpha) * rgb_frame[:, :, n_chan]
        )
    return frame


def mujoco_loop(
    save_path: Text,
    params: Dict,
    env,
    scene_option,
    camera: Text = "walker/close_profile",
    height: int = 1200,
    width: int = 1920,
):
    """Rendering loop for generating mujoco videos.

    Args:
        save_path (Text): Path to save overlay video
        params (Dict): stac parameters dictionary
        env ([type]): rodent environment
        scene_option ([type]): MjvOption rendering options
        camera (Text, optional): Name of the camera to use for rendering. Defaults to "walker/close_profile".
        height (int, optional): Camera height in pixels. Defaults to 1200.
        width (int, optional): Camera width in pixels. Defaults to 1920.
    """
    prev_time = env.physics.time()
    n_frame = 0
    with imageio.get_writer(save_path, fps=FPS) as video:
        env.task.after_step(env.physics, None)
        reconArr = env.physics.render(
            height,
            width,
            camera_id=camera,
            scene_option=scene_option,
        )
        video.append_data(reconArr)
        n_frame += 1
        while prev_time < env._time_limit:
            while (np.round(env.physics.time() - prev_time, decimals=5)) < params[
                "_TIME_BINS"
            ]:
                env.physics.step()
            env.task.after_step(env.physics, None)

            # env.physics.contexts.mujoco._ptr.contents.fogRGBA[2] = 1.0
            reconArr = env.physics.render(
                height,
                width,
                camera_id=camera,
                scene_option=scene_option,
            )
            # segArr = env.physics.render(
            #     height,
            #     width,
            #     camera_id=camera,
            #     scene_option=scene_option,
            #     segmentation=True,
            # )
            # background = segArr[:, :, 0] < 0.0
            # for i in range(reconArr.shape[2]):
            #     fr = reconArr[:, :, i]
            #     fr[background] = 255
            #     reconArr[:, :, i] = fr

            # import pdb

            # pdb.set_trace()
            video.append_data(reconArr)

            n_frame += 1
            # print(n_frame)
            prev_time = np.round(env.physics.time(), decimals=2)


def overlay_loop(
    save_path: Text,
    video_path: Text,
    calibration_path: Text,
    frames: np.ndarray,
    params: Dict,
    env,
    scene_option,
    camera: Text = "walker/close_profile",
    height: int = 1200,
    width: int = 1920,
):
    """Rendering loop for generating overlay videos.

    Args:
        save_path (Text): Path to save overlay video
        video_path (Text): Path to undistorted rgb video
        calibration_path (Text): Path to calibration dannce.mat file.
        frames (np.ndarray): Frames to render
        params (Dict): stac parameters dictionary
        env ([type]): rodent environment
        scene_option ([type]): MjvOption rendering options
        camera (Text, optional): Name of the camera to use for rendering. Defaults to "walker/close_profile".
        height (int, optional): Camera height in pixels. Defaults to 1200.
        width (int, optional): Camera width in pixels. Defaults to 1920.
    """
    prev_time = env.physics.time()
    reader = imageio.get_reader(video_path)
    n_frame = 0
    cam_params = load_calibration(calibration_path)
    env.task.after_step(env.physics, None)
    with imageio.get_writer(save_path, fps=FPS) as video:
        for n_frame in range(len(frames)):
            if n_frame > 0:
                while (np.round(env.physics.time() - prev_time, decimals=5)) < params[
                    "_TIME_BINS"
                ]:
                    env.physics.step()
            frame = render_overlay(
                frames,
                env,
                scene_option,
                camera,
                height,
                width,
                reader,
                n_frame,
                cam_params,
            )
            video.append_data(frame)
            print(n_frame)
            prev_time = np.round(env.physics.time(), decimals=2)


def render_overlay(
    frames, env, scene_option, camera, height, width, reader, n_frame, cam_params
):
    env.task.after_step(env.physics, None)
    reconArr = env.physics.render(
        height,
        width,
        camera_id=camera,
        scene_option=scene_option,
    )
    segArr = env.physics.render(
        height,
        width,
        camera_id=camera,
        scene_option=scene_option,
        segmentation=True,
    )
    rgbArr = reader.get_data(frames[n_frame])
    frame = overlay_frame(rgbArr, cam_params, reconArr, segArr)
    return frame


def setup_arena(kp_data, params, alpha=1.0):
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
            pedestal_center=params["pedestal_center"],
            pedestal_radius=params["pedestal_radius"],
            pedestal_height=params["pedestal_height"],
            arena_diameter=params["_ARENA_DIAMETER"],
            arena_center=params["_ARENA_CENTER"],
        )
    elif params["ARENA_TYPE"] == "Standard":
        env = rodent_environments.rodent_mocap(kp_data, params)
    elif params["ARENA_TYPE"] == "DannceArena":
        if "_ARENA_DIAMETER" in params:
            diameter = params["_ARENA_DIAMETER"]
        else:
            diameter = None

        if "_ARENA_CENTER" in params:
            center = params["_ARENA_CENTER"]
        else:
            center = None
        env = rodent_environments.rodent_mocap(
            kp_data,
            params,
            arena_diameter=diameter,
            arena_center=center,
            alpha=alpha,
        )

    return env


def render_loop(
    render_video,
    save_path,
    headless,
    params,
    env,
    scene_option,
    camera: Text = "walker/close_profile",
    height=1200,
    width=1920,
):
    prev_time = env.physics.time()
    n_frame = 0
    frames, seg_frames = [], []
    if headless & render_video:
        while prev_time < env._time_limit:
            while (np.round(env.physics.time() - prev_time, decimals=5)) < params[
                "_TIME_BINS"
            ]:
                env.physics.step()
            env.task.after_step(env.physics, None)
            rgbArr = env.physics.render(
                height,
                width,
                camera_id=camera,
                scene_option=scene_option,
            )
            frames.append(rgbArr)
            # segArr = env.physics.render(
            #     1200,
            #     1920,
            #     camera_id=camera,
            #     scene_option=scene_option,
            #     segmentation=True,
            # )
            # seg_frames.append(segArr.astype(np.uint8))
            n_frame += 1
            print(n_frame)
            prev_time = np.round(env.physics.time(), decimals=2)
    # Otherwise, use the viewer
    else:
        from dm_control import viewer

        viewer.launch(env)
    if env.task.V is not None:
        env.task.V.release()
    print(n_frame, flush=True)
    if save_path is not None and render_video:
        env.task.video_name = save_path
        print("Rendering: ", env.task.video_name)
        write_video(save_path, frames)

        # filename, file_extension = os.path.splitext(save_path)
        # seg_path = filename + "_seg" + file_extension
        # write_video(seg_path, seg_frames)

    # image = np.concatenate(frames, axis=1)
    # save_path = save_path.split(".mp4")[0] + ".png"
    # if os.path.exists(save_path):
    #     os.remove(save_path)
    # imageio.imwrite(save_path, image)


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
