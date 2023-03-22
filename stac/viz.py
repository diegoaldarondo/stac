import pickle
import os
import stac.view_stac as view_stac
import numpy as np
import h5py
import imageio
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.io import loadmat
from typing import Text, List, Dict, Tuple
import dm_control.mujoco.math as mjmath
from scipy.spatial.transform import Rotation as R
import stac.rodent_environments as rodent_environments
import stac.util as util

FPS = 50
# OFFSET_PATH = "C:/data/virtual_rat/2021_06_21_1/total.p"
# OFFSET_PATH = "C:/data/virtual_rat/2020_12_22_2/total.p"
# PARAM_PATH = "C:/data/virtual_rat/2021_06_21_1/params.yaml"
# CALIBRATION_PATH = "C:/data/virtual_rat/2020_12_22_2/temp_dannce.mat"
ALPHA_BASE_VALUE = 0.5
Z_OFFSET = 0.013

# Standard image height for dannce rig data
HEIGHT = 1200


def load_reconstruction(reconstruction_path: Text) -> np.ndarray:
    """Load reconstruction data from a comic embedding

    Args:
        reconstruction_path (Text): Path to comic embedding file.

    Returns:
        [np.ndarray]: qpos
    """
    with h5py.File(reconstruction_path, "r") as f:
        qpos = f["qpos"][:]
    return qpos


def trim_video(video_path: Text, save_path: Text, frames: np.ndarray):
    """Write a trimmed version of a video to disk containing frames

    Args:
        video_path (Text): Path to original video
        save_path (Text): Save path for trimmed video
        frames (np.ndarray): Frames to include in trimmed video
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    reader = imageio.get_reader(video_path)
    with imageio.get_writer(save_path, fps=FPS) as video:
        for n_frame in frames:
            frame = reader.get_data(n_frame)
            video.append_data(frame)


def add_qualifier(video_path: Text, qualifier: Text) -> Text:
    """Utility to add qualifier to paths

    Args:
        video_path (Text): Path to original video
        qualifier (Text): qualifier name

    Returns:
        Text: qualifier path
    """
    filename, file_extension = os.path.splitext(video_path)
    qualifier_path = filename + qualifier + file_extension
    return qualifier_path


def quat2eul(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to XYZ euler angles

    Args:
        quat (np.ndarray): (w, ix, iy, iz) quaternion

    Returns:
        np.ndarray: XYZ euler angles in degrees
    """
    qw, qx, qy, qz, = (
        quat[0],
        quat[1],
        quat[2],
        quat[3],
    )
    euy = np.arctan2(2 * qy * qw - 2 * qx * qz, 1 - 2 * qy**2 - 2 * qz**2)
    euz = np.arcsin(2 * qx * qy + 2 * qz * qw)
    eux = np.arctan2(2 * qx * qw - 2 * qy * qz, 1 - 2 * qx**2 - 2 * qz**2)
    return np.array([np.rad2deg(eux), np.rad2deg(euy), np.rad2deg(euz)])


def get_project_paths(
    project_folder: Text, camera: Text
) -> Tuple[Text, Text, Text, Text]:
    """Get relevant paths given a project folder

    Args:
        project_folder (Text): Path to project folder
        camera (Text): Name of camera to use for video path.

    Returns:
        Tuple[Text, Text, Text, Text]: param_path, calibration_path, video_path, offset_path
    """
    param_path = os.path.join(project_folder, "stac_params", "params.yaml")
    calibration_path = os.path.join(project_folder, "temp_dannce.mat")
    if not os.path.exists(calibration_path):
        dannce_files = [
            os.path.join(project_folder, f)
            for f in os.listdir(project_folder)
            if "dannce.mat" in f
        ]
        calibration_path = dannce_files[0]
    video_path = os.path.join(project_folder, "videos", camera, "0.mp4")
    offset_path = os.path.join(project_folder, "stac", "total.p")
    if not os.path.exists(offset_path):
        offset_path = os.path.join(project_folder, "stac", "offset.p")
    offset_path = os.path.join(project_folder, "stac", "offset.p")
    return param_path, calibration_path, video_path, offset_path


def get_social_project_paths(
    project_folder: Text, camera: Text
) -> Tuple[Text, Text, Text, Text]:
    """Get relevant paths given a project folder

    Args:
        project_folder (Text): Path to project folder
        camera (Text): Name of camera to use for video path.

    Returns:
        Tuple[Text, Text, Text, Text]: param_path, calibration_path, video_path, offset_path
    """
    param_path = os.path.join(project_folder, "stac_params", "params.yaml")
    calibration_path = os.path.join(project_folder, "temp_dannce.mat")
    if not os.path.exists(calibration_path):
        dannce_files = [
            os.path.join(project_folder, f)
            for f in os.listdir(project_folder)
            if "dannce.mat" in f
        ]
        calibration_path = dannce_files[0]
    video_path = os.path.join(project_folder, "videos", camera, "0.mp4")
    offset_paths = [
        os.path.join(project_folder, "stac", "total1.p"),
        os.path.join(project_folder, "stac", "total2.p"),
    ]
    if not os.path.exists(offset_paths[0]):
        offset_paths = [
            os.path.join(project_folder, "stac", "offset1.p"),
            os.path.join(project_folder, "stac", "offset2.p"),
        ]
    return param_path, calibration_path, video_path, offset_paths


def generate_video(
    project_folder: Text,
    frames: np.ndarray,
    save_path: Text,
    video_type: Text,
    model_data_path: Text = None,
    camera: Text = "Camera1",
    use_stac: bool = False,
    segmented: bool = False,
    height: int = 1200,
    width: int = 1920,
    registration_xml: bool = False,
):
    """Generate a video given a project folder.

    Args:
        project_folder (Text): Path to project folder.
        frames (np.ndarray): Frames to include in video.
        save_path (Text): Path to save video
        video_type (Text): Video type. Can be ["overlay", "mujoco"].
        model_data_path (Text, optional): Path to comic embedding. Defaults to None.
        camera (Text, optional): Name of camera to render in mujoco. Defaults to "Camera1".
        use_stac (bool, optional): If True, use the stac registration. Defaults to False.
        segmented (bool, optional): If True, segment the background of the mujoco render. Defaults to False.
        height (int, optional): Height of video in pixels. Defaults to 1200.
        width (int, optional): Width of video in pixels. Defaults to 1920.
        registration_xml (bool, optional): If true, use the registration xml file. Defaults to False.
    """
    params, env, scene_option, video_path, calibration_path = setup_video_scene(
        project_folder,
        model_data_path,
        frames,
        camera,
        segmented,
        use_stac,
        registration_xml,
        video_type,
    )
    if video_type == "mujoco":
        # Render the videos
        view_stac.mujoco_loop(
            save_path,
            params,
            env,
            scene_option,
            camera=camera,
            height=height,
            width=width,
        )
    elif video_type == "overlay":
        # Render the videos
        view_stac.overlay_loop(
            save_path,
            video_path,
            calibration_path,
            frames,
            params,
            env,
            scene_option,
            camera,
        )


def generate_variability_video(
    project_folder: Text,
    variability,
    frames: np.ndarray,
    save_path: Text,
    model_data_path: Text = None,
    camera: Text = "walker/close_profile",  # "Camera1",
    segmented: bool = False,
    height: int = 1200,
    width: int = 1920,
    registration_xml: bool = False,
):
    param_path, calibration_path, _, offset_path = get_project_paths(
        project_folder, camera
    )
    params = util.load_params(param_path)
    params["n_frames"] = len(frames) - 1
    params[
        "XML_PATH"
    ] = "/n/home02/daldarondo/LabDir/Diego/code/dm/stac/models/rodent_variability.xml"
    camparams = view_stac.load_calibration(calibration_path)
    camera_kwargs = convert_cameras(camparams)

    _, offsets, kp_data, _, q_names = view_stac.load_data(
        offset_path, return_q_names=True
    )
    qpos = load_reconstruction(model_data_path)
    qpos = qpos[frames, ...].copy()
    kp_data = kp_data[frames, ...].copy()
    variability = variability[frames, ...].copy()
    qpos = view_stac.fix_tail(qpos, q_names)
    env = rodent_environments.rodent_variability(
        kp_data, variability, params, alpha=0.0
    )
    view_stac.setup_sites(qpos, offsets, env)
    env.task.render_video = True
    env.task.initialize_episode(env.physics, 0)
    scene_option = view_stac.setup_variability_scene()
    view_stac.mujoco_loop(
        save_path,
        params,
        env,
        scene_option,
        camera=camera,
        height=height,
        width=width,
    )


def setup_video_scene(
    project_folder: Text,
    model_data_path: Text,
    frames: np.ndarray,
    camera: Text,
    segmented: bool,
    use_stac: bool,
    registration_xml: bool,
    video_type: Text,
) -> Tuple:
    """[summary]

    Args:
        project_folder (Text): Path to project folder
        model_data_path (Text): Path to comic embedding
        frames (np.ndarray): Frames to include in video
        camera (Text): Camera to render mujoco
        segmented (bool): If True, segment the background.
        use_stac (bool): If True, use the stac registration.
        registration_xml (bool): If True, use the registration xml
        video_type (Text): Video type. Can be ["overlay", "mujoco"].

    Returns:
        Tuple: params, env, scene_option, video_path, calibration_path
    """
    param_path, calibration_path, video_path, offset_path = get_project_paths(
        project_folder, camera
    )
    params = view_stac.load_calibration(calibration_path)
    camera_kwargs = convert_cameras(params)

    qpos, offsets, kp_data, _, q_names = view_stac.load_data(
        offset_path, return_q_names=True
    )
    if not use_stac:
        qpos = load_reconstruction(model_data_path)

    # Crop data to frames
    # if video_type == "overlay":
    #     qpos[:, 2] -= Z_OFFSET

    # qpos[:, 2] -= .01
    qpos = qpos[frames, ...].copy()
    qpos = view_stac.fix_tail(qpos, q_names)
    kp_data = np.zeros((qpos.shape[0], kp_data.shape[1]))
    n_frames = len(frames)

    # Prepare the environment
    params, env, scene_option = view_stac.setup_visualization(
        param_path,
        qpos,
        offsets,
        kp_data,
        n_frames,
        render_video=True,
        segmented=segmented,
        camera_kwargs=camera_kwargs,
        registration_xml=registration_xml,
    )

    return params, env, scene_option, video_path, calibration_path


def setup_social_video_scene(
    project_folder: Text,
    model_data_path: Text,
    frames: np.ndarray,
    camera: Text,
    segmented: bool,
    use_stac: bool,
    registration_xml: bool,
    video_type: Text,
) -> Tuple:
    """[summary]

    Args:
        project_folder (Text): Path to project folder
        model_data_path (Text): Path to comic embedding
        frames (np.ndarray): Frames to include in video
        camera (Text): Camera to render mujoco
        segmented (bool): If True, segment the background.
        use_stac (bool): If True, use the stac registration.
        registration_xml (bool): If True, use the registration xml
        video_type (Text): Video type. Can be ["overlay", "mujoco"].

    Returns:
        Tuple: params, env, scene_option, video_path, calibration_path
    """
    param_path, calibration_path, video_path, offset_paths = get_social_project_paths(
        project_folder, camera
    )
    params = view_stac.load_calibration(calibration_path)
    camera_kwargs = convert_cameras(params)

    # STOPPED HERE  1/5/2023
    qpos = []
    qpos, offsets, kp_data, _, q_names = view_stac.load_data(
        offset_path, return_q_names=True
    )

    # Crop data to frames
    # if video_type == "overlay":
    #     qpos[:, 2] -= Z_OFFSET

    # qpos[:, 2] -= .01
    # qpos = qpos[frames, ...].copy()
    qpos = view_stac.fix_tail(qpos, q_names)
    kp_data = np.zeros((qpos.shape[0], kp_data.shape[1]))
    n_frames = len(frames)

    # Prepare the environment
    params, env, scene_option = view_stac.setup_visualization(
        param_path,
        qpos,
        offsets,
        kp_data,
        n_frames,
        render_video=True,
        segmented=segmented,
        camera_kwargs=camera_kwargs,
        registration_xml=registration_xml,
    )

    return params, env, scene_option, video_path, calibration_path


def convert_cameras(params) -> List[Dict]:
    """Convert cameras from matlab convention to mujoco convention.

    Args:
        params ([type]): Camera parameters structure

    Returns:
        List[Dict]: List of dicts containing kwargs for mujoco camera addition through worldbody.
    """
    camera_kwargs = []
    for i, cam in enumerate(params):
        # Matlab camera X faces the opposite direction of mujoco X
        rot = R.from_matrix(cam.r.T)
        eul = rot.as_euler("zyx")
        eul[2] += np.pi
        modified_rot = R.from_euler("zyx", eul)
        quat = modified_rot.as_quat()

        # Convert the quaternion convention from scipy.spatial.transform.Rotation to mujoco.
        quat = quat[np.array([3, 0, 1, 2])]
        quat[0] *= -1

        # The y field of fiew is a function of the focal y and the image height.
        fovy = 2 * np.arctan(HEIGHT / (2 * cam.K[1, 1])) / (2 * np.pi) * 360
        camera_kwargs.append(
            {
                "name": "Camera%d" % (i + 1),
                "pos": -cam.t @ cam.r.T / 1000,
                "fovy": fovy,
                "quat": quat,
            }
        )
    return camera_kwargs


def submit_render_mujoco_videos():
    """Helper function for batch rendering of mujoco videos for brady movies."""
    with open("_parameters.p", "rb") as file:
        in_dict = pickle.load(file)
    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    bout = in_dict["bouts"][job_id]
    beh = in_dict["names"][job_id]
    bout_id = in_dict["bout_ids"][job_id]
    project_folder = in_dict["project_folder"]
    model_data_path = in_dict["model_data_path"]

    if len(bout) > 0:
        bout_folder = "examples/%s" % (beh)
        os.makedirs(bout_folder, exist_ok=True)
        generate_video(
            project_folder,
            bout.squeeze(),
            "examples/%s/bout_%d.mp4" % (beh, bout_id),
            "mujoco",
            camera="walker/close_profile",
            height=608,
            width=608,
            segmented=True,
            model_data_path=model_data_path,
        )

    if len(bout) > 0:
        bout_folder = "examples/%s" % (beh)
        os.makedirs(bout_folder, exist_ok=True)
        generate_video(
            project_folder,
            bout.squeeze(),
            "examples/%s/overlay_%d.mp4" % (beh, bout_id),
            "overlay",
            model_data_path=model_data_path,
        )


def submit_render_tile_videos():
    """Helper function for batch rendering of brady movies."""
    with open("_parameters.p", "rb") as file:
        in_dict = pickle.load(file)
    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    beh_name = in_dict["behavior_names"][job_id]
    generate_tile_video("examples/%s" % (beh_name))


def tile_frame(videos: List[np.ndarray], n_rows: int, n_frame: int) -> np.ndarray:
    """Tile a single frame for brady movies.

    Args:
        videos (np.ndarray): List of movies
        n_rows (int): Number of rows to make brady movie
        n_frame (int): Frame number to make frame.

    Returns:
        np.ndarray: Tiled frame
    """
    h, w = videos[0].shape[0:2]
    frame = np.zeros((h * n_rows, w * n_rows, 3), dtype=np.uint8)
    for i in range(n_rows):
        for j in range(n_rows):
            try:
                vid = videos[i * n_rows + j]

            # If the video doesn't exist, just keep it black.
            except IndexError:
                vid = np.zeros((h, w, 3, 1), dtype=np.uint8)
            block_frame = np.mod(n_frame, vid.shape[3])
            block = vid[:, :, :, block_frame]
            frame[i * h : (i + 1) * h, j * w : (j + 1) * w, ...] = block
    return frame


def generate_tile_video(video_folder: Text, n_tiles: int = 25, n_frames: int = 500):
    """Write a tile video to disk

    Args:
        video_folder (Text): Path to video folder containing individual bouts.
        n_tiles (int, optional): Number of tiles to include in video. Defaults to 25.
        n_frames (int, optional): Total number of frames in video. Defaults to 500.
    """
    files = [
        os.path.join(video_folder, f) for f in os.listdir(video_folder) if "bout_" in f
    ]
    sorting = np.argsort([int(f.split("_")[-1].split(".mp4")[0]) for f in files])
    files = [files[i] for i in sorting]
    n_files = len(files)
    # include = np.random.choice(n_files, np.min([n_tiles, n_files]), replace=False)
    include = np.arange(np.min([n_tiles, n_files]))
    files = [files[i] for i in include]

    # Load videos in memory
    videos = []
    for f in files:
        reader = imageio.get_reader(f)
        vid = [frame for frame in reader]
        vid = vid[2:-2]
        videos.append(np.stack(vid, axis=3).astype("uint8"))

    # Tile videos
    with imageio.get_writer(os.path.join(video_folder, "tile.mp4"), fps=50) as video:
        for n_frame in range(n_frames):
            frame = tile_frame(videos, int(np.sqrt(n_tiles)), n_frame)
            video.append_data(frame)
