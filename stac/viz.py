import pickle
import stac.view_stac as view_stac
import numpy as np
from typing import Text, List, Dict
from scipy.spatial.transform import Rotation as R
import yaml

# Standard image shape for dannce rig data
HEIGHT = 1200
WIDTH = 1920


def convert_camera(cam, idx):
    """Convert a camera from Matlab convention to Mujoco convention."""
    # Matlab camera X faces the opposite direction of Mujoco X
    rot = R.from_matrix(cam.r.T)
    eul = rot.as_euler("zyx")
    eul[2] += np.pi
    modified_rot = R.from_euler("zyx", eul)
    quat = modified_rot.as_quat()

    # Convert the quaternion convention from scipy.spatial.transform.Rotation to Mujoco.
    quat = quat[np.array([3, 0, 1, 2])]
    quat[0] *= -1
    # The y field of fiew is a function of the focal y and the image height.
    fovy = 2 * np.arctan(HEIGHT / (2 * cam.K[1, 1])) / (2 * np.pi) * 360
    return {
        "name": f"Camera{idx + 1}",
        "pos": -cam.t @ cam.r.T / 1000,
        "fovy": fovy,
        "quat": quat,
    }


def convert_cameras(params) -> List[Dict]:
    """Convert cameras from Matlab convention to Mujoco convention.

    Args:
        params: Camera parameters structure

    Returns:
        List[Dict]: List of dicts containing kwargs for Mujoco camera addition through worldbody.
    """
    camera_kwargs = [convert_camera(cam, idx) for idx, cam in enumerate(params)]
    return camera_kwargs


def render_mujoco(
    param_path: Text,
    data_path: Text,
    save_path: Text,
    frames: np.ndarray = None,
    camera: Text = "walker/close_profile",
    calibration_path: Text = None,
    segmented: bool = False,
    height: int = HEIGHT,
    width: int = WIDTH,
    xml_path: Text = None,
):
    """Render a video of a STAC dataset using mujoco.

    Args:
        param_path (Text): Path to the stac parameters file.
        data_path (Text): Path to the stac data file.
        save_path (Text): Path to save the video.
        frames (np.ndarray, optional): Frames to render. Defaults to None.
        camera (Text, optional): Camera to render. Defaults to "walker/close_profile".
        calibration_path (Text, optional): Path to the calibration file. Defaults to None.
        segmented (bool, optional): Whether to segment the rendered video. Defaults to False.
        height (int, optional): Image height. Defaults to 1200.
        width (int, optional): Image width. Defaults to 1920.
        xml_path (Text, optional): Path to the xml file. Defaults to None.
    """
    xml_path, qpos, kp_data, n_frames, offsets, camera_kwargs = load_data(
        param_path, data_path, frames, calibration_path, xml_path
    )

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
        registration_xml=xml_path,
    )
    view_stac.mujoco_loop(
        save_path,
        params,
        env,
        scene_option,
        camera=camera,
        height=height,
        width=width,
    )


def load_data(
    param_path: str,
    data_path: str,
    frames: np.ndarray,
    calibration_path: str,
    xml_path: str,
):
    # Load XML path from parameters file if not provided
    if xml_path is None:
        with open(param_path, "rb") as file:
            params = yaml.safe_load(file)
        xml_path = params["XML_PATH"]

    # Load data from the data file
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    # Process qpos data
    qpos = np.stack(data["qpos"][:], axis=0)
    q_names = data["names_qpos"][:]
    qpos = view_stac.fix_tail(qpos, q_names)

    # Filter qpos and kp_data based on provided frames
    if frames is not None:
        qpos = qpos[frames, ...].copy()
        kp_data = data["kp_data"][frames, ...].copy()
    n_frames = qpos.shape[0]
    offsets = data["offsets"]

    # Load camera parameters and convert them if a calibration path is provided
    if calibration_path is not None:
        params = view_stac.load_calibration(calibration_path)
        camera_kwargs = convert_cameras(params)
    else:
        camera_kwargs = None
    return xml_path, qpos, kp_data, n_frames, offsets, camera_kwargs


def render_overlay(
    param_path: str,
    video_path: str,
    data_path: str,
    save_path: str,
    frames=None,
    camera="Camera1",
    calibration_path=None,
    segmented=False,
    height=HEIGHT,
    width=WIDTH,
    xml_path=None,
):
    xml_path, qpos, kp_data, n_frames, offsets, camera_kwargs = load_data(
        param_path, data_path, frames, calibration_path, xml_path
    )

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
        registration_xml=xml_path,
    )
    view_stac.overlay_loop(
        save_path=save_path,
        video_path=video_path,
        calibration_path=calibration_path,
        frames=frames,
        params=params,
        env=env,
        scene_option=scene_option,
        camera=camera,
        height=height,
        width=width,
    )
