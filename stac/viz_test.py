import unittest
import stac.viz as viz
import stac.util as util
import os
import numpy as np

""" Function signatures for viz.py
generate_tile_video(video_folder:str, n_tiles:int=25, n_frames:int=500)
get_social_project_paths(project_folder:str, camera:str) -> Tuple[str, str, str, str]
setup_social_video_scene(project_folder:str, model_data_path:str, frames:numpy.ndarray, camera:str, segmented:bool, use_stac:bool, registration_xml:bool, video_type:str) -> Tuple
tile_frame(videos:List[numpy.ndarray], n_rows:int, n_frame:int) -> numpy.ndarray
trim_video(video_path:str, save_path:str, frames:numpy.ndarray)
"""


PROJECT_FOLDER = (
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1"
)
MODEL_DATA_PATH = os.path.join(
    PROJECT_FOLDER, "npmp", "rodent_tracking_model_24186410_2", "logs", "data.hdf5"
)
param_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params", "params.yaml"
)
results_path = os.path.join(PROJECT_FOLDER, "stac", "total.p")

calibration_path = os.path.join(
     os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "calibration.mat"
)
video_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "videos", "Camera1", "0.mp4"
)

class TestViz(unittest.TestCase):
    def test_convert_cameras(self):
        calibration_path = os.path.join(PROJECT_FOLDER, "temp_dannce.mat")
        camparams = util.loadmat(calibration_path)["params"]
        camera_kwargs = viz.convert_cameras(camparams)
        self.assertEqual(len(camera_kwargs), 6)

    def test_render_mujoco(self):
        save_path = "test.mp4"
        viz.render_mujoco(
            param_path,
            results_path,
            save_path,
            frames=np.arange(10),
        )
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_render_overlay(self):
        save_path = "test.mp4"
        viz.render_overlay(
            param_path,
            video_path,
            results_path,
            save_path,
            frames=np.arange(10),
            camera="Camera1",
            calibration_path=calibration_path,
            segmented=True,
        )
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)


if __name__ == "__main__":
    unittest.main()
