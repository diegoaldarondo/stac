import unittest
import viz
import view_stac
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


class TestViz(unittest.TestCase):
    def test_add_qualifier(self):
        video_path = "test.mp4"
        qualifier = "test"
        self.assertEqual(viz.add_qualifier(video_path, qualifier), "testtest.mp4")

    def test_convert_cameras(self):
        calibration_path = os.path.join(PROJECT_FOLDER, "temp_dannce.mat")
        camparams = view_stac.load_calibration(calibration_path)
        camera_kwargs = viz.convert_cameras(camparams)
        self.assertEqual(len(camera_kwargs), 6)

    def test_get_project_paths(self):
        camera = "Camera1"
        param_path, calibration_path, video_path, offset_path = viz.get_project_paths(
            PROJECT_FOLDER, camera
        )
        self.assertEqual(
            param_path, os.path.join(PROJECT_FOLDER, "stac_params", "params.yaml")
        )
        self.assertEqual(
            calibration_path, os.path.join(PROJECT_FOLDER, "temp_dannce.mat")
        )
        self.assertEqual(
            video_path, os.path.join(PROJECT_FOLDER, "videos", "Camera1", "0.mp4")
        )
        self.assertTrue(
            (offset_path == os.path.join(PROJECT_FOLDER, "stac", "total.p"))
            or (offset_path == os.path.join(PROJECT_FOLDER, "stac", "offset.p"))
        )

    def test_setup_video_scene(self):
        camera = "Camera1"
        video_type = "mujoco"
        frames = [0, 1, 2]
        scene = viz.setup_video_scene(
            PROJECT_FOLDER,
            MODEL_DATA_PATH,
            frames,
            camera,
            False,
            False,
            False,
            video_type,
        )
        self.assertIsInstance(scene, tuple)

    def test_generate_video_mujoco(self):
        camera = "Camera1"
        video_type = "mujoco"
        frames = [0, 1, 2]
        save_path = "test.mp4"
        viz.generate_video(
            PROJECT_FOLDER,
            frames,
            save_path,
            video_type,
            MODEL_DATA_PATH,
            camera,
            False,
            False,
            1200,
            1920,
            False,
        )
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_generate_video_mujoco_segmented(self):
        camera = "Camera1"
        video_type = "mujoco"
        frames = [0, 1, 2]
        save_path = "test.mp4"
        viz.generate_video(
            PROJECT_FOLDER,
            frames,
            save_path,
            video_type,
            MODEL_DATA_PATH,
            camera,
            False,
            True,
            1200,
            1920,
            False,
        )
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_generate_video_overlay(self):
        camera = "Camera1"
        video_type = "overlay"
        frames = [0, 1, 2]
        save_path = "test.mp4"
        viz.generate_video(
            PROJECT_FOLDER,
            frames,
            save_path,
            video_type,
            MODEL_DATA_PATH,
            camera,
            False,
            False,
            1200,
            1920,
            False,
        )
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_generate_video_mujoco_stac(self):
        camera = "Camera1"
        video_type = "mujoco"
        frames = [0, 1, 2]
        save_path = "test.mp4"
        viz.generate_video(
            PROJECT_FOLDER,
            frames,
            save_path,
            video_type,
            MODEL_DATA_PATH,
            camera,
            True,
            False,
            1200,
            1920,
            True,
        )
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_generate_video_overlay_stac(self):
        camera = "Camera1"
        video_type = "overlay"
        frames = [0, 1, 2]
        save_path = "test.mp4"
        viz.generate_video(
            PROJECT_FOLDER,
            frames,
            save_path,
            video_type,
            MODEL_DATA_PATH,
            camera,
            True,
            False,
            1200,
            1920,
            True,
        )
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_generate_variability_video(self):
        frames = [0, 1, 2]
        variability = np.random.rand(3, 38)
        save_path = "test.mp4"
        viz.generate_variability_video(
            PROJECT_FOLDER,
            variability,
            frames,
            save_path,
            MODEL_DATA_PATH,
            segmented=False,
            height=1200,
            width=1920,
            registration_xml=False,
        )
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_tile_frame(self):
        videos = [np.random.rand(10, 10, 3, 10) for _ in range(4)]
        frame = viz.tile_frame(videos, n_rows=2, n_frame=4)
        self.assertEqual(frame.shape, (20, 20, 3))


if __name__ == "__main__":
    unittest.main()
