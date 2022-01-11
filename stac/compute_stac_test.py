from absl.testing import absltest
import stac.compute_stac as cs

DATA_PATH = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2/DANNCE/predict02/save_data_AVG.mat"
PARAM_PATH = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2/stac_params/params.yaml"
OFFSET_PATH = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2/stac/offset.p"
SAVE_PATH = "/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/stac/test/test.p"
START_FRAME = 0
END_FRAME = 50
N_TEST_KEYPOINTS = 23


class StacTest(absltest.TestCase):
    def setUp(self):
        self.stac = cs.STAC(
            DATA_PATH,
            PARAM_PATH,
            save_path=SAVE_PATH,
            start_frame=START_FRAME,
            end_frame=END_FRAME,
            verbose=True,
        )

    def test_load_dataset(self):
        params = self.stac.params
        kp_data, _ = cs.preprocess_data(
            params["data_path"],
            params["start_frame"],
            params["end_frame"],
            params["skip"],
            params,
        )
        self.assertEqual(kp_data.shape[0], params["end_frame"] - params["start_frame"])
        self.assertEqual(kp_data.shape[1], N_TEST_KEYPOINTS * 3)

    def test_stac_fit(self):
        data = self.stac.fit()

    def test_stac_transform(self):
        data = self.stac.transform(OFFSET_PATH)

    def test_stac_save(self):
        data = self.stac.transform(OFFSET_PATH)
        self.stac.save(data)


if __name__ == "__main__":
    absltest.main()
