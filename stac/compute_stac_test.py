from absl.testing import absltest
import stac.compute_stac as cs
import os
import stac.util as util

DATA_PATH = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2/DANNCE/predict02/save_data_AVG.mat"
PARAM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params", "params.yaml"
)
OFFSET_PATH = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2/stac/offset.p"
SAVE_PATH = "/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/stac/test/test.p"
N_FRAMES = 3


class StacTest(absltest.TestCase):
    def setUp(self):
        self.kp_data = util.loadmat(DATA_PATH)["pred"][:] / 1000
        self.stac = cs.STAC(PARAM_PATH)

    def test_stac_fit(self):
        self.stac.fit(self.kp_data[:N_FRAMES])

    def test_stac_transform(self):
        data = self.stac.transform(self.kp_data[:N_FRAMES], OFFSET_PATH)

    def test_stac_save(self):
        self.stac.save(SAVE_PATH)
        os.remove(SAVE_PATH)


if __name__ == "__main__":
    absltest.main()
