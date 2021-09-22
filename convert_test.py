import os
from absl.testing import absltest
import convert

test_file = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac/total.p"
offset_path = "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM25.p"
params_path_target = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac_params/params_conversion.yaml"
params_path_source = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac_params/params.yaml"
save_path = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/mocap_conversion/data.mat"


class ConvertTest(absltest.TestCase):
    # def test_convert(self):
    #     convert.convert(
    #         test_file,
    #         offset_path,
    #         params_path_source,
    #         params_path_target,
    #         save_path,
    #         n_frames=100,
    #     )
    def test_parallel_convert(self):
        stac_paths = [
            "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac/total.p"
        ]
        for path in stac_paths:
            pc = convert.ParallelConverter(path, test=True)
            pc.submit()


if __name__ == "__main__":
    absltest.main()
