import os
from absl.testing import absltest
import convert

test_file = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac/total.p"
offset_path = "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM25.p"
params_path_target = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac_params/params_conversion.yaml"
params_path_source = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac_params/params.yaml"
save_path = "./test/data.mat"


test_comic_file = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/npmp/rodent_tracking_model_21380833_2_final/logs/data.hdf5"
comic_offset_path = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac/offset.p"
comic_params_path = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac_params/params.yaml"
project_folder = (
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1"
)


class ConvertTest(absltest.TestCase):
    def test_convert(self):
        convert.convert(
            test_file,
            offset_path,
            params_path_source,
            params_path_target,
            save_path,
            start_frame=0,
            end_frame=100,
        )

    def test_comic_convert(self):
        convert.convert(
            test_comic_file,
            comic_offset_path,
            comic_params_path,
            comic_params_path,
            save_path,
            start_frame=0,
            end_frame=100,
        )

    def test_parallel_convert(self):
        stac_paths = [
            "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac/total.p"
        ]
        for path in stac_paths:
            pc = convert.ParallelConverter(path, project_folder, test=True)
            pc.submit()

    def test_parallel_comic_convert(self):
        stac_paths = [test_comic_file]
        for path in stac_paths:
            pc = convert.ComicParallelConverter(path, project_folder, test=True)
            pc.submit()


if __name__ == "__main__":
    absltest.main()
