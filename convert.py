"""Module to convert qpos files to sites for arbitrary marker sets."""
import os
import pickle
from typing import Text, Dict, List, Tuple
import numpy as np
from scipy.io import savemat, loadmat
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco.wrapper.mjbindings import mjlib
from stac import util
from stac import rodent_environments
import h5py
import time

_M_TO_MM = 1000
N_MOCAP_DIMS = 60
N_DANNCE_DIMS = 69
MOCAP_OFFSET_PATH = (
    "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM25.p"
)
PROJECT_FOLDERS = [
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_3",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_2",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_3",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_1",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_2",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_3",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_25_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_25_3",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_3",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_2",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_3",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_3",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_29_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_29_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_30_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_30_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_31_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_31_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_01_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_01_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_02_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_02_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_04_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_04_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_05_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_05_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_06_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_06_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_07_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_07_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_08_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_09_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_21_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_21_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_22_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_22_2",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_23_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_24_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_25_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_26_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_28_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_29_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_30_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_01_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_02_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_03_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_05_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_06_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_07_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_08_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_10_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_11_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_12_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_13_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_14_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_15_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_16_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_17_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_18_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_19_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_28_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_29_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_30_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_31_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_01_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_02_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_03_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_04_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_05_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_06_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_07_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_08_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_09_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_10_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_11_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_12_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_13_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_14_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_15_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_16_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_17_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_18_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_19_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_20_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_21_1",
]


def load_data(file_path: Text) -> Dict:
    """Load data from a chunk

    Args:
        file_path (Text): Path to a qpos pickle file.

    Returns:
        Dict: Model registration dictionary.
    """
    with open(file_path, "rb") as file:
        in_dict = pickle.load(file)
    return in_dict


def load_comic_data(file_path: Text) -> Dict:
    with h5py.File(file_path, "r") as file:
        q = file["qpos"][:].astype("float")
        data = {
            "qpos": q,
            "kp_data": np.zeros((q.shape[0], N_DANNCE_DIMS)),
        }
    return data


def convert(
    file_path: Text,
    offset_path: Text,
    params_path_source: Text,
    params_path_target: Text,
    save_path: Text,
    start_frame: int = 0,
    end_frame: int = -1,
):
    """Convert qpos to the desired keypoint representation

    Args:
        file_path (Text): Path to the qpos pickle file.
        offset_path (Text): Path to the offset pickle file.
        params_path_source (Text): Path the the stac params .yaml file.
        params_path_target (Text): Path the the stac params .yaml file.
        save_path (Text): Path the the output .mat file.
        start_frame (int): Frame to start converting.
        end_frame (int): Frame to finish converting.
        n_frames (int, optional): Number of frames to convert. Default all.
    """
    print("file_path: ", file_path, flush=True)
    print("offset_path: ", offset_path, flush=True)
    print("params_path_source: ", params_path_source, flush=True)
    print("params_path_target: ", params_path_target, flush=True)
    print("save_path: ", save_path, flush=True)

    params, data = load(file_path, params_path_source, params_path_target)

    if end_frame == -1:
        params["n_frames"] = data["qpos"].shape[0]
    else:
        params["n_frames"] = int(end_frame - start_frame)
    print(data["qpos"].shape)
    data["qpos"] = data["qpos"][start_frame:end_frame, ...]
    print(data["kp_data"].shape)
    data["kp_data"] = data["kp_data"][start_frame:end_frame, ...]

    env = setup_environment(
        params,
        data,
    )

    # Load the offsets and set the sites to those positions.
    with open(offset_path, "rb") as file:
        in_dict = pickle.load(file)

    sites = env.task._walker.body_sites
    env.physics.bind(sites).pos[:] = in_dict["offsets"]
    for n_site, pos in enumerate(env.physics.bind(sites).pos):
        sites[n_site].pos = pos

    # Set the environment with the qpos.
    env.task.precomp_qpos = data["qpos"]

    env.task.initialize_episode(env.physics, 0)
    prev_time = env.physics.time()

    sites = np.copy(env.physics.bind(env.task._walker.body_sites).xpos[:])

    # Loop through the clip, saving the sites on each frame.
    n_frame = 0
    walker_body_sites = np.zeros((params["n_frames"], sites.shape[0], sites.shape[1]))
    while prev_time < env._time_limit:
        while (np.round(env.physics.time() - prev_time, decimals=5)) < params[
            "_TIME_BINS"
        ]:
            env.physics.step()
        env.task.after_step(env.physics, None)
        print(env.task.frame)
        print(n_frame)
        walker_body_sites[n_frame, ...] = np.copy(
            env.physics.bind(env.task._walker.body_sites).xpos[:]
        )
        if (
            n_frame == params["n_frames"] - 1
            or env.task.frame == params["n_frames"] - 1
        ):
            print(n_frame, flush=True)
            break
        n_frame += 1
        prev_time = np.round(env.physics.time(), decimals=2)

    # Save the results
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Reformatting. Last frame is usually zero because of physics rollout.
    walker_body_sites = np.transpose(walker_body_sites, (0, 2, 1)) * _M_TO_MM
    walker_body_sites[-1, ...] = walker_body_sites[-2, ...]
    savemat(save_path, {"walker_body_sites": walker_body_sites})


def load(
    file_path: Text, params_path_source: Text, params_path_target: Text
) -> Tuple[Dict, Dict]:
    """Load parameters and data

    Args:
        file_path (Text): Path to the qpos pickle file.
        params_path_source (Text): Path the the stac params .yaml file.
        params_path_target (Text): Path the the stac params .yaml file.

    Returns:
        Tuple[Dict, Dict]: Parameters, data
    """
    # Load the parameters and data
    params_source = util.load_params(params_path_source)
    params = util.load_params(params_path_target)
    params["scale_factor"] = params_source["scale_factor"]
    print(os.path.splitext(file_path)[1])
    if os.path.splitext(file_path)[1] == ".hdf5":
        data = load_comic_data(file_path)
        params["n_kp_dims"] = N_DANNCE_DIMS
    else:
        data = load_data(file_path)
        params["n_kp_dims"] = N_MOCAP_DIMS
    return params, data


def setup_environment(params: Dict, data: Dict):
    """Set up environment to convert keypoint sets.

    Args:
        params (Dict): Stac parameters dictionary
        data (Dict): Stac data dictionary

    Returns:
        [type]: Dm control rodent environment
    """
    # Build the environment
    env = rodent_environments.rodent_mocap(
        data["kp_data"][:, : params["n_kp_dims"]],
        params,
        arena_diameter=params["_ARENA_DIAMETER"],
        arena_center=params["_ARENA_CENTER"],
    )
    rescale.rescale_subtree(
        env.task._walker._mjcf_root,
        params["scale_factor"],
        params["scale_factor"],
    )
    mjlib.mj_kinematics(env.physics.model.ptr, env.physics.data.ptr)
    # Center of mass position
    mjlib.mj_comPos(env.physics.model.ptr, env.physics.data.ptr)
    env.reset()
    return env


def submit():
    """Run one batch job on the cluster."""
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    with open("_parameters.p", "rb") as file:
        parameters = pickle.load(file)
    params = parameters[task_id]

    convert(
        params["stac_path"],
        params["offset_path"],
        params["params_path_source"],
        params["params_path_target"],
        params["save_path"],
        start_frame=params["start_frame"],
        end_frame=params["end_frame"],
    )


class ParallelConverter:
    """Convert keypoint sets using parallel chunks."""

    def __init__(
        self,
        stac_path: Text,
        project_folder: Text,
        offset_path: Text = MOCAP_OFFSET_PATH,
        n_samples_per_job: int = 5000,
        save_folder: Text = "mocap_conversion",
        test: bool = False,
    ):
        """Initialize ParallelConverter

        Args:
            stac_path (Text): Path to stac file.
            offset_path (Text, optional): Path to offset file. Defaults to MOCAP_OFFSET_PATH.
            n_samples_per_job (int, optional): Number of samples per chunk. Defaults to 5000.
            test (bool, optional): If True, test the converter. Defaults to False.
            save_folder (Text, optional): Name of folder (in project folder) in which to save results.
                                          Defaults to "mocap_conversion"
        """
        self.stac_path = stac_path
        self.offset_path = offset_path
        self.save_folder = save_folder
        self.test = test
        self.project_folder = project_folder
        self.params_path_source = os.path.join(
            self.project_folder, "stac_params", "params.yaml"
        )
        self.params_path_target = os.path.join(
            self.project_folder, "stac_params", "params_conversion.yaml"
        )
        self.n_samples_per_job = n_samples_per_job
        data = self._load()
        self.n_frames = data["qpos"].shape[0]

    def _load(self) -> Dict:
        """Helper to load data

        Returns:
            Dict: Data
        """
        return load_data(self.stac_path)

    def generate_batches(self) -> List[Dict]:
        """Generate batches parameters.

        Returns:
            List[Dict]: List of dictionaries containing batch submission parameters.
        """
        start_frames = np.arange(0, self.n_frames, self.n_samples_per_job)
        end_frames = start_frames + self.n_samples_per_job
        end_frames[-1] = np.min([end_frames[-1], self.n_frames])

        batches = []
        for i in range(start_frames.shape[0]):
            batch = {
                "stac_path": self.stac_path,
                "offset_path": self.offset_path,
                "params_path_source": self.params_path_source,
                "params_path_target": self.params_path_target,
                "save_path": os.path.join(
                    self.project_folder,
                    self.save_folder,
                    str(start_frames[i]) + ".mat",
                ),
                "start_frame": start_frames[i],
                "end_frame": end_frames[i],
            }
            batches.append(batch)
        return batches

    def submit(self):
        if self.test:
            cmd = "sbatch --array=0 cluster/convert.sh"
            print(cmd)
        else:
            cmd = "sbatch --array=0-%d cluster/convert.sh" % (len(self.batches) - 1)
            os.system(cmd)


class ComicParallelConverter(ParallelConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self):
        return load_comic_data(self.stac_path)


def merge(mocap_conversion_path: Text):
    """Merge the mocap_conversion for a single project folder.

    Args:
        mocap_conversion_path (Text): Path to mocap conversion folder
    """
    files = [file for file in os.listdir(mocap_conversion_path) if ".mat" in file]
    files = [file for file in files if "data" not in file]
    indices = [int(file.split(".mat")[0]) for file in files]
    I = np.argsort(indices)
    files = [os.path.join(mocap_conversion_path, files[i]) for i in I]

    keypoints = []
    for file in files:
        data = loadmat(file)
        keypoints.append(data["walker_body_sites"].copy())

    keypoints = np.concatenate(keypoints)
    savemat(
        os.path.join(mocap_conversion_path, "data.mat"),
        {"walker_body_sites": keypoints},
    )


def submit_comic_convert(comic_model: Text):
    """Submit conversion jobs to the cluster for the sessions in PROJECT_FOLDERS/npmp/comic_model.

    Args:
        comic_model (Text): name of the comic model data to convert.
    """

    data_paths = [
        os.path.join(pf, "npmp", comic_model, "logs", "data.hdf5")
        for pf in PROJECT_FOLDERS
    ]
    all_batches = []
    for path, pf in zip(data_paths, PROJECT_FOLDERS):
        pc = ComicParallelConverter(
            path,
            pf,
            offset_path=os.path.join(pf, "stac", "offset.p"),
            save_folder=os.path.join("npmp", comic_model, "kp_conversion"),
        )
        pc.params_path_target = pc.params_path_source
        pc.batches = pc.generate_batches()
        all_batches.append(pc.batches.copy())
    jobs = [job for batch in all_batches for job in batch]
    with open("_parameters.p", "wb") as file:
        pickle.dump(jobs, file)
    print(len(jobs))
    for job in jobs:
        print(job["stac_path"])
    cmd = "sbatch --array=0-%d cluster/convert.sh" % (len(jobs) - 1)
    print(cmd)
    os.system(cmd)


def submit_convert():
    """Submit conversion jobs to the cluster for the sessions in PROJECT_FOLDERS."""
    stac_paths = [os.path.join(pf, "stac", "total.p") for pf in PROJECT_FOLDERS]
    all_batches = []
    for path, pf in zip(stac_paths, PROJECT_FOLDERS):
        pc = ParallelConverter(path, pf)
        pc.batches = pc.generate_batches()
        all_batches.append(pc.batches.copy())
    jobs = [job for batch in all_batches for job in batch]
    with open("_parameters.p", "wb") as file:
        pickle.dump(jobs, file)
    print(len(jobs))
    for job in jobs:
        print(job["stac_path"])
    cmd = "sbatch --array=0-%d cluster/convert.sh" % (len(jobs) - 1)
    os.system(cmd)


def merge_mocap_conversion():
    """Merge the mocap_conversion sessions in PROJECT_FOLDERS."""
    mocap_conversion_paths = [
        os.path.join(pf, "mocap_conversion") for pf in PROJECT_FOLDERS
    ]
    for path in mocap_conversion_paths:
        print(path, flush=True)
        merge(path)


def merge_comic_conversion(comic_model: Text):
    """Merge the mocap_conversion sessions in PROJECT_FOLDERS."""
    mocap_conversion_paths = [
        os.path.join(pf, "npmp", comic_model, "kp_conversion") for pf in PROJECT_FOLDERS
    ]
    for path in mocap_conversion_paths:
        print(path, flush=True)
        merge(path)