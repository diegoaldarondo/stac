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
import yaml
import subprocess

_M_TO_MM = 1000
N_MOCAP_DIMS = 60
N_DANNCE_DIMS = 69
MOCAP_OFFSET_PATH = (
    "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM25.p"
)
PROJECT_FOLDERS = [
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2",
    "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_21_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_28_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_28_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_01_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/duke/2022_02_16_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/duke/2022_02_25_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/espie/2022_03_10_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/espie/2022_03_15_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_16_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_24_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/gerry/2022_05_30_1",
    # "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_03_1",
]

CONVERT_PROJECT_SCRIPT = (
    lambda n_array, project_folder, comic_model: f"""#!/bin/bash
#SBATCH -J convert_mocap
#SBATCH --array=0-{n_array}
#SBATCH -N 1                # number of nodes
#SBATCH -c 1               # Number of threads (cores)
#SBATCH -p olveczky,shared,serial_requeue,cox # Number of threads (cores)
#SBATCH --mem 5000        # memory for all cores
#SBATCH -t 1-00:00          # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o logs/Job.convert_mocap.%N.%j.out    # STDOUT
#SBATCH -e logs/Job.convert_mocap.%N.%j.err    # STDERR
#SBATCH --constraint="intel&avx2"
source ~/.bashrc
setup_mujoco210_3.7
python -c "import convert; convert.submit_project('{project_folder}', '{comic_model}')"
"""
)


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


def submit_project(pf, comic_model):
    """Run one batch job on the cluster."""
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    with open(os.path.join(pf, f"_convert_parameters_{comic_model}.p"), "rb") as file:
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


def submit_comic_convert_project(project_folder: Text, comic_model: Text):
    """Submit conversion jobs to the cluster for the sessions in PROJECT_FOLDERS/npmp/comic_model.

    Args:
        project_folder (Text): Path to the project folder to convert.
        comic_model (Text): name of the comic model data to convert.
    """
    data_path = os.path.join(project_folder, "npmp", comic_model, "logs", "data.hdf5")
    pc = ComicParallelConverter(
        data_path,
        project_folder,
        offset_path=os.path.join(project_folder, "stac", "offset.p"),
        save_folder=os.path.join("npmp", comic_model, "kp_conversion"),
    )
    pc.params_path_target = pc.params_path_source
    pc.batches = pc.generate_batches()
    jobs = pc.batches

    with open(
        os.path.join(project_folder, f"_convert_parameters_{comic_model}.p"), "wb"
    ) as file:
        pickle.dump(jobs, file)
    # cmd = (
    #     f"sbatch --array=0-{len(jobs) - 1} cluster/convert_project.sh {project_folder}"
    # )
    # print(cmd)
    script = CONVERT_PROJECT_SCRIPT(len(jobs) - 1, project_folder, comic_model)
    output = subprocess.check_output("sbatch", input=script, universal_newlines=True)
    job_id = output.strip().split()[-1]
    return job_id


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
    # for job in jobs:
    #     print(job["stac_path"])
    cmd = "sbatch --array=0-%d cluster/convert.sh" % (len(jobs) - 1)
    os.system(cmd)


def submit_project_convert(project_folder: Text):
    """Submit conversion jobs to the cluster a project_folder."""
    stac_path = os.path.join(project_folder, "stac", "total.p")
    all_batches = []
    pc = ParallelConverter(stac_path, project_folder)
    pc.batches = pc.generate_batches()
    all_batches.append(pc.batches.copy())
    jobs = [job for batch in all_batches for job in batch]
    with open(os.path.join(project_folder, "_convert_parameters.p"), "wb") as file:
        pickle.dump(jobs, file)
    print(len(jobs))
    cmd = (
        "sbatch --wait --array=0-%d /n/holylfs02/LABS/olveczky_lab/Diego/code/dm/stac/cluster/convert_project.sh %s"
        % (len(jobs) - 1, project_folder)
    )
    os.system(cmd)


def merge_mocap_conversion():
    """Merge the mocap_conversion sessions in PROJECT_FOLDERS."""
    mocap_conversion_paths = [
        os.path.join(pf, "mocap_conversion") for pf in PROJECT_FOLDERS
    ]
    for path in mocap_conversion_paths:
        print(path, flush=True)
        merge(path)


def merge_mocap_conversion_project(project_folder: Text):
    """Merge the mocap_conversion sessions in PROJECT_FOLDERS."""
    mocap_conversion_path = os.path.join(project_folder, "mocap_conversion")
    merge(mocap_conversion_path)


def merge_comic_conversion(comic_model: Text):
    """Merge the mocap_conversion sessions in PROJECT_FOLDERS."""
    mocap_conversion_paths = [
        os.path.join(pf, "npmp", comic_model, "kp_conversion") for pf in PROJECT_FOLDERS
    ]
    for path in mocap_conversion_paths:
        print(path, flush=True)
        merge(path)


def merge_comic_conversion_project(project_folder: Text, comic_model: Text):
    """Merge the mocap_conversion sessions in PROJECT_FOLDERS."""
    path = os.path.join(project_folder, "npmp", comic_model, "kp_conversion")
    merge(path)


def submit_noise_analysis_conversion(config_path: str):
    """Submit conversion jobs to the cluster for the sessions in the noise analysis config"

    Args:
        config_path (str): Path to the noise_analysis config.
    """
    # Load the yaml config file
    with open(config_path, "rb") as file:
        cfg = yaml.safe_load(file)
    model = "rodent_tracking_model_24189285_2"
    all_batches = []
    for noise_type in cfg["latent_noise"]:
        for noise_gain in cfg["noise_gain"]:
            comic_model = f"noise_analysis/{model}/{noise_type}{noise_gain}"
            data_paths = [
                os.path.join(pf, "npmp", comic_model, "logs", "data.hdf5")
                for pf in PROJECT_FOLDERS
            ]
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


def merge_noise_analysis_conversion(config_path: str):
    """Merge the mocap_conversion sessions in PROJECT_FOLDERS."""
    with open(config_path, "rb") as file:
        cfg = yaml.safe_load(file)
    model = "rodent_tracking_model_24189285_2"
    for noise_type in cfg["latent_noise"]:
        for noise_gain in cfg["noise_gain"]:
            comic_model = f"noise_analysis/{model}/{noise_type}{noise_gain}"
            merge_comic_conversion(comic_model)
