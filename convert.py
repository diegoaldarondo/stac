"""Module to convert qpos files to sites for arbitrary marker sets."""
import os
import pickle
from typing import Text, Dict
import numpy as np
from scipy.io import savemat, loadmat
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco.wrapper.mjbindings import mjlib
from stac import util
from stac import rodent_environments
import time

_M_TO_MM = 1000
N_MOCAP_DIMS = 60
MOCAP_OFFSET_PATH = (
    "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM25.p"
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

    # Load the parameters and data
    params_source = util.load_params(params_path_source)
    params = util.load_params(params_path_target)
    params["scale_factor"] = params_source["scale_factor"]
    data = load_data(file_path)

    if end_frame == -1:
        params["n_frames"] = data["qpos"].shape[0]
    else:
        params["n_frames"] = int(end_frame - start_frame)
    print(data["qpos"].shape)
    data["qpos"] = data["qpos"][start_frame:end_frame, ...]
    print(data["kp_data"].shape)
    data["kp_data"] = data["kp_data"][start_frame:end_frame, ...]

    env = setup_environment(params, data)

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
        # print(walker_body_sites[n_frame])
        # print(n_frame, flush=True)
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
        data["kp_data"][:, :N_MOCAP_DIMS],
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
    def __init__(
        self,
        stac_path: Text,
        offset_path: Text = MOCAP_OFFSET_PATH,
        n_samples_per_job: int = 5000,
        test: bool = False,
    ):
        self.stac_path = stac_path
        self.offset_path = offset_path
        self.test = test
        self.project_folder = os.path.dirname(os.path.dirname(self.stac_path))
        self.params_path_source = os.path.join(
            self.project_folder, "stac_params", "params.yaml"
        )
        self.params_path_target = os.path.join(
            self.project_folder, "stac_params", "params_conversion.yaml"
        )
        self.n_samples_per_job = n_samples_per_job
        data = load_data(self.stac_path)
        self.n_frames = data["qpos"].shape[0]
        self.batches = self.generate_batches()

    def generate_batches(self):
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
                    "mocap_conversion",
                    str(start_frames[i]) + ".mat",
                ),
                "start_frame": start_frames[i],
                "end_frame": end_frames[i],
            }
            batches.append(batch)

        # with open("_parameters.p", "wb") as file:
        #     pickle.dump(batches, file)

        return batches

    def submit(self):
        if self.test:
            cmd = "sbatch --array=0 cluster/convert.sh"
        else:
            cmd = "sbatch --array=0-%d cluster/convert.sh" % (len(self.batches) - 1)

        os.system(cmd)


def merge(mocap_conversion_path):
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


# if __name__ == "__main__":
#     stac_paths = [
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_3/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_3/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_3/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_25_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_25_3/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_3/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_3/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_3/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_29_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_29_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_30_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_30_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_31_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_31_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_01_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_01_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_02_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_02_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_04_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_04_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_05_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_05_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_06_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_06_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_07_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_07_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_08_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_09_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_21_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_21_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_22_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_22_2/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_23_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_24_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_25_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_26_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_28_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_29_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_30_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_01_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_02_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_03_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_05_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_06_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_07_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_08_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_10_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_11_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_12_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_13_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_14_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_15_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_16_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_17_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_18_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_19_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_28_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_29_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_30_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_31_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_01_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_02_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_03_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_04_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_05_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_06_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_07_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_08_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_09_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_10_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_11_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_12_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_13_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_14_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_15_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_16_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_17_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_18_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_19_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_20_1/stac/total.p",
#         "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_21_1/stac/total.p",
#     ]
#     all_batches = []
#     for path in stac_paths:
#         pc = ParallelConverter(path)
#         all_batches.append(pc.batches.copy())
#     jobs = [job for batch in all_batches for job in batch]
#     with open("_parameters.p", "wb") as file:
#         pickle.dump(jobs, file)
#     print(len(jobs))
#     for job in jobs:
#         print(job["stac_path"])
#     cmd = "sbatch --array=0-%d cluster/convert.sh" % (len(jobs) - 1)
#     os.system(cmd)

# pc.submit()
# time.sleep(1)

if __name__ == "__main__":
    mocap_conversion_paths = [
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_3/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_3/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_3/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_25_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_25_3/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_3/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_3/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_3/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_29_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_29_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_30_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_30_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_31_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_31_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_01_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_01_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_02_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_02_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_04_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_04_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_05_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_05_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_06_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_06_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_07_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_07_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_08_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2021_01_09_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_21_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_21_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_22_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_22_2/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_23_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_24_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_25_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_26_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_28_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_29_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_06_30_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_01_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_02_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_03_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_05_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_06_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_07_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_08_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_10_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_11_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_12_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_13_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_14_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_15_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_16_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_17_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_18_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/bud/2021_07_19_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_28_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_29_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_30_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_31_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_01_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_02_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_03_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_04_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_05_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_06_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_07_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_08_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_09_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_10_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_11_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_12_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_13_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_14_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_15_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_16_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_17_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_18_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_19_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_20_1/mocap_conversion",
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_21_1/mocap_conversion",
    ]
    for path in mocap_conversion_paths:
        print(path, flush=True)
        merge(path)


# ANIMAL = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art"
# SESSIONS = [
#     "2020_12_21_1",
#     "2020_12_21_2",
#     "2020_12_22_1",
#     "2020_12_22_2",
#     "2020_12_22_3",
#     "2020_12_23_1",
#     "2020_12_23_2",
#     "2020_12_23_3",
#     "2020_12_24_1",
#     "2020_12_24_2",
#     "2020_12_24_3",
#     "2020_12_25_1",
#     "2020_12_25_2",
#     "2020_12_25_3",
#     "2020_12_26_1",
#     "2020_12_26_2",
#     "2020_12_26_3",
#     "2020_12_27_1",
#     "2020_12_27_2",
#     "2020_12_27_3",
#     "2020_12_28_1",
#     "2020_12_28_2",
#     "2020_12_28_3",
#     "2020_12_29_1",
#     "2020_12_29_2",
#     "2020_12_30_1",
#     "2020_12_30_2",
#     "2020_12_31_1",
#     "2020_12_31_2",
#     "2021_01_01_1",
#     "2021_01_01_2",
#     "2021_01_02_1",
#     "2021_01_02_2",
#     "2021_01_04_1",
#     "2021_01_04_2",
#     "2021_01_05_1",
#     "2021_01_05_2",
#     "2021_01_06_1",
#     "2021_01_06_2",
#     "2021_01_07_1",
#     "2021_01_07_2",
#     "2021_01_08_1",
#     "2021_01_09_1",
#     "2021_01_09_2",
#     "2021_01_10_0",
#     "2021_01_10_1",
#     "2021_01_10_2",
#     "2021_01_11_1",
#     "2021_01_11_2",
#     "2021_01_20_1",
#     "2021_01_21_1",
#     "2021_01_22_1",
#     "2021_01_23_1",
#     "2021_01_24_1",
#     "2021_01_25_1",
#     "2021_01_26_1",
#     "2021_01_27_1",
#     "2021_01_28_1",
#     "2021_01_29_1",
#     "2021_01_30_1",
#     "2021_02_07_1",
#     "2021_02_08_1",
#     "2021_02_11_1",
# ]
# FILE_PATH = [os.path.join(ANIMAL, sess, "stac", "total.p") for sess in SESSIONS]
# SAVE_PATH = [
#     os.path.join(ANIMAL, sess, "mocap_conversion", "data.mat") for sess in SESSIONS
# ]
# OFFSET_PATH = (
#     "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM25.p"
# )
# PARAMS_PATH_TARGET = os.path.join(
#     ANIMAL, "2020_12_22_1", "stac_params", "params_conversion.yaml"
# )
# PARAMS_PATH_SOURCE = os.path.join(
#     ANIMAL, "2020_12_22_1", "stac_params", "params.yaml"
# )
# PARAMETERS = []
# for fp, sp in zip(FILE_PATH, SAVE_PATH):
#     args = [fp, OFFSET_PATH, PARAMS_PATH_SOURCE, PARAMS_PATH_TARGET, sp]
#     PARAMETERS.append(args)
# with open("_parameters.p", "wb") as FILE:
#     pickle.dump(PARAMETERS, FILE)

# CMD = "sbatch --array=0-%d cluster/convert.sh" % (len(FILE_PATH) - 1)
# os.system(CMD)
