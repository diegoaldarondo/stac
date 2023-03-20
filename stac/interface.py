"""Interface with hpc to submit stac jobs."""
import os
import sys
import numpy as np
import pickle
import yaml
import h5py
from scipy.io import loadmat
from typing import List, Dict, Text


def load_params(param_path: Text) -> Dict:
    """Load parameters

    Args:
        param_path (Text): Path to parameters yaml file

    Returns:
        Dict: Parameters dictionary.
    """
    with open(param_path, "rb") as file:
        params = yaml.safe_load(file)
    return params


def get_unfinished(base_folder: Text, start_frames: np.ndarray) -> List:
    """Get unfinished jobs

    Args:
        base_folder (Text): Folder in which to look for clips.
        start_frames (np.ndarray): Starting frames of clips

    Returns:
        List: List of booleans denoting if each clip is finished or not.
    """
    is_unfinished = []
    for start_frame in start_frames:
        if not os.path.exists(os.path.join(base_folder, "%d.p" % (start_frame))):
            is_unfinished.append(True)
        else:
            is_unfinished.append(False)
    return is_unfinished


def save_variables(save_path: Text, commands: List[Dict]):
    """Save the stac batch commands

    Args:
        save_path (Text): File in which to save the commands
        commands (List[Dict]): List of commands to save.
    """
    out_dict = {"commands": commands}
    with open(save_path, "wb") as f:
        pickle.dump(out_dict, f)


def load_variables(save_path: Text) -> List[Dict]:
    """Load the commands.

    Args:
        save_path (Text): Description

    Returns:
        List[Dict]: Return list of commands
    """
    with open(save_path, "rb") as f:
        in_dict = pickle.load(f)
    return in_dict["commands"]


def get_clip_duration(data_path: Text) -> int:
    """Get the duration of the entire clip.

    Tries mocap format 1, dannce format, and mocap format 2.

    Args:
        data_path (Text): Path to the keypoint file

    Returns:
        int: Duration of clip in frames
    """
    try:
        M = loadmat(data_path)
        try:
            clip_duration = M["predictions"]["ArmL"][0, 0].shape[0]
        except KeyError:
            clip_duration = M["pred"].shape[0]
    except (NotImplementedError, FileNotFoundError):
        with h5py.File(data_path, "r") as f:
            clip_duration = f["mocapstruct_here"]["markers_preproc"]["ArmL"].shape[1]
    return clip_duration


def submit():
    """Submit the job to the cluster."""
    param_path = sys.argv[1]
    params = load_params(param_path)
    if params["clip_duration"] is None:
        params["clip_duration"] = get_clip_duration(params["data_path"])
    start_frames = np.arange(0, params["clip_duration"], params["snippet_duration"])
    # start_frames = start_frames[:5]
    end_frames = start_frames + params["snippet_duration"]
    end_frames[-1] = params["clip_duration"]

    commands = []
    for i in range(len(start_frames)):
        commands.append(
            {
                "start_frame": start_frames[i],
                "end_frame": end_frames[i],
                "clip_duration": params["clip_duration"],
                "base_folder": params["base_folder"],
                "data_path": params["data_path"],
                "param_path": params["param_path"],
                "offset_path": params["offset_path"],
            }
        )

    save_variables(params["temp_file_name"], commands)
    n_jobs = len(start_frames)
    print("Number of jobs: ", n_jobs)
    cmd = (
        "sbatch --wait --array=0-%d --exclude=holy2c18111 submit_stac_single_batch.sh %s"
        % (n_jobs - 1, param_path)
    )

    print(cmd)
    sys.exit(os.WEXITSTATUS(os.system(cmd)))


def submit_unfinished():
    """Submit unfinished jobs to the cluster."""
    run_param_path = sys.argv[1]
    params = load_params(run_param_path)
    # For every file in base_folder and data_path, break it up into chunks
    commands = []
    for base_folder, data_path, param_path in zip(
        [params["base_folder"]], [params["data_path"]], [params["param_path"]]
    ):
        params["clip_duration"] = get_clip_duration(data_path)
        start_frames = np.arange(0, params["clip_duration"], params["snippet_duration"])
        # start_frames = start_frames[0:2]
        end_frames = start_frames + params["snippet_duration"]
        end_frames[-1] = params["clip_duration"]

        is_unfinished = get_unfinished(base_folder, start_frames)

        for i, v in enumerate(is_unfinished):
            if v:
                commands.append(
                    {
                        "start_frame": start_frames[i],
                        "end_frame": end_frames[i],
                        "clip_duration": params["clip_duration"],
                        "base_folder": base_folder,
                        "data_path": data_path,
                        "param_path": param_path,
                        "offset_path": params["offset_path"],
                    }
                )

    save_variables(params["temp_file_name"], commands)
    n_jobs = len(commands)
    if n_jobs > 0:
        print("Number of jobs: ", n_jobs)
        cmd = (
            "sbatch --wait --array=0-%d%%1000 --exclude=holy2c18111 submit_stac_single_batch.sh %s"
            % (n_jobs - 1, run_param_path)
        )
        print(cmd)
        sys.exit(os.WEXITSTATUS(os.system(cmd)))


def compute_single_batch():
    """Compute for a single batch.

    CLI entry for single batch.

    Args:
        run_param_path (Text): Path to run parameters .yaml file.
    """
    import stac.compute_stac as compute_stac

    run_param_path = sys.argv[1]
    params = load_params(run_param_path)
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # task_id = 0

    commands = load_variables(params["temp_file_name"])
    command = commands[task_id]
    save_path = os.path.join(command["base_folder"], "%d.p" % (command["start_frame"]))
    print(command)
    print(save_path)
    st = compute_stac.STAC(
        command["data_path"],
        command["param_path"],
        save_path=save_path,
        offset_path=command["offset_path"],
        start_frame=command["start_frame"],
        end_frame=command["end_frame"],
        verbose=True,
    )
    data = st.transform(offset_path=command["offset_path"])
    st.save(data, save_path=save_path)
