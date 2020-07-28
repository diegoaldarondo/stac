import os
import sys
import numpy as np
import pickle
import yaml
import h5py
from scipy.io import loadmat


def load_params(param_path):
    with open(param_path, "rb") as f:
        params = yaml.safe_load(f)
    return params


def get_unfinished(base_folder, start_frames):
    is_unfinished = []
    for i, sf in enumerate(start_frames):
        if not os.path.exists(os.path.join(base_folder, "%d.p" % (sf))):
            is_unfinished.append(True)
        else:
            is_unfinished.append(False)
    return is_unfinished


def save_variables(save_path, commands):
    out_dict = {"commands": commands}
    with open(save_path, "wb") as f:
        pickle.dump(out_dict, f)


def load_variables(save_path):
    with open(save_path, "rb") as f:
        in_dict = pickle.load(f)
    return in_dict["commands"]


def get_clip_duration(data_path):
    M = loadmat(data_path)
    clip_duration = M["predictions"]["ArmL"][0, 0].shape[0]
    # import pdb
    # pdb.set_trace()
    # with h5py.File(data_path, "r") as f:
    #     # clip_duration = f['mocapstruct_here']['markers_preproc']['ArmL'].shape[1]
    #     clip_duration = f["predictions"]["ArmL"].shape[1]
    return clip_duration


# Print to std out if just asking for the number of commands
# (for sbatch --array=0-$print_num_commands ...)
if __name__ == "__main__":
    valid_options = any(
        [option in sys.argv[1] for option in ["--submit", "--submit-unfinished"]]
    )
    if len(sys.argv) >= 2 and valid_options:
        if sys.argv[1] == "--submit":
            param_path = sys.argv[2]
            params = load_params(param_path)
            if params["clip_duration"] is None:
                params["clip_duration"] = get_clip_duration(params["data_path"])
            start_frames = np.arange(
                0, params["clip_duration"], params["snippet_duration"]
            )
            end_frames = start_frames + params["snippet_duration"]
            save_variables(params["temp_file_name"], start_frames, end_frames)
            n_jobs = len(start_frames)
            print("Number of jobs: ", n_jobs)
            cmd = (
                "sbatch --array=0-%d --partition=shared,olveczky,serial_requeue --exclude=seasmicro25,holy2c18111 cluster/submit_compute_stac_clip.sh %s"
                % (n_jobs - 1, param_path)
            )
            print(cmd)
            # os.system(cmd)

        if sys.argv[1] == "--submit-unfinished":
            run_param_path = sys.argv[2]
            params = load_params(run_param_path)

            # For every file in base_folder and data_path, break it up into chunks
            commands = []
            for base_folder, data_path, param_path in zip(
                params["base_folder"], params["data_path"], params["param_path"]
            ):
                params["clip_duration"] = get_clip_duration(data_path)

                start_frames = np.arange(
                    0, params["clip_duration"], params["snippet_duration"]
                )
                # start_frames = start_frames[0:2]
                end_frames = start_frames + params["snippet_duration"]
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

            # start_frames = start_frames[:2]
            # end_frames = end_frames[:2]
            # commands = commands[0:2]
            save_variables(params["temp_file_name"], commands)
            n_jobs = len(commands)
            print("Number of jobs: ", n_jobs)
            cmd = (
                "sbatch --array=0-%d --partition=shared,olveczky,serial_requeue --exclude=seasmicro25,holy2c18111 cluster/submit_compute_stac_clip.sh %s"
                % (n_jobs - 1, run_param_path)
            )
            print(cmd)
            os.system(cmd)

    # Otherwise, run the command for the appropriate job id
    elif len(sys.argv) == 2 and os.path.splitext(sys.argv[1])[-1] == ".yaml":
        import compute_stac

        run_param_path = sys.argv[1]
        params = load_params(run_param_path)
        task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        # task_id = 0

        commands = load_variables(params["temp_file_name"])
        command = commands[task_id]
        save_path = os.path.join(
            command["base_folder"], "%d.p" % (command["start_frame"])
        )
        print(command)
        print(save_path)
        compute_stac.handle_args(
            command["data_path"],
            command["param_path"],
            save_path=save_path,
            offset_path=command["offset_path"],
            verbose=True,
            process_snippet=False,
            start_frame=command["start_frame"],
            end_frame=command["end_frame"],
            skip=1,
        )
