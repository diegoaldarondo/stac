"""Merge snippets into a contiguous file."""
import numpy as np
import pickle
import clize
import os
import argparse
from scipy.io import savemat


def _sort_fn(file: Text):
    """Helper to sort files by their name.

    Args:
        file (Text): File name

    Returns:
        TYPE: File int value
    """
    return int(file.split(".p")[0])


def get_chunk_data(chunk_path: Text) -> Tuple:
    """Load data from a chunk

    Args:
        chunk_path (Text): Path to a chunk

    Returns:
        Tuple: q (np.ndarray): array of qpos
            x (np.ndarray): array of xpos
            kp_data (np.ndarray): array of keypoint data
            walker_body_sites (np.ndarray): array of walker sites
            in_dict (Dict): chunk file
    """
    with open(chunk_path, "rb") as f:
        in_dict = pickle.load(f)
        q = np.stack(in_dict["qpos"], axis=0)
        for frame in in_dict["xpos"]:
            frame = [marker[0] for marker in frame]
        x = np.stack(in_dict["xpos"], axis=0)
        walker_body_sites = np.stack(in_dict["walker_body_sites"], axis=0)
        kp_data = in_dict["kp_data"]
    return q, x, kp_data, walker_body_sites, in_dict


def merge(folder: Text, *, save_path: Text = None):
    """Merge snippets into a contiguous file.

    Args:
        folder (Text): Path to folder containing chunks.
        save_path (Text, optional): Path to file in which to save merge.
    """

    def hasNumbers(input_string: Text) -> bool:
        """Helper function to find files with digits in filename.

        Args:
            input_string (Text): String to parse

        Returns:
            bool: True if string contains a digit
        """
        return any(char.isdigit() for char in input_string)

    files = [f for f in os.listdir(folder) if (".p" in f) and hasNumbers(f)]
    files = sorted(files, key=_sort_fn)

    q, x, kp_data = [], [], []

    # Get the sizes of data from first chunk
    q, x, kp_data, walker_body_sites, in_dict = get_chunk_data(
        os.path.join(folder, files[0])
    )
    q_rem, _, _, _, _ = get_chunk_data(os.path.join(folder, files[-1]))

    n_frames = q.shape[0]
    remainder = q_rem.shape[0]
    out_dict = in_dict
    out_dict["qpos"] = np.zeros((n_frames * (len(files) - 1) + remainder, q.shape[1]))
    out_dict["xpos"] = np.zeros(
        (n_frames * (len(files) - 1) + remainder, x.shape[1], x.shape[2])
    )
    out_dict["walker_body_sites"] = np.zeros(
        (
            n_frames * (len(files) - 1) + remainder,
            walker_body_sites.shape[1],
            walker_body_sites.shape[2],
        )
    )
    out_dict["kp_data"] = np.zeros(
        (n_frames * (len(files) - 1) + remainder, kp_data.shape[1])
    )

    for i, file in enumerate(files):
        print("%d of %d" % (i, len(files)), flush=True)
        q_i, x_i, kp_data_i, walker_body_sites_i, _ = get_chunk_data(
            os.path.join(folder, file)
        )
        start_frame = i * n_frames
        end_frame = np.min([((i + 1) * n_frames), out_dict["qpos"].shape[0]])
        out_dict["qpos"][start_frame:end_frame, ...] = q_i
        out_dict["xpos"][start_frame:end_frame, ...] = x_i
        out_dict["walker_body_sites"][start_frame:end_frame, ...] = walker_body_sites_i
        out_dict["kp_data"][start_frame:end_frame, ...] = kp_data_i

    if save_path is None:
        save_path = os.path.join(folder, "total.p")
    with open(save_path, "wb") as f:
        pickle.dump(out_dict, f, protocol=4)
    mat_path = save_path.split(".p")[0] + ".mat"
    for k, v in out_dict.items():
        if v is None:
            out_dict[k] = "None"
    savemat(mat_path, out_dict)


def stac_merge():
    """CLI Entrypoint to merge stac chunks into one file."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stac_folder",
        help="Path to stac folder containing stacced snippets.",
    )
    args = parser.parse_args()
    merge(args.stac_folder)


if __name__ == "__main__":
    clize.run(merge)
