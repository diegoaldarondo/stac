"""Merge snippets into a contiguous file."""
import numpy as np
import pickle
import clize
import os


def _sort_fn(file):
    return int(file.split(".p")[0])


def get_chunk_data(chunk_path):
    with open(chunk_path, "rb") as f:
        in_dict = pickle.load(f)
        q = np.stack(in_dict["qpos"], axis=0)
        for frame in in_dict["xpos"]:
            frame = [marker[0] for marker in frame]
        x = np.stack(in_dict["xpos"], axis=0)
        kp_data = in_dict["kp_data"]
    return q, x, kp_data, in_dict


def merge(folder, *, save_path=None):
    """Merge snippets into a contiguous file."""
    files = [
        f
        for f in os.listdir(folder)
        if (".p" in f) and ("total" not in f and "offset" not in f)
    ]
    files = sorted(files, key=_sort_fn)

    q, x, kp_data = [], [], []

    # Get the sizes of data from first chunk
    q, x, kp_data, in_dict = get_chunk_data(os.path.join(folder, files[0]))
    q_rem, _, _, _ = get_chunk_data(os.path.join(folder, files[-1]))

    n_frames = q.shape[0]
    remainder = q_rem.shape[0]
    q = np.zeros((n_frames * (len(files) - 1) + remainder, q.shape[1]))
    x = np.zeros((n_frames * (len(files) - 1) + remainder, x.shape[1], x.shape[2]))
    kp_data = np.zeros((n_frames * (len(files) - 1) + remainder, kp_data.shape[1]))

    for i, file in enumerate(files):
        print("%d of %d" % (i, len(files)), flush=True)
        q_i, x_i, kp_data_i, _ = get_chunk_data(os.path.join(folder, file))
        start_frame = i * n_frames
        end_frame = np.min([((i + 1) * n_frames), q.shape[0]])
        q[start_frame:end_frame, ...] = q_i
        x[start_frame:end_frame, ...] = x_i
        kp_data[start_frame:end_frame, ...] = kp_data_i

    out_dict = in_dict
    out_dict["qpos"] = q
    out_dict["xpos"] = x
    out_dict["kp_data"] = kp_data
    if save_path is None:
        save_path = os.path.join(folder, "total.p")
    with open(save_path, "wb") as f:
        pickle.dump(out_dict, f, protocol=4)


if __name__ == "__main__":
    clize.run(merge)
