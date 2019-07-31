"""Merge snippets into a contiguous file."""
import numpy as np
import pickle
import clize
import os


def merge(folder, *, save_path=None):
    """Merge snippets into a contiguous file."""
    files = [f for f in os.listdir(folder)
             if ('.p' in f) and ('total' not in f)]
    files.sort()

    q = []
    kp_data = []
    for file in files:
        with open(os.path.join(folder, file), 'rb') as f:
            in_dict = pickle.load(f)
            q.append(np.stack(in_dict['qpos'], axis=0))
            kp_data.append(in_dict['kp_data'])

    q = np.concatenate(q, axis=0)
    kp_data = np.concatenate(kp_data, axis=0)

    out_dict = in_dict
    out_dict['qpos'] = q
    out_dict['kp_data'] = q
    if save_path is None:
        save_path = os.path.join(folder, 'total.p')
    with open(save_path, 'wb') as f:
        pickle.dump(out_dict, f)


if __name__ == "__main__":
    clize.run(merge)
