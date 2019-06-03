"""View the base model."""
from dm_control import viewer
import clize
import rodent_environments
import numpy as np
import util


def view_model(*, param_path='./params/baseParams.yaml'):
    """View the model with base parameters."""
    kp_data = np.zeros((100, 60))
    params = util.load_params(param_path)
    params['n_frames'] = kp_data.shape[0]
    env = rodent_environments.rodent_mocap(kp_data, params)
    viewer.launch(env)


if __name__ == '__main__':
    clize.run(view_model)
