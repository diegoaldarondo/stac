"""Environment for rodent modeling with dm_control and motion capture."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dm_control import composer
from dm_control.locomotion.arenas import floors
import walkers
import tasks
import arenas

_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

MM_TO_METER = 1000
# Height of head above which the rat is considered standing.
_TORQUE_THRESHOLD = 60
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = .4  # Spatial scale of terrain bumps (in meters).
_TOP_CAMERA_DISTANCE = 100
_TOP_CAMERA_Y_PADDING_FACTOR = 1.1
PEDESTAL_WIDTH = .099
PEDESTAL_HEIGHT = .054


def rodent_mocap(
        kp_data, params, random_state=None,
        use_hfield=False, hfield_image=None, pedestal_center=None,
        pedestal_height=None, pedestal_radius=None, arena_diameter=None):
    """View a rat with mocap sites."""
    # Build a position-controlled Rat
    walker = walkers.Rat(
        initializer=None, params=params,
        observable_options={'egocentric_camera': dict(enabled=True)})
    if use_hfield:
        process_objects = False
        if hfield_image is None:
            hfield_image = arenas._load_hfield(
                params['data_path'], params['scale_factor'])
            process_objects = True
        arena = arenas.RatArena(
            hfield_image, params, process_objects,
            pedestal_center=pedestal_center, pedestal_height=pedestal_height,
            pedestal_radius=pedestal_radius, arena_diameter=arena_diameter)
        task = tasks.ViewMocap_Hfield(walker, arena, kp_data, params=params)
    else:
        # Build a Floor arena
        arena = floors.Floor(size=(1, 1))
        arena._ground_geom.pos = [0., 0., -.02]
        # Build a mocap viewing task
        task = tasks.ViewMocap(walker, arena, kp_data, params=params)

    time_limit = params['_TIME_BINS'] * (params['n_frames'] - 1)
    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)
