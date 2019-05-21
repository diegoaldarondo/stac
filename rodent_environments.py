"""."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from datetime import datetime
from dm_control import composer
from dm_control import mjcf
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control.locomotion.walkers import initializers
from dm_control.locomotion.arenas import floors as arenas
from dm_control.locomotion.walkers import base
from dm_control.composer.observation import observable
from scipy.spatial.transform import Rotation as rotation
import numpy as np
import os
import scipy.optimize


_XML_PATH = '/home/diego/code/olveczky/dm/stac/models/rat_may17.xml'
_PARTS_TO_ZERO = ['toe', 'ankle', 'finger', 'wrist']
# _PARTS_TO_ZERO = ['toe', 'ankle', 'finger']
# _PARTS_TO_ZERO = ['ankle']

KEYPOINT_MODEL_PAIRS = {"ArmL": "hand_L",
                        "ArmR": "hand_R",
                        "ElbowL": "lower_arm_L",
                        "ElbowR": "lower_arm_R",
                        "HeadB": "skull",
                        "HeadF": "skull",
                        "HeadL": "skull",
                        "HipL": "upper_leg_L",
                        "HipR": "upper_leg_R",
                        "KneeL": "lower_leg_L",
                        "KneeR": "lower_leg_R",
                        "Offset1": "vertebra_1",
                        "Offset2": "vertebra_1",
                        "ShinL": "foot_L",
                        "ShinR": "foot_R",
                        "ShoulderL": "upper_arm_L",
                        "ShoulderR": "upper_arm_R",
                        "SpineF": "vertebra_cervical_5",
                        "SpineL": "pelvis",
                        "SpineM": "vertebra_1"}
LEFT_LEG = "0 0 .3 1"
RIGHT_LEG = ".3 0 0 1"
LEFT_ARM = "0 0 .8 1"
RIGHT_ARM = ".8 0 0 1"
HEAD = ".8 .8 0 1"
SPINE = ".8 .8 .8 1"
KEYPOINT_COLOR_PAIRS = {"ArmL": LEFT_ARM,
                        "ArmR": RIGHT_ARM,
                        "ElbowL": LEFT_ARM,
                        "ElbowR": RIGHT_ARM,
                        "HeadB": HEAD,
                        "HeadF": HEAD,
                        "HeadL": HEAD,
                        "HipL": LEFT_LEG,
                        "HipR": RIGHT_LEG,
                        "KneeL": LEFT_LEG,
                        "KneeR": RIGHT_LEG,
                        "Offset1": SPINE,
                        "Offset2": SPINE,
                        "ShinL": LEFT_LEG,
                        "ShinR": RIGHT_LEG,
                        "ShoulderL": LEFT_ARM,
                        "ShoulderR": RIGHT_ARM,
                        "SpineF": SPINE,
                        "SpineL": SPINE,
                        "SpineM": SPINE}

# KEYPOINT_INITIAL_OFFSETS = {"ArmL": "0. 0. 0.",
#                             "ArmR": "0. 0. 0.",
#                             "ElbowL": "0. 0. 0.",
#                             "ElbowR": "0. 0. 0.",
#                             "HeadB": "0. -.025 .045",
#                             "HeadF": ".025 -.025 .045",
#                             "HeadL": "0. .025 .045",
#                             "HipL": "0. 0. 0.005",
#                             "HipR": "0. 0. 0.005",
#                             "KneeL": "0. 0. 0.",
#                             "KneeR": "0. 0. 0.",
#                             "Offset1": "0.015 .0155 -0.005",
#                             "Offset2": "-0.015 .015 -0.005",
#                             "ShinL": "0.015 0.01 0.0125",
#                             "ShinR": "0.015 -0.01 0.0125",
#                             "ShoulderL": "0. 0. 0.",
#                             "ShoulderR": "0. 0. 0.",
#                             "SpineF": "0. 0. 0.005",
#                             "SpineL": "0. 0. 0.005",
#                             "SpineM": "0. 0. 0.005"}
KEYPOINT_INITIAL_OFFSETS = {"ArmL": "0. 0. 0.",
                            "ArmR": "0. 0. 0.",
                            "ElbowL": "0. 0. 0.",
                            "ElbowR": "0. 0. 0.",
                            "HeadB": "0. -.025 .045",
                            "HeadF": ".025 -.025 .045",
                            "HeadL": "0. .025 .045",
                            "HipL": "0.03 0. 0.04",
                            "HipR": "0. 0. 0.005",
                            "KneeL": "0. 0. 0.03",
                            "KneeR": "0. 0. 0.",
                            "Offset1": "0.015 .0155 -0.005",
                            "Offset2": "-0.015 .015 -0.005",
                            "ShinL": "0.02 0. 0.015",
                            "ShinR": "0.015 -0.01 0.0125",
                            "ShoulderL": "0. 0. 0.",
                            "ShoulderR": "0. 0. 0.",
                            "SpineF": "0. 0. 0.005",
                            "SpineL": "0. 0. 0.005",
                            "SpineM": "0. 0. 0.005"}
# _Q_TO_XMAT = {ankle_L}

_TIME_BINS = .03
_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

# Height of head above which the rat is considered standing.
_STAND_HEIGHT = 1.5
_TORQUE_THRESHOLD = 60


def rodent_mocap(kp_data, n_frames, random_state=None):
    """View a rat with mocap sites."""
    # Build a position-controlled Rat
    walker = Rat(initializer=initializers.ZerosInitializer(),
                 observable_options={'egocentric_camera': dict(enabled=True)})

    # Build a Floor arena that is obstructed by walls.
    arena = arenas.Floor(size=(1, 1))

    # Build a mocap viewing task
    task = ViewMocap(walker, arena, kp_data)
    return composer.Environment(time_limit=_TIME_BINS*(n_frames-1),
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


class ViewMocap(composer.Task):
    """A ViewMocap task."""

    def __init__(self,
                 walker,
                 arena,
                 kp_data,
                 walker_spawn_position=(0, 0, 0),
                 walker_spawn_rotation=None,
                 physics_timestep=0.001,
                 control_timestep=0.025,
                 precomp_qpos=None,
                 precomp_xpos=None,
                 render_video=False,
                 width=600,
                 height=480,
                 video_name=None,
                 fps=30.0):
        """Initialize ViewMocap environment.

        :param walker,
        :param arena,
        :param kp_data,
        :param walker_spawn_position=(0, 0, 0),
        :param walker_spawn_rotation=None,
        :param physics_timestep=0.001,
        :param control_timestep=0.025,
        :param precomp_qpos=None,
        :param precomp_xpos=None,
        :param render_video=False,
        :param width=600,
        :param height=480,
        :param video_name=None,
        :param fps=30.0
        """
        self._arena = arena
        self._walker = walker
        self._walker.create_root_joints(self._arena.attach(self._walker))
        self._walker_spawn_position = walker_spawn_position
        self._walker_spawn_rotation = walker_spawn_rotation
        self.kp_data = kp_data
        self.sites = []
        self.precomp_qpos = precomp_qpos
        self.precomp_xpos = precomp_xpos
        self.render_video = render_video
        self.width = width
        self.height = height
        self.fps = fps
        self.V = None
        if video_name is None:
            now = datetime.now()
            self.video_name = now.strftime("clips/%m_%d_%Y_%H_%M_%S.mp4")
        else:
            self.video_name = video_name
        for id, name in enumerate(KEYPOINT_MODEL_PAIRS):
            start = (np.random.rand(3)-.5)*.001
            site = self._arena.mjcf_model.worldbody.add('site', name=name,
                                           type='sphere',
                                           size=[.005],
                                           rgba=KEYPOINT_COLOR_PAIRS[name],
                                           pos=start,
                                           group=2)

            self.sites.append(site)
        enabled_observables = []
        enabled_observables += self._walker.observables.proprioception
        enabled_observables += self._walker.observables.kinematic_sensors
        enabled_observables += self._walker.observables.dynamic_sensors
        enabled_observables.append(self._walker.observables.sensors_touch)
        enabled_observables.append(self._walker.observables.egocentric_camera)
        for obs in enabled_observables:
            obs.enabled = True

            self.set_timesteps(physics_timestep=physics_timestep,
                               control_timestep=control_timestep)

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        self._arena.regenerate(random_state)
        self._arena.mjcf_model.visual.map.znear = 0.0002
        # self._arena.mjcf_model.visual.map.zfar = 4.

    def initialize_episode(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)

    def get_reward(self, physics):
        return 0.0

    def get_discount(self, physics):
        return 1.

    def grab_frame(self, physics):
        """Grab a frame from the simulation using render and opencv."""
        # Get RGB rendering of env
        scene_option = wrapper.MjvOption()
        scene_option.geomgroup[2] = 0
        rgbArr = physics.render(self.height, self.width,
                                camera_id='walker/side',
                                scene_option=scene_option)
        return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

    def _get_euler_angles(self, R):
        rot_x = np.arctan2(R[2, 1], R[2, 2])
        rot_y = np.arctan2(R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        rot_z = np.arctan2(R[1, 0], R[0, 0])
        return rot_x, rot_y, rot_z

    def _loss(self, q, physics, name):
        physics.named.data.qpos[name] = q
        mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
        if '_L' in name:
            dir = '_L'
        else:
            dir = '_R'

        if 'ankle' in name:
            R = physics.named.data.xmat['walker/foot' + dir]
        elif 'wrist' in name:
            R = physics.named.data.xmat['walker/hand' + dir]
        else:
            R = physics.named.data.xmat[name]

        rot_x, rot_y, rot_z = self._get_euler_angles(R.copy().reshape(3, 3))
        return rot_x**2 + rot_y**2

    def after_step(self, physics, random_state):
        """Update the mujoco markers on each step."""
        # Get the frame
        self.frame = physics.time()
        self.frame = np.floor(self.frame/_TIME_BINS).astype('int32')

        # Set the mocap marker positions
        physics.bind(self.sites).pos[:] = \
            np.reshape(self.kp_data[self.frame, :].T, (-1, 3))

        # Set qpose if it has been precomputed.
        if self.precomp_qpos is not None:
            physics.named.data.qpos[:] = self.precomp_qpos[self.frame]
            # Make certain parts parallel to the floor for cosmetics
            for id, name in enumerate(physics.named.data.qpos.axes.row.names):
                if any(part in name for part in _PARTS_TO_ZERO):
                    # Doing it through optimization is pretty easy, but a hack
                    q0 = physics.named.data.qpos[name].copy()
                    q_opt = \
                        scipy.optimize.minimize(
                            lambda q: self._loss(q, physics, name),
                            q0, options={'maxiter': 5})
                    physics.named.data.qpos[name] = q_opt.x

            physics.named.data.qvel[:] = 0.
            physics.named.data.qacc[:] = 0.
            # Forward kinematics for rendering
            mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

        if self.render_video:
            if not os.path.isdir('./clips'):
                os.makedir('./clips')
            if self.V is None:
                self.V = cv2.VideoWriter(self.video_name,
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         self.fps,
                                         (self.width, self.height))
            self.V.write(self.grab_frame(physics))


class Rat(base.Walker):
    """A position-controlled rat with control range scaled to [-1, 1]."""

    def _build(self,
               name='walker',
               marker_rgba=None,
               initializer=None):
        self._mjcf_root = mjcf.from_path(self._xml_path)
        if name:
            self._mjcf_root.model = name

        # Set corresponding marker color if specified.
        if marker_rgba is not None:
            for geom in self.marker_geoms:
                geom.set_attributes(rgba=marker_rgba)
        self.body_sites = []
        # Add keypoint sites to the mjcf model, and a reference to the sites as
        # an attribute for easier access
        for key, v in KEYPOINT_MODEL_PAIRS.items():
            parent = self._mjcf_root.find('body', v)
            site = parent.add('site', name=key,
                              type='sphere',
                              size=[.005],
                              rgba="0 0 0 1",
                              pos=KEYPOINT_INITIAL_OFFSETS[key])
            self.body_sites.append(site)
        super(Rat, self)._build(initializer=initializer)

    @property
    def upright_pose(self):
        return base.WalkerPose(xpos=_UPRIGHT_POS, xquat=_UPRIGHT_QUAT)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @composer.cached_property
    def actuators(self):
        return tuple(self._mjcf_root.find_all('actuator'))

    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find('body', 'torso')

    @composer.cached_property
    def head(self):
        return self._mjcf_root.find('body', 'skull')

    @composer.cached_property
    def left_arm_root(self):
        return self._mjcf_root.find('body', 'scapula_L')

    @composer.cached_property
    def right_arm_root(self):
        return self._mjcf_root.find('body', 'scapula_R')

    @composer.cached_property
    def ground_contact_geoms(self):
        return tuple(self._mjcf_root.find('body', 'foot_L').find_all('geom') +
                     self._mjcf_root.find('body', 'foot_R').find_all('geom'))

    @composer.cached_property
    def standing_height(self):
        return _STAND_HEIGHT

    @composer.cached_property
    def end_effectors(self):
        return (self._mjcf_root.find('body', 'lower_arm_R'),
                self._mjcf_root.find('body', 'lower_arm_L'),
                self._mjcf_root.find('body', 'foot_R'),
                self._mjcf_root.find('body', 'foot_L'))

    @composer.cached_property
    def observable_joints(self):
        return tuple(actuator.joint for actuator in self.actuators
                     if actuator.joint is not None)

    @composer.cached_property
    def bodies(self):
        return tuple(self._mjcf_root.find_all('body'))

    @composer.cached_property
    def egocentric_camera(self):
        return self._mjcf_root.find('camera', 'egocentric')

    @property
    def marker_geoms(self):
        return (self._mjcf_root.find('geom', 'lower_arm_R'),
                self._mjcf_root.find('geom', 'lower_arm_L'))

    @property
    def _xml_path(self):
        return _XML_PATH

    def _build_observables(self):
        return RodentObservables(self)


class RodentObservables(base.WalkerObservables):
    """Observables for the Rat."""

    @composer.observable
    def head_height(self):
        return observable.Generic(
            lambda physics: physics.bind(self._entity.head).xpos[2])

    @composer.observable
    def sensors_torque(self):
        return observable.MJCFFeature(
            'sensordata', self._entity.mjcf_model.sensor.torque,
            corruptor=lambda v,
            random_state: np.tanh(2 * v / _TORQUE_THRESHOLD))

    @composer.observable
    def actuator_activation(self):
        return observable.MJCFFeature('act',
                                      self._entity.mjcf_model.find_all('actuator'))

    @composer.observable
    def appendages_pos(self):
        """Equivalent to `end_effectors_pos` with the head's position appended."""
        def relative_pos_in_egocentric_frame(physics):
            end_effectors_with_head = (
                self._entity.end_effectors + (self._entity.head,))
            end_effector = physics.bind(end_effectors_with_head).xpos
            torso = physics.bind(self._entity.root_body).xpos
            xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
            return np.reshape(np.dot(end_effector - torso, xmat), -1)
        return observable.Generic(relative_pos_in_egocentric_frame)

    @property
    def proprioception(self):
        return [
                self.joints_pos,
                self.joints_vel,
                self.actuator_activation,
                self.body_height,
                self.end_effectors_pos,
                self.appendages_pos,
                self.world_zaxis
                ] + self._collect_from_attachments('proprioception')
