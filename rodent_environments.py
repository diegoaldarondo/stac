"""Environment for rodent modeling with dm_control and motion capture."""
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
import numpy as np
import scipy.optimize
from scipy import ndimage

_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

# # Height of head above which the rat is considered standing.
_TORQUE_THRESHOLD = 60
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = .4  # Spatial scale of terrain bumps (in meters).


def rodent_mocap(kp_data, params, random_state=None):
    """View a rat with mocap sites."""
    # Build a position-controlled Rat
    walker = Rat(initializer=initializers.ZerosInitializer(), params=params,
                 observable_options={'egocentric_camera': dict(enabled=True)})

    # Build a Floor arena that is obstructed by walls.
    arena = arenas.Floor(size=(1, 1))

    # Build a mocap viewing task
    task = ViewMocap(walker, arena, kp_data, params=params)
    time_limit = params['_TIME_BINS']*(params['n_frames']-1)
    return composer.Environment(time_limit=time_limit,
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
                 render_video=False,
                 width=600,
                 height=480,
                 video_name=None,
                 params=None,
                 fps=30.0):
        """Initialize ViewMocap environment.

        :param walker: Rodent walker
        :param arena: Arena defining floor
        :param kp_data: Keypoint data (t x (n_marker*ndims))
        :param walker_spawn_position: Initial spawn position.
        :param walker_spawn_rotation: Initial spawn rotation.
        :param physics_timestep: Timestep for physics simulation
        :param control_timestep: Timestep for controller
        :param precomp_qpos: Precomputed list of qposes.
        :param render_video: If true, render a video of the simulation.
        :param width: Width of video
        :param height: Height of video
        :param video_name: Name of video
        :param fps: Frame rate of video
        """
        self._arena = arena
        self._walker = walker
        self._walker.create_root_joints(self._arena.attach(self._walker))
        self._walker_spawn_position = walker_spawn_position
        self._walker_spawn_rotation = walker_spawn_rotation
        self.kp_data = kp_data
        self.sites = []
        self.precomp_qpos = precomp_qpos
        self.render_video = render_video
        self.width = width
        self.height = height
        self.fps = fps
        self.V = None
        self.params = params
        if video_name is None:
            now = datetime.now()
            self.video_name = now.strftime("clips/%m_%d_%Y_%H_%M_%S.mp4")
        else:
            self.video_name = video_name
        for id, name in enumerate(self.params['_KEYPOINT_MODEL_PAIRS']):
            start = (np.random.rand(3)-.5)*.001
            rgba = self.params['_KEYPOINT_COLOR_PAIRS'][name]
            site = self._arena.mjcf_model.worldbody.add('site', name=name,
                                                        type='sphere',
                                                        size=[.005],
                                                        rgba=rgba,
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
        """Return arena root."""
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        """Initialize an arena episode."""
        self._arena.regenerate(random_state)
        self._arena.mjcf_model.visual.map.znear = 0.0002
        # self._arena.mjcf_model.visual.map.zfar = 4.

    def initialize_episode(self, physics, random_state):
        """Reinitialize the pose of the walker."""
        self._walker.reinitialize_pose(physics, random_state)

    def get_reward(self, physics):
        """Get reward."""
        return 0.0

    def get_discount(self, physics):
        """Get discount."""
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
        self.frame = \
            np.floor(self.frame/self.params['_TIME_BINS']).astype('int32')

        # Set the mocap marker positions
        physics.bind(self.sites).pos[:] = \
            np.reshape(self.kp_data[self.frame, :].T, (-1, 3))

        # Set qpose if it has been precomputed.
        if self.precomp_qpos is not None:
            physics.named.data.qpos[:] = self.precomp_qpos[self.frame]
            physics.named.data.qpos['walker/mandible'] = \
                self.params['_MANDIBLE_POS']

            # Make certain parts parallel to the floor for cosmetics
            for id, name in enumerate(physics.named.data.qpos.axes.row.names):
                if any(part in name for part in self.params['_PARTS_TO_ZERO']):
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
            # if not os.path.isdir('./clips'):
            #     os.makedir('./clips')
            if self.V is None:
                self.V = cv2.VideoWriter(self.video_name,
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         self.fps,
                                         (self.width, self.height))
            self.V.write(self.grab_frame(physics))


class ViewMocap_Hfield(ViewMocap):
    """View mocap while modeling uneven terrain."""

    def __init__(self,
                 walker,
                 arena,
                 kp_data,
                 walker_spawn_position=(0, 0, 0),
                 walker_spawn_rotation=None,
                 physics_timestep=0.001,
                 control_timestep=0.025,
                 precomp_qpos=None,
                 render_video=False,
                 width=600,
                 height=480,
                 video_name=None,
                 params=None,
                 fps=30.0):
        """Initialize ViewMocap environment with heightfield.

        :param walker: Rodent walker
        :param arena: Arena defining heightfield
        :param kp_data: Keypoint data (t x (n_marker*ndims))
        :param walker_spawn_position: Initial spawn position.
        :param walker_spawn_rotation: Initial spawn rotation.
        :param physics_timestep: Timestep for physics simulation
        :param control_timestep: Timestep for controller
        :param precomp_qpos: Precomputed list of qposes.
        :param render_video: If true, render a video of the simulation.
        :param width: Width of video
        :param height: Height of video
        :param video_name: Name of video
        :param fps: Frame rate of video
        """
        super(ViewMocap_Hfield, self).__init__(walker, arena, kp_data,
                                               precomp_qpos=precomp_qpos,
                                               render_video=render_video,
                                               width=width,
                                               height=height,
                                               video_name=video_name,
                                               params=params,
                                               fps=fps)

    def initialize_episode(self, physics, random_state):
        """Set the state of the environment at the start of each episode.

        :param physics: An instance of `Physics`.
        """
        print('Made it to top')
        # Get heightfield resolution, assert that it is square.
        res = physics.model.hfield_nrow[_HEIGHTFIELD_ID]
        assert res == physics.model.hfield_ncol[_HEIGHTFIELD_ID]

        # # Sinusoidal bowl shape.
        # row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
        # radius = np.clip(np.sqrt(col_grid**2 + row_grid**2), .04, 1)
        # bowl_shape = .5 - np.cos(2*np.pi*radius)/2

        # Random smooth bumps.
        terrain_size = 2 * physics.model.hfield_size[_HEIGHTFIELD_ID, 0]
        bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)
        bumps = np.random.uniform(_TERRAIN_SMOOTHNESS, 1,
                                  (bump_res, bump_res)) - .5
        smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
        # Terrain is elementwise product.
        # terrain = bowl_shape * smooth_bumps
        terrain = smooth_bumps
        start_idx = physics.model.hfield_adr[_HEIGHTFIELD_ID]
        physics.model.hfield_data[start_idx:start_idx+res**2] = terrain.ravel()

        # super(ViewMocap_Hfield, self).initialize_episode(physics)

        # If we have a rendering context, we need to re-upload the modified
        # heightfield data.
        if physics.contexts:
            with physics.contexts.gl.make_current() as ctx:
                ctx.call(mjlib.mjr_uploadHField,
                         physics.model.ptr,
                         physics.contexts.mujoco.ptr,
                         _HEIGHTFIELD_ID)
        print('Made it to bot')
        self._walker.reinitialize_pose(physics, random_state)
        # # Initial configuration.
        # orientation = self.random.randn(4)
        # orientation /= np.linalg.norm(orientation)
        # _find_non_contacting_height(physics, orientation)


class Rat(base.Walker):
    """A position-controlled rat with control range scaled to [-1, 1]."""

    def _build(self, params=None,
               name='walker',
               marker_rgba=None,
               initializer=None):
        self.params = params
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
        for key, v in self.params['_KEYPOINT_MODEL_PAIRS'].items():
            parent = self._mjcf_root.find('body', v)
            pos = self.params['_KEYPOINT_INITIAL_OFFSETS'][key]
            site = parent.add('site', name=key,
                              type='sphere',
                              size=[.005],
                              rgba="0 0 0 1",
                              pos=pos)
            self.body_sites.append(site)
        super(Rat, self)._build(initializer=initializer)

    @property
    def upright_pose(self):
        """Reset pose to upright position."""
        return base.WalkerPose(xpos=_UPRIGHT_POS,
                               xquat=_UPRIGHT_QUAT)

    @property
    def mjcf_model(self):
        """Return the model root."""
        return self._mjcf_root

    @composer.cached_property
    def actuators(self):
        """Return all actuators."""
        return tuple(self._mjcf_root.find_all('actuator'))

    @composer.cached_property
    def root_body(self):
        """Return the body."""
        return self._mjcf_root.find('body', 'torso')

    @composer.cached_property
    def head(self):
        """Return the head."""
        return self._mjcf_root.find('body', 'skull')

    @composer.cached_property
    def left_arm_root(self):
        """Return the left arm."""
        return self._mjcf_root.find('body', 'scapula_L')

    @composer.cached_property
    def right_arm_root(self):
        """Return the right arm."""
        return self._mjcf_root.find('body', 'scapula_R')

    @composer.cached_property
    def ground_contact_geoms(self):
        """Return ground contact geoms."""
        return tuple(self._mjcf_root.find('body', 'foot_L').find_all('geom') +
                     self._mjcf_root.find('body', 'foot_R').find_all('geom'))

    @composer.cached_property
    def standing_height(self):
        """Return standing height."""
        return self.params['_STAND_HEIGHT']

    @composer.cached_property
    def end_effectors(self):
        """Return end effectors."""
        return (self._mjcf_root.find('body', 'lower_arm_R'),
                self._mjcf_root.find('body', 'lower_arm_L'),
                self._mjcf_root.find('body', 'foot_R'),
                self._mjcf_root.find('body', 'foot_L'))

    @composer.cached_property
    def observable_joints(self):
        """Return observable joints."""
        return tuple(actuator.joint for actuator in self.actuators
                     if actuator.joint is not None)

    @composer.cached_property
    def bodies(self):
        """Return all bodies."""
        return tuple(self._mjcf_root.find_all('body'))

    @composer.cached_property
    def egocentric_camera(self):
        """Return the egocentric camera."""
        return self._mjcf_root.find('camera', 'egocentric')

    @property
    def marker_geoms(self):
        """Return the lower arm geoms."""
        return (self._mjcf_root.find('geom', 'lower_arm_R'),
                self._mjcf_root.find('geom', 'lower_arm_L'))

    @property
    def _xml_path(self):
        """Return the path to th model .xml file."""
        return self.params['_XML_PATH']

    def _build_observables(self):
        return RodentObservables(self)


class RodentObservables(base.WalkerObservables):
    """Observables for the Rat."""

    @composer.observable
    def head_height(self):
        """Observe the head height."""
        return observable.Generic(
            lambda physics: physics.bind(self._entity.head).xpos[2])

    @composer.observable
    def sensors_torque(self):
        """Observe the torque sensors."""
        return observable.MJCFFeature(
            'sensordata', self._entity.mjcf_model.sensor.torque,
            corruptor=lambda v,
            random_state: np.tanh(2 * v / _TORQUE_THRESHOLD))

    @composer.observable
    def actuator_activation(self):
        """Observe the actuator activation."""
        model = self._entity.mjcf_model
        return observable.MJCFFeature('act', model.find_all('actuator'))

    @composer.observable
    def appendages_pos(self):
        """Equivalent to `end_effectors_pos` with head's position appended."""
        def relative_pos_in_egocentric_frame(physics):
            end_effectors_with_head = (
                self._entity.end_effectors + (self._entity.head,))
            end_effector = physics.bind(end_effectors_with_head).xpos
            torso = physics.bind(self._entity.root_body).xpos
            xmat = \
                np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
            return np.reshape(np.dot(end_effector - torso, xmat), -1)
        return observable.Generic(relative_pos_in_egocentric_frame)

    @property
    def proprioception(self):
        """Return proprioceptive information."""
        return [
                self.joints_pos,
                self.joints_vel,
                self.actuator_activation,
                self.body_height,
                self.end_effectors_pos,
                self.appendages_pos,
                self.world_zaxis
                ] + self._collect_from_attachments('proprioception')
