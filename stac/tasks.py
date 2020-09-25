"""Task for rat mocap."""
from dm_control import composer
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control.mujoco.wrapper.mjbindings import enums
import numpy as np
import cv2
from datetime import datetime
import scipy.optimize

_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)
MM_TO_METER = 1000

# Height of head above which the rat is considered standing.
_TORQUE_THRESHOLD = 60
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 0.4  # Spatial scale of terrain bumps (in meters).
_TOP_CAMERA_DISTANCE = 100
_TOP_CAMERA_Y_PADDING_FACTOR = 1.1
PEDESTAL_WIDTH = 0.099
PEDESTAL_HEIGHT = 0.054


class ViewMocap(composer.Task):
    """A ViewMocap task."""

    def __init__(
        self,
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
        fps=30.0,
    ):
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
        for id, name in enumerate(self.params["_KEYPOINT_MODEL_PAIRS"]):
            start = (np.random.rand(3) - 0.5) * 0.001
            rgba = self.params["_KEYPOINT_COLOR_PAIRS"][name]
            site = self._arena.mjcf_model.worldbody.add(
                "site",
                name=name,
                type="sphere",
                size=[0.005],
                rgba=rgba,
                pos=start,
                group=2,
            )
            self.sites.append(site)
        enabled_observables = []
        enabled_observables += self._walker.observables.proprioception
        enabled_observables += self._walker.observables.kinematic_sensors
        enabled_observables += self._walker.observables.dynamic_sensors
        enabled_observables.append(self._walker.observables.sensors_touch)
        enabled_observables.append(self._walker.observables.egocentric_camera)
        for obs in enabled_observables:
            obs.enabled = True

            self.set_timesteps(
                physics_timestep=physics_timestep, control_timestep=control_timestep
            )

    @property
    def root_entity(self):
        """Return arena root."""
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        """Initialize an arena episode."""
        # self._arena.regenerate(random_state)
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
        return 1.0

    def grab_frame_and_seg(self, physics):
        """Grab a frame from the simulation using render and opencv."""
        # Get RGB rendering of env
        scene_option = wrapper.MjvOption()
        # scene_option.geomgroup[3] = 0
        # scene_option.geomgroup[1] = 0
        # scene_option.geomgroup[2] = 0
        # import pdb;
        # pdb.set_trace()

        # for i in range(len(scene_option._ptr.contents.geomgroup)):
        #     scene_option.geomgroup[i] = False
        #     scene_option._ptr.contents.geomgroup[i] = False
        # import pdb;
        # pdb.set_trace()
        physics.model.skin_rgba[0][3] = 0.0
        scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_LIGHT] = False
        scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_TEXTURE] = False

        rgbArr = physics.render(
            self.height, self.width, camera_id="CameraE", scene_option=scene_option
        )
        seg = physics.render(
            self.height,
            self.width,
            camera_id="CameraE",
            scene_option=scene_option,
            segmentation=True,
        )

        # import pdb
        # pdb.set_trace()
        bkgrd = (seg[:, :, 0] == -1) & (seg[:, :, 1] == -1)
        floor = (seg[:, :, 0] == 0) & (seg[:, :, 1] == 5)
        rgbArr[:, :, 0] *= ~bkgrd[:, :]
        rgbArr[:, :, 1] *= ~bkgrd[:, :]
        rgbArr[:, :, 2] *= ~bkgrd[:, :]
        rgbArr[:, :, 0] *= ~floor[:, :]
        rgbArr[:, :, 1] *= ~floor[:, :]
        rgbArr[:, :, 2] *= ~floor[:, :]
        return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

    def grab_frame(self, physics):
        """Grab a frame from the simulation using render and opencv."""
        # Get RGB rendering of env
        scene_option = wrapper.MjvOption()
        # scene_option.geomgroup[2] = 0
        scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
        rgbArr = physics.render(
            self.height,
            self.width,
            camera_id="walker/close_profile",
            scene_option=scene_option,
        )
        return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

    def _get_euler_angles(self, R):
        rot_x = np.arctan2(R[2, 1], R[2, 2])
        rot_y = np.arctan2(R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        rot_z = np.arctan2(R[1, 0], R[0, 0])
        return rot_x, rot_y, rot_z

    def _loss(self, q, physics, name):
        physics.named.data.qpos[name] = q
        mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
        if "_L" in name:
            dir = "_L"
        else:
            dir = "_R"

        if "ankle" in name:
            R = physics.named.data.xmat["walker/foot" + dir]
        elif "wrist" in name:
            R = physics.named.data.xmat["walker/hand" + dir]
        else:
            R = physics.named.data.xmat[name]

        rot_x, rot_y, rot_z = self._get_euler_angles(R.copy().reshape(3, 3))
        return rot_x ** 2 + rot_y ** 2

    def after_step(self, physics, random_state):
        """Update the mujoco markers on each step."""
        # Get the frame
        self.frame = physics.time()
        self.frame = np.floor(self.frame / self.params["_TIME_BINS"]).astype("int32")
        # Set the mocap marker positions
        physics.bind(self.sites).pos[:] = np.reshape(
            self.kp_data[self.frame, :].T, (-1, 3)
        )

        # Set qpose if it has been precomputed.
        if self.precomp_qpos is not None:
            physics.named.data.qpos[:] = self.precomp_qpos[self.frame]
            physics.named.data.qpos["walker/mandible"] = self.params["_MANDIBLE_POS"]

            # Make certain parts parallel to the floor for cosmetics
            for id, name in enumerate(physics.named.data.qpos.axes.row.names):
                if any(part in name for part in self.params["_PARTS_TO_ZERO"]):
                    # Doing it through optimization is pretty easy, but a hack
                    q0 = physics.named.data.qpos[name].copy()
                    q_opt = scipy.optimize.minimize(
                        lambda q: self._loss(q, physics, name),
                        q0,
                        options={"maxiter": 5},
                    )
                    physics.named.data.qpos[name] = q_opt.x

            physics.named.data.qvel[:] = 0.0
            physics.named.data.qacc[:] = 0.0
            # Forward kinematics for rendering
            mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

        if self.render_video:
            if self.V is None:
                self.V = cv2.VideoWriter(
                    self.video_name,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (self.width, self.height),
                )
            self.V.write(self.grab_frame(physics))


class ViewMocap_Hfield(ViewMocap):
    """View mocap while modeling uneven terrain."""

    def __init__(
        self,
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
        fps=30.0,
    ):
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
        super(ViewMocap_Hfield, self).__init__(
            walker,
            arena,
            kp_data,
            precomp_qpos=precomp_qpos,
            render_video=render_video,
            width=width,
            height=height,
            video_name=video_name,
            params=params,
            fps=fps,
        )

    def set_heightfield(self, physics):
        """Set the physics.hfield_data to self.hfield_image."""
        res = physics.model.hfield_nrow[_HEIGHTFIELD_ID]
        assert res == physics.model.hfield_ncol[_HEIGHTFIELD_ID]

        # Find the bounds of the arena in the hfield.
        start_idx = physics.model.hfield_adr[_HEIGHTFIELD_ID]
        physics.model.hfield_data[
            start_idx : start_idx + res ** 2
        ] = self._arena.hfield.ravel()

    def initialize_episode(self, physics, random_state):
        """Set the state of the environment at the start of each episode.

        :param physics: An instance of `Physics`.
        """
        self.set_heightfield(physics)

        # If we have a rendering context, we need to re-upload the modified
        # heightfield data.
        if physics.contexts:
            with physics.contexts.gl.make_current() as ctx:
                ctx.call(
                    mjlib.mjr_uploadHField,
                    physics.model.ptr,
                    physics.contexts.mujoco.ptr,
                    _HEIGHTFIELD_ID,
                )
        self._walker.reinitialize_pose(physics, random_state)