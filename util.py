"""Utility functions to convert from mocap positions to mujoco sensors."""
import numpy as np
from dm_control import mjcf
from dm_control import suite
from dm_control import composer
import h5py
import os
KEYPOINT_MODEL_PAIRS = {"ArmL": "hand_L",
                        "ArmR": "hand_R",
                        "ElbowL": "upper_arm_L",
                        "ElbowR": "upper_arm_R",
                        "HeadB": "skull",
                        "HeadF": "skull",
                        "HeadL": "skull",
                        "HipL": "pelvis",
                        "HipR": "pelvis",
                        "KneeL": "upper_leg_L",
                        "KneeR": "upper_leg_R",
                        "Offset1": "vertebra_2",
                        "Offset2": "vertebra_5",
                        "ShinL": "lower_leg_L",
                        "ShinR": "lower_leg_R",
                        "ShoulderL": "scapula_L",
                        "ShoulderR": "scapula_R",
                        "SpineF": "vertebra_1",
                        "SpineL": "vertebra_6",
                        "SpineM": "vertebra_3"}
_TIME_BINS = .03


class Physics(mjcf.Physics):
    """Physics simulation with additional features for the Rat domain."""

    def torso_upright(self):
        """Return projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat['torso', 'zz']

    def head_height(self):
        """Return the height of the head."""
        return self.named.data.site_xpos['head', 'z']

    def pelvis_height(self):
        """Return the height of the pelvis."""
        return self.named.data.xpos['pelvis', 'z']

    def head_upright(self):
        """Return projection from z-axes of head to the z-axes of world."""
        return self.named.data.xmat['skull', 'zz']

    def center_of_mass_velocity(self):
        """Return the velocity of the center-of-mass."""
        return self.named.data.subtree_linvel['torso']

    def com_velocity(self):
        """Return the COM."""
        torso_frame = self.named.data.xmat['torso'].reshape(3, 3).copy()
        return self.center_of_mass_velocity().dot(torso_frame)

    def com_forward_velocity(self):
        """Return the forward velocity of the center-of-mass."""
        return self.com_velocity()[0]

    def joint_angles(self):
        """Return the configuration without the root joint."""
        return self.data.qpos[7:]  # Skip the 7 entries of the free root joint.

    def normalised_actuated_angles(self):
        """Return the normalised configuration without the root joint."""
        normalized_angles = []
        for act_id in range(0, self.model.nu):
            if self.model.actuator_trntype[act_id] == 0:
                joint_id = self.model.actuator_trnid[act_id, 0]
                joint_range = self.model.jnt_range[joint_id, :]
                a = 2. / np.diff(joint_range)
                b = -a * np.mean(joint_range)
                joint_angle = self.data.qpos[self.model.jnt_qposadr[joint_id]]
                normalized_angles.append(a * joint_angle + b)
        return np.asarray(normalized_angles)

    def joint_velocities(self):
        """Return the velocity without the root joint."""
        return self.data.qvel[6:]  # Skip the 6 DoFs of the free root joint.

    def inertial_sensors(self):
        """Return inertial sensor data."""
        sensordata = self.named.data.sensordata[['accelerometer',
                                                 'velocimeter',
                                                 'gyro']]
        return sensordata

    def touch_sensors(self):
        """Return touch sensor data."""
        sensordata = self.named.data.sensordata[['palm_L', 'palm_R',
                                                 'sole_R', 'sole_L']]
        return np.arcsinh(sensordata)

    def extremities(self):
        """Return end effector positions in egocentric frame."""
        torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
        com_pos = self.named.data.subtree_com['torso']
        positions = []
        for side in ('_L', '_R'):
            for limb in ('hand', 'toe'):
                torso_to_limb = self.named.data.xpos[limb + side] - com_pos
                positions.append(torso_to_limb.dot(torso_frame))
        return np.hstack(positions)


class RatWithMocapSites():
    """Convert kp_data kp_data to mjcf_model sites."""

    def __init__(self, model_path, kp_data, kp_names, frame=0, name='Rat'):
        """Initialize site converter.

        :param model_path: Path to .xml file defining rat.
        :param kp_data: Numpy array of (t x n_kps).
        :param kp_names: List of names for each keypoint.
        :param frame: Frame of mocap data to initialize mjcf model.
        :param name: Name of mjcf.RootElement model.
        """
        self.modelpath = model_path
        self.kp_data = kp_data
        self.kp_names = kp_names
        # self.physics = suite.rat.Physics.from_xml_string(
        #                     *suite.rat.get_model_and_assets())
        self.mjcf_model = mjcf.from_file(self.modelpath)
        # self.mjcf_model = self.physics.model

        # Add keypoint sites to the mjcf model, and a reference to the sites as
        # an attribute for easier access
        for i, name in enumerate(self.kp_names):
            start = (np.random.rand(3)-.5)*.001
            parent = self.mjcf_model.worldbody.find('body',
                                                    KEYPOINT_MODEL_PAIRS[name])

            site = parent.add('site', name='mocap_'+name,
                                           type='sphere',
                                           size=[.005],
                                           rgba=".6 0 0 1",
                                           pos=[0, 0, 0])
            site = self.mjcf_model.worldbody.add('site', name=name,
                                                 type='sphere',
                                                 size=[.005],
                                                 rgba="0 .6 0 1",
                                                 pos=start)
            setattr(self, name, site)

        self.physics = Physics.from_mjcf_model(self.mjcf_model)

    @staticmethod
    def _kp2dim(id):
        """Keypoint ID to kp dimensions conversion."""
        return id*3 + np.array([0, 1, 2])

    def _update_sites(self, frame):
        for i, name in enumerate(self.kp_names):
            setattr(getattr(self, name), 'pos',
                    self.kp_data[frame, self._kp2dim(i)])

    def get_sites(self, frame=-1):
        """Get the mocap data represented in sites.

        :param frame: If not None, get sites at that frame.
        """
        if frame >= 0:
            self._update_sites(frame)
        return [getattr(self, name) for name in self.kp_names]


def load_kp_data_from_file(filename, struct_name='markers_preproc'):
    """Format kp_data files from matlab to python through hdf5.

    :param filename: Path to v7.3 mat file containing
    :param struct_name: Name of the struct to load.
    """
    with h5py.File(filename, 'r') as f:
        data = f[struct_name]
        kp_names = [k for k in data.keys()]

        # Concatenate the data for each keypoint, and format to (t x n_dims)
        kp_data = \
            np.concatenate([data[name][:] for name in kp_names]).T
        return kp_data, kp_names


def load_snippets_from_file(folder):
    """Load snippets from file and return list of kp_data."""
    filenames = [os.path.join(folder, f) for f in os.listdir(folder)
                 if os.path.isfile(os.path.join(folder, f))]
    snippets = [None]*len(filenames)
    for i, filename in enumerate(filenames):
        with h5py.File(filename, 'r') as f:
            data = f['data']
            kp_names = [k for k in data.keys()]
            # Concatenate the data for each keypoint,
            # and format to (t x n_dims)
            snippets[i] = \
                np.concatenate([data[name][:] for name in kp_names]).T
    return snippets, kp_names
