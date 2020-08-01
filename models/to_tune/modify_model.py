"""Modify model dimensions.

Performs four changes to the model:
1. Global downscaling of model size.
2. Rescaling arm and leg lengths to match measured lengths.
3. Rescaling skull dimensions to match measured lengths.
4. Rescale the densities of geoms to match measured mass distribution.
"""
from dm_control import mjcf
from dm_control import viewer
from dm_control import suite
import numpy as np
import re
import pandas as pd
from dm_control.mujoco.wrapper.mjbindings import mjlib

MM_TO_METER = 1000
# Leave about 17g for some vertebrae
MODEL_MASS = 0.318


def load_length_data(lengths_path):
    """Load length measurement data."""
    length_data = pd.ExcelFile(lengths_path)
    data = length_data.parse("Sheet1")
    data = data.to_numpy()
    # Index into the part length data, and separate names and data
    data = data[5:, :]
    measured_names = data[:, 0]
    data = data[:, 1:]
    return data, measured_names


def load_mass_dist_data(mass_dist_path):
    """Load mass distribution data."""
    mass_data = pd.ExcelFile(mass_dist_path)
    data = mass_data.parse("Sheet1")
    data = data.to_numpy()
    parts = data[1:, 0]
    masses = data[1:, 2:4]
    return masses, parts


def view_model():
    """View a model with a default environment."""
    # Load an environment from the Control Suite.
    env = suite.load(domain_name="rat", task_name="stand")
    # Launch the viewer application.
    viewer.launch(env)


def load_model(model_path):
    """Load a model from path."""
    return mjcf.from_path(model_path)


def write_model(model, save_path):
    """Write a model to file."""
    s = model.to_xml_string()
    s = re.sub("rat_skin.*skn", "rat_skin.skn", s)
    with open(save_path, "w") as f:
        f.write(s)


def get_bone_distance(physics, joint_pair):
    """Measure the distance between joint pairs.

    :param joint_pair: List of physics.named.data.size_xpos names
    """
    joint0 = physics.named.data.site_xpos[joint_pair[0]].copy()
    joint1 = physics.named.data.site_xpos[joint_pair[1]].copy()
    length = np.sqrt(np.sum((joint0 - joint1) ** 2))
    return length


def get_bone_ratios(bone_dict):
    """Measure the ratios between bone pairs.

    :param bone_dict: Dictionary of bone-length pairs
    """
    n_bones = len(bone_dict.keys())
    ratio_mat = np.zeros((n_bones, n_bones))
    ratio_dict = {}
    for i, (bone0, length0) in enumerate(bone_dict.items()):
        for j, (bone1, length1) in enumerate(bone_dict.items()):
            ratio = length0 / length1
            ratio_mat[i, j] = ratio
            ratio_dict[bone0 + "-" + bone1] = ratio
    return ratio_dict, ratio_mat


def get_skull_dims(physics):
    """Measure the dimensions of the skull."""
    atlas_pos = physics.named.data.xpos["vertebra_atlant"].copy()
    T0_pos = physics.named.data.geom_xpos["skull_T0_collision"].copy()
    length = np.sqrt(np.sum((atlas_pos - T0_pos) ** 2)) * MM_TO_METER

    eyeL_pos = physics.named.data.geom_xpos["eye_L_collision"].copy()
    eyeR_pos = physics.named.data.geom_xpos["eye_R_collision"].copy()
    width = np.sqrt(np.sum((eyeL_pos - eyeR_pos) ** 2)) * MM_TO_METER
    return {"length": length, "width": width}


# Globally scale down the model
# TODO(scale_ratio): global_scale_ratio was calculated early on based on the
# ratio of long bone (humerus, radius, ulna, tibia, femur) lengths between
# the model and skeletons/data. Should  automate.
def scale_model(model, global_scale_ratio=0.82):
    """Globally downscale the model by the global_scale_ratio."""
    for g in model.find_all("geom"):
        if g.pos is not None and "eye" not in g.name:
            g.pos *= global_scale_ratio
    for b in model.find_all("body"):
        if b.pos is not None and "eye" not in b.name:
            b.pos *= global_scale_ratio
    for s in model.find_all("site"):
        if s.pos is not None and "eye" not in s.name:
            s.pos *= global_scale_ratio
    return model


def scale_arms_and_legs(model, joint_pairs, bone_lengths):
    """Scale particular arm and leg joints to match data."""
    env = suite.load(domain_name="rat", task_name="stand")
    model_lengths = {
        k: get_bone_distance(env.physics, jp) * MM_TO_METER
        for k, jp in joint_pairs.items()
    }
    ratio = [bone_lengths[k] / model_lengths[k] for k in model_lengths.keys()]

    # List all of the geoms, bodies, and sites that should be modified to
    # account for changing bone lengths
    model_name_pairs = {
        "humerus": ["elbow", "humerus", "lower_arm"],
        "radius": ["wrist", "radius", "ulna", "hand"],
        "femur": [
            "knee",
            "upper_leg_L0_collision",
            "upper_leg_R0_collision",
            "lower_leg",
        ],
        "tibia": ["ankle", "foot"],
        "metatarsal": ["toe"],
        "hand": ["finger", "hand_L_collision", "hand_R_collision"],
    }

    # For each bone, scale each geom, body, or site to match the ratios in real
    # data
    for i, (bone, model_id) in enumerate(model_name_pairs.items()):
        for g in model.find_all("geom"):
            if any(part in g.name for part in model_id):
                if bone == "radius" and any(
                    part in g.name for part in ["hand_L_collision", "hand_R_collision"]
                ):
                    continue
                if g.pos is not None:
                    g.pos *= ratio[i]
                if g.size is not None:
                    g.size *= ratio[i]
        for b in model.find_all("body"):
            if any(part in b.name for part in model_id):
                if b.pos is not None:
                    b.pos *= ratio[i]
        for s in model.find_all("site"):
            if any(part in s.name for part in model_id):
                if s.pos is not None:
                    s.pos *= ratio[i]
    return model


def scale_skull(model):
    """Scale the skull to match data."""
    env = suite.load(domain_name="rat", task_name="stand")
    model_dims = get_skull_dims(env.physics)
    ratio = [skull_dims[k] / model_dims[k] for k in model_dims.keys()]
    model_name_pairs = {
        "length": ["jaw", "skull", "eye"],
        "width": ["jaw", "skull", "eye"],
    }
    for i, (bone, model_id) in enumerate(model_name_pairs.items()):
        for g in model.find_all("geom"):
            if any(part in g.name for part in model_id):
                if g.pos is not None:
                    g.pos[i] *= ratio[i]
                if "eye" in g.name:
                    continue
                if g.size is not None:
                    g.size[i] *= ratio[i]
        for b in model.find_all("body"):
            if any(part in b.name for part in model_id):
                if b.pos is not None:
                    b.pos[i] *= ratio[i]
        for s in model.find_all("site"):
            if any(part in s.name for part in model_id):
                if s.pos is not None:
                    s.pos[i] *= ratio[i]
    return model


def scale_densities(expected_masses, geom_mass_key_pairs):
    """Scale geom densities to match data mass distribution.

    :param expected_masses: Dictionary of target masses.
    :param geom_mass_key_pairs: Dictionary linking measured names to
                                corresponding model names
    """
    part_mass = 0
    env = suite.load(domain_name="rat", task_name="stand")
    model_masses = env.physics.named.model.body_mass
    model_names = model_masses.axes.row.names
    for model_kw, real_kw in geom_mass_key_pairs.items():
        for name in model_names:
            if model_kw in name:
                part_mass += model_masses[name]
        mass_ratio = expected_masses[real_kw] / part_mass
        for g in model.find_all("geom"):
            if model_kw in g.name:
                if g.density is None:
                    # TODO(automate_default_density):
                    # taken from collision class
                    g.density = 500.0
                g.density = g.density * mass_ratio
        part_mass = 0
    return model


if __name__ == "__main__":
    # Base_model_path is the path to the original model as it was on may 17th
    base_model_path = (
        "/home/diego/code/olveczky/dm/stac/models/to_tune/rat_may17_radians.xml"
    )

    # new_model_path is the model used by /dm_control/suite/rat.py.
    # I modified the xml_path of rat.py so that I could iterate easier.
    new_model_path = "/home/diego/.envs/mujoco200_3.7/lib/python3.6/site-packages/dm_control/suite/rat_temp.xml"

    # Path to bone length measurements
    lengths_path = "/home/diego/data/dm/stac/body_measurements/Length_measurements.xlsx"

    # Path to mass distribution measurements
    mass_dist_path = "/home/diego/data/dm/stac/body_measurements/Mass_distribution.xlsx"

    # Load in the length data
    data, measured_names = load_length_data(lengths_path)
    # Average across all individuals
    data = np.nanmean(data, axis=1)

    # Calculate the ideal lengths for each part by
    # Averaging between left and right pairs.
    measured_lengths = {key: val for key, val in zip(measured_names, data)}
    ideal_lengths = {}
    for key, length in measured_lengths.items():
        if any([side in key for side in ["_L", "_R"]]):
            right = key[:-2] + "_R"
            left = key[:-2] + "_L"
            length = (measured_lengths[right] + measured_lengths[left]) / 2.0
            key = key[:-2]
        ideal_lengths[key] = np.round(length, decimals=1)

    # Link parts to the appropriate length
    # TODO(HANDS): Hand structure is a little strange in the model,
    # namely the radius and ulna do not meet in the wrist,
    # and it makes the hands look strange if using realistic
    # dimensions of ~5-6 mm
    bone_lengths = {
        "humerus": ideal_lengths["humerus"],
        "radius": ideal_lengths["wrist_olecranon"],
        "femur": ideal_lengths["femur"],
        "tibia": ideal_lengths["tibia"],
        "metatarsal": ideal_lengths["ankle_pad"],
        "hand": 9.0,
    }
    skull_dims = {
        "length": ideal_lengths["skull_length"],
        "width": ideal_lengths["skull_width"],
    }

    # Link parts to the pairs of sites needed to estimate part length.
    joint_pairs = {
        "humerus": ["shoulder_L", "elbow_L"],
        "radius": ["elbow_L", "wrist_L"],
        "femur": ["hip_L", "knee_L"],
        "tibia": ["knee_L", "ankle_L"],
        "metatarsal": ["ankle_L", "toe_L"],
        "hand": ["wrist_L", "finger_L"],
    }
    skull_pairs = {
        "length": ["head", "skull_T0_collision"],
        "width": ["eye_R_collision", "eye_L_collision"],
    }

    # Load the base model and set the temporary model to the base model.
    base_model = load_model(base_model_path)
    write_model(base_model, new_model_path)

    # Scale the model and set the temporary model to the scaled model.
    model = scale_model(base_model)
    write_model(model, new_model_path)

    # Scale the model arms and legs to match data.
    model = scale_arms_and_legs(model, joint_pairs, bone_lengths)
    write_model(model, new_model_path)

    # Scale the skull to match data
    model = scale_skull(model)
    write_model(model, new_model_path)

    # Load in the mass distribution
    masses, parts = load_mass_dist_data(mass_dist_path)
    real_ratios = {
        name: val / np.sum(masses[0, :])
        for name, val in zip(parts, np.sum(masses, axis=1))
    }

    # Desired model mass (does not account for the weight of some vertebrae)
    expected_masses = {}
    for key, ratio in real_ratios.items():
        if any([side in key for side in ["_L", "_R"]]):
            right = key[:-2] + "_R"
            left = key[:-2] + "_L"
            ratio = (real_ratios[right] + real_ratios[left]) / 2.0
        expected_masses[key] = ratio * MODEL_MASS

    # Keys are the keywords that correspond to geom(s) in the model.
    # Values are keywords for the corresponding part in real data.
    geom_mass_key_pairs = {
        "torso": "torso",
        "pelvis": "gut_liver",
        "upper_leg_L": "upper_limb_L",
        "upper_leg_R": "upper_limb_R",
        "lower_leg_L": "lower_limb_L",
        "lower_leg_R": "lower_limb_R",
        "foot_L": "foot_L",
        "foot_R": "foot_R",
        "skull": "skull",
        "jaw": "jaw",
        "scapula_L": "scapula_L",
        "scapula_R": "scapula_R",
        "upper_arm_L": "humerus_L",
        "upper_arm_R": "humerus_R",
        "lower_arm_R": "forelimb_R",
        "lower_arm_L": "forelimb_L",
        "hand_L": "paw_L",
        "hand_R": "paw_R",
        "vertebra_C": "Tail",
    }

    model = scale_densities(expected_masses, geom_mass_key_pairs)

    write_model(model, new_model_path)
    env = suite.load(domain_name="rat", task_name="stand")
    mass = mjlib.mj_getTotalmass(env.physics.model.ptr)
    print("Total mass is: %f kilograms" % (mass))
