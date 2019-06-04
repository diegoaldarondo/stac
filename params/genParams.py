"""Generate animal parameter .yaml files."""
import yaml
import sys
sys.path.insert('/home/diego/code/olveczky/dm/stac')
from optimize_scale_factor import optimize_scale_factor

out_path = "./params/baseParams.yaml"
param_base_path = '/home/diego/code/olveczky/dm/stac/params/baseParams.yaml'
data_path = \
    '/home/diego/data/dm/stac/Restpose_files/nolj_Recording_day8_caff3_nolj_imputed_JDM25.mat'
start_frame = 30900
LEFT_LEG = "0 0 .3 1"
RIGHT_LEG = ".3 0 0 1"
LEFT_ARM = "0 0 .8 1"
RIGHT_ARM = ".8 0 0 1"
HEAD = ".8 .8 0 1"
SPINE = ".8 .8 .8 1"
data = dict(
    _XML_PATH='/home/diego/code/olveczky/dm/stac/models/rat_june3.xml',
    _PARTS_TO_ZERO=['toe', 'ankle'],
    _KEYPOINT_MODEL_PAIRS={"ArmL": "hand_L",
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
                           "SpineM": "vertebra_1"},
    _KEYPOINT_COLOR_PAIRS={"ArmL": LEFT_ARM,
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
                           "SpineM": SPINE},
    _KEYPOINT_INITIAL_OFFSETS={"ArmL": "-0.01 0.005 0.01",
                               "ArmR": "-0.01 -0.005 0.01",
                               "ElbowL": "0. 0.01 0.",
                               "ElbowR": "0. -0.01 0.",
                               "HeadB": "0. -.025 .04",
                               "HeadF": ".025 -.025 .04",
                               "HeadL": "0. .025 .04",
                               "HipL": "0. 0.015 0.015",
                               "HipR": "0. -0.015 0.015",
                               "KneeL": "0.01 0.015 0.015",
                               "KneeR": "0.01 -0.015 0.015",
                               "Offset1": "0.015 .025 -0.005",
                               "Offset2": "-0.015 .025 -0.005",
                               "ShinL": "0.02 0.015 0.02",
                               "ShinR": "0.02 -0.015 0.02",
                               "ShoulderL": "0. 0. 0.",
                               "ShoulderR": "0. 0. 0.",
                               "SpineF": "0. 0. 0.0225",
                               "SpineL": "0. 0. 0.02",
                               "SpineM": "0. 0. 0.015"},
    _TIME_BINS=.03,
    _MANDIBLE_POS=-.297,
    _FTOL=1e-4,
    _ROOT_FTOL=1e-8,
    _LIMB_FTOL=0.000001,
    _DIFF_STEP=3e-8,
    _SITES_TO_REGULARIZE=["ArmL", "ArmR", "ElbowL",
                          "ElbowR", "KneeL", "KneeR",
                          "ShinL", "ShinR"],
    _IS_LIMB=['scapula', 'hip', 'knee',
              'shoulder', 'elbow'],
    _ARM_JOINTS=['shoulder', 'scapula', 'elbow', 'wrist', 'finger'],
    _ARENA_DIAMETER=0.5842,
    _STAND_HEIGHT=1.5,
    _USE_HFIELD=1,
    hfield_image_path='/home/diego/code/olveczky/dm/stac/params/floor_models/test_floormap.p',
    q_reg_coef=0.,
    m_reg_coef=.9,
    scale_factor=optimize_scale_factor(data_path,
                                       param_base_path, start_frame),
    z_perc=5,
    z_offset=.02,
    adaptive_z_offset_value=.02,
    temporal_reg_coef=.2
)

with open(out_path, 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
