import numpy as np
import yaml

from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere
from environments.d3il.d3il_sim.utils.sim_path import sim_framework_path
from .gen_obj import generate_HDARObj_from_dict

init_end_eff_pos = [0.525, -0.28, 0.6]


def get_task_setting(task_type: str):
    task_setting_path = sim_framework_path(
        "demos",
        "unity",
        "human_demostration_ar",
        "hdar_task",
        "task_setting",
        f"{task_type}.yaml",
    )
    with open(task_setting_path, "r") as f:
        task_setting = yaml.load(f, Loader=yaml.FullLoader)
    return task_setting


def get_obj_list():

    task_setting_path = sim_framework_path('models/mj/common-objects/table/table_objects.yaml')

    with open(task_setting_path, "r") as f:
        task_setting = yaml.load(f, Loader=yaml.FullLoader)

    objects_config = task_setting["objects"]

    object_dict = {}

    for obj_name, obj_config in objects_config.items():
        new_obj = generate_HDARObj_from_dict(obj_name, obj_config)
        object_dict[obj_name] = new_obj

    return object_dict