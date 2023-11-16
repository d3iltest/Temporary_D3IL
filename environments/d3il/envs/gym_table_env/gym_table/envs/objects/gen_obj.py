import os
import abc
from xml.etree import ElementTree as Et
import yaml
from math import radians

from environments.d3il.d3il_sim.core.sim_object.sim_object import SimObject
from environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import MjXmlLoadable
from environments.d3il.d3il_sim.utils.sim_path import sim_framework_path
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat
from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere, Cylinder
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject


def generate_HDARObj_from_dict(obj_name, obj_config: dict):
    if obj_config["source"] == "ycb":
        return HDARYCBObject(
            obj_type=obj_config["type"],
            obj_name=obj_name,
            init_pos=obj_config["init_pos"],
            init_quat=obj_config["init_quat"],
            static=obj_config.get("static", False),
            visual_only=obj_config.get("visual_only", False),
        )
    elif obj_config["source"] == "primitive_object":
        if obj_config["type"] == "Box":
            obj_class = Box
        elif obj_config["type"] == "Sphere":
            obj_class = Sphere
        return obj_class(
            name=obj_name,
            init_pos=obj_config["init_pos"],
            init_quat=obj_config["init_quat"],
            mass=obj_config.get("mass", 0.5),
            size=obj_config["size"],
            rgba=obj_config["rgba"],
            static=obj_config.get("static", False),
            visual_only=obj_config.get("visual_only", False),
        )


class HDARSimObject(abc.ABC):
    def __init__(
        self,
        obj_type: str,
        obj_name: str,
        init_pos,
        init_quat,
    ):
        if init_pos is None:
            self.init_pos = [0, 0, 0]
        else:
            self.init_pos = init_pos
            assert (
                len(init_pos) == 3
            ), "Error, parameter pos has to be three dimensional."

        if init_quat is None:
            self.init_quat = [0, 0, 0, 0]
        elif len(init_quat) == 3:
            self.init_quat = euler2quat([radians(deg) for deg in init_quat])
        elif len(init_quat) == 4:
            self.init_quat = init_quat
        else:
            assert "Error, parameter quat has to be four dimensional."

        self.type = obj_type
        self.name = obj_name


class HDARYCBObject(HDARSimObject, YCBMujocoObject):
    ycb_path = "../SFModels/YCB/models/ycb"

    def __init__(
        self,
        obj_type: str,
        obj_name: str,
        init_pos,
        init_quat,
        static=False,
        visual_only=False,
    ):
        HDARSimObject.__init__(
            self,
            obj_type,
            obj_name,
            init_pos,
            init_quat,
        )
        YCBMujocoObject.__init__(
            self,
            ycb_base_folder=sim_framework_path("models/mj/common-objects/SF-ObjectDataset/YCB/"),
            object_id=obj_type,
            object_name=obj_name,
            pos=self.init_pos,
            quat=self.init_quat,
        )
        # for mujoco model to unity prefab transformation
        self.rot_offset = [-90.0, 0.0, 90.0]
        self.static = static
        self.visual_only = visual_only

    def generate_xml(self) -> str:
        info_file = os.path.join(self.object_folder, "info.yml")
        assert os.path.isfile(
            info_file
        ), f"The file {info_file} was not found. Did you specify the path to the object folder correctly?"

        with open(info_file, "r") as f:
            info_dict = yaml.safe_load(f)

        original_file = info_dict["original_file"]
        submesh_files = info_dict["submesh_files"]
        submesh_props = info_dict["submesh_props"]
        weight = info_dict["weight"]
        material_map = info_dict["material_map"]

        root = Et.Element("mujoco", attrib={"model": self.name})

        # Assets and Worldbody
        assets = Et.SubElement(root, "asset")
        worldbody = Et.SubElement(root, "worldbody")
        body_attributes = {
            "name": f"{self.name}",
            "pos": " ".join(map(str, self.pos)),
            "quat": " ".join(map(str, self.quat)),
        }
        body = Et.SubElement(worldbody, "body", attrib=body_attributes)

        ## Texture and Material
        texture_attributes = {
            "type": "2d",
            "name": f"{self.name}_tex",
            "file": os.path.join(self.object_folder, material_map),
        }
        texture = Et.SubElement(assets, "texture", attrib=texture_attributes)

        material_attributes = {
            "name": f"{self.name}_mat",
            "texture": texture_attributes["name"],
            "specular": "0.5",
            "shininess": "0.5",
        }
        material = Et.SubElement(assets, "material", attrib=material_attributes)

        # Meshes
        orig_mesh_attributes = {
            "name": f"{self.name}_orig",
            "file": os.path.join(self.object_folder, original_file),
        }
        orig_mesh = Et.SubElement(assets, "mesh", attrib=orig_mesh_attributes)

        orig_geom_attributes = {
            "material": material_attributes["name"],
            "mesh": orig_mesh_attributes["name"],
            "group": "2",
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
        }
        orig_geom = Et.SubElement(body, "geom", attrib=orig_geom_attributes)

        for i, (submesh_file, submesh_prop) in enumerate(
            zip(submesh_files, submesh_props)
        ):
            collision_mesh_attributes = {
                "name": f"{self.name}_coll_{i}",
                "file": os.path.join(self.object_folder, submesh_file),
            }
            collision_mesh = Et.SubElement(
                assets, "mesh", attrib=collision_mesh_attributes
            )
            collision_geom_attributes = {
                "mesh": collision_mesh_attributes["name"],
                "mass": str(weight * submesh_prop),
                "group": "3",
                "type": "mesh",
                "conaffinity": "1",
                "contype": "1",
                "condim": "4",
                "friction": "0.95 0.3 0.1",
                "rgba": "1 1 1 1",
                "solimp": "0.998 0.998 0.001",
                "solref": "0.001 1",
            }
            if self.visual_only:
                collision_geom_attributes["contype"] = "0"
                collision_geom_attributes["conaffinity"] = "0"
            collision_geom = Et.SubElement(
                body, "geom", attrib=collision_geom_attributes
            )
        if not self.static:
            joint_attributes = {
                "damping": "0.0001",
                "name": f"{self.name}:joint",
                "type": "free",
            }
            joint = Et.SubElement(body, "joint", attrib=joint_attributes)

        return Et.tostring(root)