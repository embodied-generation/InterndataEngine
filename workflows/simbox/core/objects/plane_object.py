import glob
import os
import random

from core.objects.base_object import register_object
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_current_stage

try:
    from omni.isaac.core.materials.omni_pbr import OmniPBR  # Isaac Sim 4.1.0 / 4.2.0
except ImportError:
    from isaacsim.core.api.materials import OmniPBR  # Isaac Sim 4.5.0

from pxr import UsdGeom, UsdPhysics


@register_object
class PlaneObject(XFormPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - size: [width, length] of the plane
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        prim_path = os.path.join(root_prim_path, cfg["name"])
        self.cfg = cfg

        # ===== Initialize =====
        stage = get_current_stage()
        plane_geom = UsdGeom.Plane.Define(stage, prim_path)
        plane_geom.CreateWidthAttr().Set(cfg["size"][0])
        plane_geom.CreateLengthAttr().Set(cfg["size"][1])
        super().__init__(prim_path=prim_path, name=cfg["name"], *args, **kwargs)

        self._create_collision_volume(stage, cfg)

    def _create_collision_volume(self, stage, cfg):
        if not bool(cfg.get("collision_enabled", False)):
            return

        thickness = float(cfg.get("collision_thickness", 0.02))
        if thickness <= 0.0:
            raise ValueError("collision_thickness must be positive when collision_enabled is true")

        collision_prim_path = f"{self.prim_path}/collision_volume"
        collision_geom = UsdGeom.Cube.Define(stage, collision_prim_path)
        collision_geom.CreateSizeAttr().Set(1.0)

        collision_prim = collision_geom.GetPrim()
        collision_xform = UsdGeom.Xformable(collision_prim)
        collision_xform.AddScaleOp().Set((float(cfg["size"][0]), float(cfg["size"][1]), thickness))
        collision_xform.AddTranslateOp().Set((0.0, 0.0, -0.5 * thickness))

        UsdPhysics.CollisionAPI.Apply(collision_prim)

        if not bool(cfg.get("collision_visible", False)):
            UsdGeom.Imageable(collision_prim).MakeInvisible()

    def get_observations(self):
        raise NotImplementedError

    def apply_texture(self, asset_root, cfg):
        texture_name = cfg["texture_lib"]
        texture_path_list = glob.glob(os.path.join(asset_root, texture_name, "*"))
        texture_path_list.sort()
        if cfg["apply_randomization"]:
            texture_id = random.randint(0, len(texture_path_list) - 1)
        else:
            texture_id = cfg["texture_id"]
        texture_path = texture_path_list[texture_id]
        mat_prim_path = f"{self.prim_path}/Looks/Material"
        if not is_prim_path_valid(mat_prim_path):
            self.mat = OmniPBR(
                prim_path=mat_prim_path,
                name="Material",
                texture_path=texture_path,
                texture_scale=cfg.get("texture_scale"),
            )
            self.apply_visual_material(self.mat)
        else:
            self.mat.set_texture(
                texture_path,
            )
