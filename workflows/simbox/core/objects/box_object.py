from core.objects.base_object import register_object
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdPhysics


@register_object
class BoxObject(XFormPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path. Present for constructor parity with other object classes.
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
        """
        del asset_root

        self.cfg = cfg
        prim_path = f"{root_prim_path}/{cfg['name']}"
        stage = get_current_stage()

        root_geom = UsdGeom.Xform.Define(stage, prim_path)
        visual_geom = UsdGeom.Cube.Define(stage, f"{prim_path}/visual")
        visual_geom.CreateSizeAttr().Set(1.0)
        visual_xform = UsdGeom.Xformable(visual_geom.GetPrim())
        visual_xform.AddTranslateOp().Set((0.0, 0.0, 0.0))
        visual_xform.AddScaleOp().Set((1.0, 1.0, 1.0))

        color = cfg.get("color")
        if isinstance(color, (list, tuple)) and len(color) == 3:
            visual_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

        collision_geom = UsdGeom.Cube.Define(stage, f"{prim_path}/collision")
        collision_geom.CreateSizeAttr().Set(1.0)
        collision_xform = UsdGeom.Xformable(collision_geom.GetPrim())
        collision_xform.AddTranslateOp().Set((0.0, 0.0, 0.0))
        collision_xform.AddScaleOp().Set((1.0, 1.0, 1.0))
        if bool(cfg.get("collision_enabled", True)):
            UsdPhysics.CollisionAPI.Apply(collision_geom.GetPrim())

        if not bool(cfg.get("collision_visible", False)):
            UsdGeom.Imageable(collision_geom.GetPrim()).MakeInvisible()

        super().__init__(prim_path=str(root_geom.GetPath()), name=cfg["name"], *args, **kwargs)

    def get_observations(self):
        translation, orientation = self.get_local_pose()
        scale = self.get_local_scale()
        return {
            "translation": translation,
            "orientation": orientation,
            "scale": scale,
        }
