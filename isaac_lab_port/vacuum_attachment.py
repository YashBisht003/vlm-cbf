from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VacuumAttachmentConfig:
    enabled: bool = True


class AutoAttachmentBackend:
    """Runtime attach/detach backend.

    Prefer the legacy PhysX attachment prim path when the old attachment interface
    exists. Fall back to a fixed joint in newer Isaac installs where the deprecated
    rigid attachment interface is no longer exposed.
    """

    def __init__(self, cfg: VacuumAttachmentConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else VacuumAttachmentConfig()
        self._physx_attachment_iface = None
        self._joint_utils = None
        self._backend_kind = "unavailable"
        self._refresh_runtime_handles()

    def _refresh_runtime_handles(self) -> None:
        self._physx_attachment_iface = None
        self._joint_utils = None
        self._backend_kind = "unavailable"
        if not self.cfg.enabled:
            return
        try:
            import omni.physx as _omni_physx
        except Exception:
            return

        legacy_getter = getattr(_omni_physx, "get_physx_attachment_interface", None)
        if callable(legacy_getter):
            try:
                iface = legacy_getter()
            except Exception:
                iface = None
            if iface is not None:
                self._physx_attachment_iface = iface
                self._backend_kind = "legacy_attachment"
                return

        try:
            from omni.physx.scripts import utils as _joint_utils
        except Exception:
            _joint_utils = None
        if _joint_utils is not None and hasattr(_joint_utils, "createJoint"):
            self._joint_utils = _joint_utils
            self._backend_kind = "fixed_joint"

    def is_available(self) -> bool:
        if not self.cfg.enabled:
            return False
        try:
            import omni.usd  # noqa: F401
            from pxr import PhysxSchema, UsdPhysics  # noqa: F401
        except Exception:
            return False
        self._refresh_runtime_handles()
        return self._backend_kind in {"legacy_attachment", "fixed_joint"}

    def attach(self, attachment_prim_path: str, parent_rigid_body_path: str, child_rigid_body_path: str) -> str:
        if not self.cfg.enabled:
            raise RuntimeError("AutoAttachment backend is disabled.")
        if not self.is_available():
            raise RuntimeError("AutoAttachment backend is unavailable in this runtime.")

        import omni.usd
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No USD stage is available for attachment creation.")

        if self._backend_kind == "legacy_attachment":
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, Sdf.Path(attachment_prim_path))
            prim = attachment.GetPrim()
            PhysxSchema.PhysxAutoAttachmentAPI.Apply(prim)
            attachment.CreateActor0Rel().SetTargets([Sdf.Path(parent_rigid_body_path)])
            attachment.CreateActor1Rel().SetTargets([Sdf.Path(child_rigid_body_path)])
            if self._physx_attachment_iface is None:
                raise RuntimeError(
                    "Legacy PhysX attachment interface is unavailable. "
                    "Refusing to create an attachment prim without computed attachment points."
                )
            self._physx_attachment_iface.compute_attachment_points(attachment_prim_path)
            return attachment_prim_path

        if self._backend_kind == "fixed_joint":
            parent_prim = stage.GetPrimAtPath(Sdf.Path(parent_rigid_body_path))
            child_prim = stage.GetPrimAtPath(Sdf.Path(child_rigid_body_path))
            if not parent_prim.IsValid():
                raise RuntimeError(f"Parent rigid body prim is invalid: {parent_rigid_body_path}")
            if not child_prim.IsValid():
                raise RuntimeError(f"Child rigid body prim is invalid: {child_rigid_body_path}")

            joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(attachment_prim_path))
            xf_cache = UsdGeom.XformCache()
            child_pose = xf_cache.GetLocalToWorldTransform(child_prim).RemoveScaleShear()
            parent_pose = xf_cache.GetLocalToWorldTransform(parent_prim).RemoveScaleShear()
            rel_pose = child_pose * parent_pose.GetInverse()
            rel_pose = rel_pose.RemoveScaleShear()

            joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_rigid_body_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(child_rigid_body_path)])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_pose.ExtractTranslation()))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel_pose.ExtractRotationQuat()))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            joint.CreateBreakForceAttr().Set(3.40282347e38)
            joint.CreateBreakTorqueAttr().Set(3.40282347e38)
            return attachment_prim_path

        raise RuntimeError("No supported attachment backend is available in this runtime.")

    def detach(self, attachment_prim_path: str) -> None:
        if not self.cfg.enabled:
            return
        try:
            import omni.usd
            from pxr import Sdf
        except Exception:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        stage.RemovePrim(Sdf.Path(attachment_prim_path))
