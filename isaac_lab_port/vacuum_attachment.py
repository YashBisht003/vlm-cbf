from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VacuumAttachmentConfig:
    enabled: bool = True


class AutoAttachmentBackend:
    """PhysX AutoAttachment backend for GPU-friendly runtime attach/detach."""

    def __init__(self, cfg: VacuumAttachmentConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else VacuumAttachmentConfig()
        self._physx_attachment_iface = None
        try:
            import omni.physx as _omni_physx

            self._physx_attachment_iface = _omni_physx.get_physx_attachment_interface()
        except Exception:
            self._physx_attachment_iface = None

    def is_available(self) -> bool:
        if not self.cfg.enabled:
            return False
        try:
            import omni.usd  # noqa: F401
            from pxr import PhysxSchema  # noqa: F401
        except Exception:
            return False
        return self._physx_attachment_iface is not None

    def attach(self, attachment_prim_path: str, parent_rigid_body_path: str, child_rigid_body_path: str) -> str:
        if not self.cfg.enabled:
            raise RuntimeError("AutoAttachment backend is disabled.")
        if not self.is_available():
            raise RuntimeError("AutoAttachment backend is unavailable in this runtime.")

        import omni.usd
        from pxr import PhysxSchema, Sdf

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No USD stage is available for attachment creation.")

        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, Sdf.Path(attachment_prim_path))
        prim = attachment.GetPrim()
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(prim)
        attachment.CreateActor0Rel().SetTargets([Sdf.Path(parent_rigid_body_path)])
        attachment.CreateActor1Rel().SetTargets([Sdf.Path(child_rigid_body_path)])

        if self._physx_attachment_iface is None:
            raise RuntimeError(
                "PhysX attachment interface is unavailable. "
                "Refusing to create an attachment prim without computed attachment points."
            )
        self._physx_attachment_iface.compute_attachment_points(attachment_prim_path)
        return attachment_prim_path

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
