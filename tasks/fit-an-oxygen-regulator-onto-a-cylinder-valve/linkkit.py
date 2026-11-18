"""linkkit: one PhysX articulation built from rigid parts on revolute and prismatic joints, for yam_arena task files.
It adds what bodykit's Linkage lacks for a screw clamp on a loose device:

- prismatic joints (a part that slides along an axis of the root part), with a drive (a spring toward a target the
  task file may move while the run goes on: ``set_target``) and limits;
- Coulomb joint friction the task file may change while the run goes on (``set_friction``);
- a loose (floating) root, so the whole device can be picked up and carried.

Every part's frame is world-aligned at the start, with its origin at ``pos``; its shapes (bodykit BoxShape, CylShape,
SphereShape, CapsuleShape) are given in that frame. Each joint's axis is a world axis at the start, through
``anchor``. Positions of revolute joints are radians, of prismatic joints metres (as Isaac Lab reports them). Parts of
one assembly do not collide with each other. Import this file next to the task file (it needs bodykit):

    from linkkit import Part, Assembly
    REG = Assembly.build("regulator", parts=[...], floating=True)

``score()`` receives an :class:`AssemblyState`: the root part's pose, every joint's position and every part's pose.
Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

from bodykit import BoxShape, CapsuleShape, CylShape, SphereShape

__all__ = ["Assembly", "AssemblyState", "Part", "ids", "install", "set_friction", "set_target"]


@dataclass(frozen=True)
class Part:
    """A rigid part. ``parent`` None: the articulation's root (exactly one part). Otherwise it hangs from ``parent`` (a
    part listed before it) by a ``joint``: "revolute" (about the world ``axis`` through ``anchor``) or "prismatic"
    (sliding along the world ``axis``). ``limits``: (low, high) in degrees (revolute) or metres (prismatic).
    ``stiffness`` (N m/rad or N/m) springs it toward its target (0 at the start; see :func:`set_target`) with
    ``damping``; ``friction`` (N m or N) is its Coulomb friction."""

    name: str
    shapes: tuple
    mass: float
    pos: tuple
    parent: str | None = None
    joint: str = "revolute"
    axis: str = "Z"
    anchor: tuple | None = None
    limits: tuple | None = None
    stiffness: float = 0.0
    damping: float = 0.0
    friction: float = 0.0
    material: tuple = (0.6, 0.5)  # (static, dynamic) friction of its shapes
    linear_damping: float = 0.0
    angular_damping: float = 0.05


@dataclass(frozen=True)
class Assembly(Box):
    """The articulation. It subclasses Box only so yam_arena's task-file checks accept it: ``pos`` is the root part's
    origin, ``size`` the bounding box of every part, ``mass`` the root's mass. ``floating``: the root is a loose body
    (else fixed to the world)."""

    parts: tuple = ()
    floating: bool = True
    solver_iterations: int = 32
    velocity_iterations: int = 4
    contact_offset: float = 0.001

    @staticmethod
    def build(name: str, parts, color=(0.5, 0.5, 0.5), **kw) -> "Assembly":
        root = [p for p in parts if p.parent is None]
        if len(root) != 1:
            raise ValueError(f"{name}: exactly one part must have no parent")
        lo, hi = _bounds(parts)
        return Assembly(name=name, pos=tuple(float(v) for v in root[0].pos), size=tuple((hi - lo).tolist()),
                        mass=float(root[0].mass), color=color, parts=tuple(parts), **kw)

    @property
    def free(self) -> bool:  # never jittered by yam_arena
        return False

    def part(self, name: str) -> Part:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)

    def local_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = _bounds(self.parts)
        return lo - np.asarray(self.pos), hi - np.asarray(self.pos)

    def _check(self) -> None:
        super()._check()
        seen = set()
        for p in self.parts:
            if p.parent is not None and p.parent not in seen:
                raise ValueError(f"{self.name}: part {p.name!r}'s parent {p.parent!r} must come before it")
            if p.joint not in ("revolute", "prismatic"):
                raise ValueError(f"{self.name}: part {p.name!r}: unknown joint {p.joint!r}")
            seen.add(p.name)


def _bounds(parts) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for p in parts:
        for s in p.shapes:
            c = np.asarray(p.pos, float) + np.asarray(s.centre, float)
            lo.append(c - s.half_extent())
            hi.append(c + s.half_extent())
    return np.min(lo, axis=0), np.max(hi, axis=0)


@dataclass(frozen=True)
class AssemblyState(ObjectState):
    """The root part's pose and bounds; ``joints``: each jointed part's joint position (rad or m) by part name;
    ``poses``: each part's (position, quaternion xyzw) in the world (env frame)."""

    joints: dict = field(default_factory=dict)
    poses: dict = field(default_factory=dict)


# -- spawning (needs Isaac Sim) -----------------------------------------------------------------------------------

_SPECS: dict = {}


def _spawn(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas.schemas import create_world_fixed_joint
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Assembly = _SPECS[cfg.body_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    origin = np.asarray(spec.pos, float)
    looks: dict = {}
    mats: dict = {}

    def look(color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            surface = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            surface.func(path, surface)
            looks[color] = path
        return looks[color]

    def mat(fr) -> str:
        fr = (float(fr[0]), float(fr[1]))
        if fr not in mats:
            path = f"{prim_path}/Looks/m{len(mats)}"
            m = sim_utils.RigidBodyMaterialCfg(static_friction=fr[0], dynamic_friction=fr[1])
            m.func(path, m)
            mats[fr] = path
        return mats[fr]

    paths: dict = {}
    for part in spec.parts:
        path = f"{prim_path}/{part.name}"
        paths[part.name] = path
        xform = UsdGeom.Xform.Define(stage, path)
        rel = np.asarray(part.pos, float) - origin
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(v) for v in rel]))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(part.mass))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.5)
        rb.CreateLinearDampingAttr(float(part.linear_damping))
        rb.CreateAngularDampingAttr(float(part.angular_damping))
        for i, shape in enumerate(part.shapes):
            if isinstance(shape, BoxShape):
                geom = UsdGeom.Cube.Define(stage, f"{path}/s{i}")
                geom.CreateSizeAttr(1.0)
                geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
                qx, qy, qz, qw = shape.quat()
                if (qx, qy, qz) != (0.0, 0.0, 0.0):
                    geom.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
                geom.AddScaleOp().Set(Gf.Vec3f(*shape.size))
            elif isinstance(shape, SphereShape):
                geom = UsdGeom.Sphere.Define(stage, f"{path}/s{i}")
                geom.CreateRadiusAttr(shape.radius)
                geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
            elif isinstance(shape, CapsuleShape):
                geom = UsdGeom.Capsule.Define(stage, f"{path}/s{i}")
                geom.CreateAxisAttr(shape.axis)
                geom.CreateRadiusAttr(shape.radius)
                geom.CreateHeightAttr(shape.height)
                geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
            elif isinstance(shape, CylShape):
                geom = UsdGeom.Cylinder.Define(stage, f"{path}/s{i}")
                geom.CreateAxisAttr(shape.axis)
                geom.CreateRadiusAttr(shape.radius)
                geom.CreateHeightAttr(shape.height)
                geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
            else:
                raise TypeError(f"{part.name}: unknown shape {shape!r}")
            gpath = str(geom.GetPrim().GetPath())
            bind_visual_material(gpath, look(tuple(shape.color)), stage=stage)
            if shape.collide:
                UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
                col = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
                col.CreateContactOffsetAttr(float(spec.contact_offset))
                col.CreateRestOffsetAttr(0.0)
                bind_physics_material(gpath, mat(part.material), stage=stage)

    for part in spec.parts:
        if part.parent is None:
            continue
        parent = spec.part(part.parent)
        anchor = np.asarray(part.anchor if part.anchor is not None else part.pos, float)
        jp = f"{paths[part.name]}/{part.name}_joint"
        if part.joint == "revolute":
            j = UsdPhysics.RevoluteJoint.Define(stage, jp)
        else:
            j = UsdPhysics.PrismaticJoint.Define(stage, jp)
        j.CreateAxisAttr(part.axis)
        if part.limits is not None:
            j.CreateLowerLimitAttr(float(part.limits[0]))
            j.CreateUpperLimitAttr(float(part.limits[1]))
        j.CreateBody0Rel().SetTargets([Sdf.Path(paths[parent.name])])
        j.CreateBody1Rel().SetTargets([Sdf.Path(paths[part.name])])
        j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in anchor - np.asarray(parent.pos, float)]))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in anchor - np.asarray(part.pos, float)]))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        if part.stiffness or part.damping:
            kind = "angular" if part.joint == "revolute" else "linear"
            per = math.pi / 180.0 if kind == "angular" else 1.0  # USD angular drives take N m per degree
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), kind)
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(part.stiffness) * per)
            drive.CreateDampingAttr(float(part.damping) * per)
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateMaxForceAttr(1.0e6)

    root_part = [p for p in spec.parts if p.parent is None][0]
    root_prim = stage.GetPrimAtPath(paths[root_part.name])
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    art = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(int(spec.solver_iterations))
    art.CreateSolverVelocityIterationCountAttr(int(spec.velocity_iterations))
    if not spec.floating:
        create_world_fixed_joint(root_prim, stage)
    return root.GetPrim()


def _object(spec: Assembly):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class AssemblySpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn)
        body_name: str = ""

    class AssemblyObject(Object):
        def _generate_articulation_cfg(self):
            actuators = {}
            for p in spec.parts:
                if p.parent is None:
                    continue
                actuators[p.name] = ImplicitActuatorCfg(
                    joint_names_expr=[f"{p.name}_joint"], stiffness=float(p.stiffness), damping=float(p.damping),
                    friction=float(p.friction), dynamic_friction=float(p.friction), viscous_friction=0.0)
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            return self._add_initial_pose_to_cfg(cfg)

    return AssemblyObject(
        name=spec.name,
        object_type=ObjectType.ARTICULATION,
        spawner_cfg=AssemblySpawnerCfg(body_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )


def as_numpy(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


_IDS: dict = {}


def ids(asset, spec: Assembly) -> dict:
    """{"joint": {part: joint index}, "body": {part: body index}} for this articulation (cached)."""
    if spec.name not in _IDS:
        jn, bn = list(asset.joint_names), list(asset.body_names)
        _IDS[spec.name] = {"joint": {p.name: jn.index(f"{p.name}_joint") for p in spec.parts if p.parent is not None},
                           "body": {p.name: bn.index(p.name) for p in spec.parts}}
    return _IDS[spec.name]


def _state(environment, spec: Assembly) -> AssemblyState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = as_numpy(base.scene.env_origins[0])
    idx = ids(asset, spec)
    q = as_numpy(asset.data.joint_pos)[0]
    bp = as_numpy(asset.data.body_pos_w)[0] - origin
    bq = as_numpy(asset.data.body_quat_w)[0]
    root = [p for p in spec.parts if p.parent is None][0]
    i0 = idx["body"][root.name]
    lo, hi = spec.local_bounds()
    pos = bp[i0]
    return AssemblyState(pos=pos, quat_xyzw=bq[i0], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pos + lo,
                         high=pos + hi, joints={n: float(q[i]) for n, i in idx["joint"].items()},
                         poses={n: (bp[i], bq[i]) for n, i in idx["body"].items()})


def set_friction(env, spec: Assembly, part: str, value: float) -> None:
    """Write the Coulomb friction (static and dynamic, N or N m) of ``part``'s joint into the simulation."""
    import torch

    base = getattr(env, "unwrapped", env)
    asset = base.scene[spec.name]
    j = ids(asset, spec)["joint"][part]
    val = torch.tensor([[float(value)]], dtype=torch.float32, device=asset.device)
    asset.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=val, joint_dynamic_friction_coeff=val,
                                                        joint_ids=[j])


def set_target(env, spec: Assembly, part: str, value: float) -> None:
    """Move the drive target (m or rad) of ``part``'s joint; it reaches the simulation with the next physics step."""
    import torch

    base = getattr(env, "unwrapped", env)
    asset = base.scene[spec.name]
    j = ids(asset, spec)["joint"][part]
    val = torch.tensor([[float(value)]], dtype=torch.float32, device=asset.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asset.set_joint_position_target_index(target=val, joint_ids=[j])


# -- hooking yam_arena.scene ----------------------------------------------------------------------------------------


def _patch_scene(scene) -> None:
    if getattr(scene, "_linkkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Assembly):
            _SPECS[spec.name] = spec
            return _object(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _state(environment, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._linkkit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_scene(module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena.scene (now if already imported, else when it is). Idempotent."""
    if "yam_arena.scene" in sys.modules:
        _patch_scene(sys.modules["yam_arena.scene"])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
