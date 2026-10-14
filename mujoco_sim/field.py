"""
RoboCup Humanoid League soccer field for MuJoCo.

Port of the Webots `RobocupSoccerField.proto` and `RobocupGoal.proto` (R2025a,
https://github.com/cyberbotics/webots/tree/R2025a/projects/samples/contests/robocup/protos,
Apache License 2.0). The field markings are flat boxes instead of the proto's
triangle meshes, and the turf physics (contact material) is not reproduced.

Coordinates: the origin is the centre of the field, x runs from the home goal
(x = -length/2) to the away goal (x = +length/2), y is across the field, z up.
The floor surface is at z = 0.
"""

import mujoco
import numpy as np

# Official dimensions [m] (RoboCup Humanoid League 2021 rules), keyed by field size.
#   length / width    playing area, measured to the outer edge of the boundary lines
#   border            strip around the playing area
#   goal_depth        depth of the goal behind the goal line
#   goal_width        inner distance between the goal posts
#   goal_area_*       goal area (depth from the goal line, width)
#   penalty_mark      distance from the goal line to the penalty mark
#   circle_diameter   centre circle, measured to the inner edge of its line
#   penalty_area_*    penalty area (depth from the goal line, width)
#   goal_height       height of the goal opening
FIELD_DIMENSIONS = {
    "kid": dict(length=9.0, width=6.0, border=1.0, goal_depth=0.6, goal_width=2.6,
                goal_area_depth=1.0, goal_area_width=3.0, penalty_mark=1.5,
                circle_diameter=1.5, penalty_area_depth=2.0, penalty_area_width=5.0,
                goal_height=1.2),
    "adult": dict(length=14.0, width=9.0, border=1.0, goal_depth=0.6, goal_width=2.6,
                  goal_area_depth=1.0, goal_area_width=4.0, penalty_mark=2.1,
                  circle_diameter=3.0, penalty_area_depth=3.0, penalty_area_width=6.0,
                  goal_height=1.8),
}

LINE_WIDTH = 0.05
MARK_BRANCH = 0.1          # half-length of the centre / penalty mark crosses
CIRCLE_SEGMENTS = 36       # sides of the polygon approximating the centre circle
LINE_THICKNESS = 0.001     # lines are flat boxes this tall, half-sunk into the floor

GRASS_RGBA = [0.3, 0.6, 0.2, 1.0]
LINE_RGBA = [1.0, 1.0, 1.0, 1.0]
NET_RGBA = [0.3, 0.3, 0.3, 0.4]

POST_RADIUS = 0.05
BACKPOST_RADIUS = 0.02
_COLLISION_GROUP = 3       # hidden in the viewer by default


def field_dimensions(size: str) -> dict[str, float]:
    """Dimensions of the "kid" or "adult" field."""
    if size not in FIELD_DIMENSIONS:
        raise ValueError(f"field size must be one of {sorted(FIELD_DIMENSIONS)}, got {size!r}")
    return FIELD_DIMENSIONS[size]


def add_field(spec, size: str = "kid") -> None:
    """Add the field markings and both goals to an `mujoco.MjSpec`.

    The spec must already contain a floor plane at z = 0; it is resized and
    painted to cover the field and its border, and its collision is unchanged.
    """
    dim = field_dimensions(size)

    floor = next(g for g in spec.worldbody.geoms if g.name == "floor")
    floor.size = [(dim["length"] / 2 + dim["border"]), (dim["width"] / 2 + dim["border"]), 0.1]
    floor.rgba = GRASS_RGBA

    _add_markings(spec, dim)
    _add_goal(spec, dim, side=-1, name="goal_home")
    _add_goal(spec, dim, side=+1, name="goal_away")


# -----------------------------------------------------------------------------
#  Markings
# -----------------------------------------------------------------------------

def _rect(spec, name: str, x0: float, x1: float, y0: float, y1: float) -> None:
    """Flat white rectangle on the floor spanning [x0, x1] x [y0, y1]."""
    spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[(x0 + x1) / 2, (y0 + y1) / 2, 0.0],
        size=[(x1 - x0) / 2, (y1 - y0) / 2, LINE_THICKNESS / 2],
        rgba=LINE_RGBA, contype=0, conaffinity=0)


def _outline(spec, name: str, x0: float, x1: float, y0: float, y1: float) -> None:
    """Rectangle outline whose outer edge is [x0, x1] x [y0, y1]; the line grows inwards."""
    w = LINE_WIDTH
    _rect(spec, f"{name}_south", x0, x1, y0, y0 + w)
    _rect(spec, f"{name}_north", x0, x1, y1 - w, y1)
    _rect(spec, f"{name}_west", x0, x0 + w, y0, y1)
    _rect(spec, f"{name}_east", x1 - w, x1, y0, y1)


def _cross(spec, name: str, cx: float, cy: float) -> None:
    """Small cross (centre mark, penalty mark) centred on (cx, cy)."""
    b, w = MARK_BRANCH, LINE_WIDTH
    _rect(spec, f"{name}_h", cx - b, cx + b, cy - w / 2, cy + w / 2)
    _rect(spec, f"{name}_v", cx - w / 2, cx + w / 2, cy - b, cy + b)


def _add_markings(spec, dim: dict[str, float]) -> None:
    half_l, half_w = dim["length"] / 2, dim["width"] / 2
    w = LINE_WIDTH

    _outline(spec, "boundary", -half_l, half_l, -half_w, half_w)
    _rect(spec, "halfway_line", -w / 2, w / 2, -half_w, half_w)

    # Centre circle: line occupies radius [d/2, d/2 + w], drawn as a polygon
    r_mid = dim["circle_diameter"] / 2 + w / 2
    for k in range(CIRCLE_SEGMENTS):
        a0 = 2 * np.pi * k / CIRCLE_SEGMENTS
        a1 = 2 * np.pi * (k + 1) / CIRCLE_SEGMENTS
        p0 = r_mid * np.array([np.cos(a0), np.sin(a0)])
        p1 = r_mid * np.array([np.cos(a1), np.sin(a1)])
        mid, chord = (p0 + p1) / 2, p1 - p0
        yaw = np.arctan2(chord[1], chord[0])
        spec.worldbody.add_geom(
            name=f"center_circle_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[mid[0], mid[1], 0.0],
            size=[np.linalg.norm(chord) / 2 + w / 2, w / 2, LINE_THICKNESS / 2],
            quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
            rgba=LINE_RGBA, contype=0, conaffinity=0)

    _cross(spec, "center_mark", 0.0, 0.0)

    for sign, team in ((-1, "home"), (+1, "away")):
        goal_line = sign * half_l
        for area, depth, width in (("goal_area", dim["goal_area_depth"], dim["goal_area_width"]),
                                   ("penalty_area", dim["penalty_area_depth"],
                                    dim["penalty_area_width"])):
            xs = sorted((goal_line, goal_line - sign * depth))
            _outline(spec, f"{team}_{area}", xs[0], xs[1], -width / 2, width / 2)
        _cross(spec, f"{team}_penalty_mark", goal_line - sign * dim["penalty_mark"], 0.0)


# -----------------------------------------------------------------------------
#  Goal
# -----------------------------------------------------------------------------

def _add_goal(spec, dim: dict[str, float], side: int, name: str) -> None:
    """Goal on the goal line at x = side * length/2, opening towards the field.

    Built in the goal's local frame (local -x points behind the goal, as in the
    proto) and mapped to the world: the away goal is the home goal rotated by pi.
    """
    c, r, br = dim["goal_depth"], POST_RADIUS, BACKPOST_RADIUS
    spacing = dim["goal_width"] + 2 * r           # distance between post centres
    height = dim["goal_height"] + r               # top of the crossbar
    x0 = side * dim["length"] / 2

    def pos(lx, ly, lz):
        return [x0 - side * lx, -side * ly, lz]   # away goal = home goal rotated by pi about z

    def add(label, geom_type, local_pos, size=None, fromto=None, rgba=LINE_RGBA,
            collide=False, hidden=False):
        kwargs = dict(name=f"{name}_{label}", type=geom_type, rgba=rgba)
        if fromto is not None:
            kwargs["fromto"] = [*pos(*fromto[:3]), *pos(*fromto[3:])]
            kwargs["size"] = [size, 0.0, 0.0]
        else:
            kwargs["pos"] = pos(*local_pos)
            kwargs["size"] = size
        if collide:
            kwargs.update(contype=1, conaffinity=1)
        else:
            kwargs.update(contype=0, conaffinity=0)
        if hidden:
            kwargs["group"] = _COLLISION_GROUP
        spec.worldbody.add_geom(**kwargs)

    # Front frame: two posts and the crossbar (these also collide)
    for ly, tag in ((-spacing / 2, "right"), (spacing / 2, "left")):
        add(f"{tag}_post", mujoco.mjtGeom.mjGEOM_CYLINDER, (0, ly, height / 2),
            size=[r, height / 2, 0.0], collide=True)
    add("crossbar", mujoco.mjtGeom.mjGEOM_CAPSULE, None, size=r,
        fromto=(0, -spacing / 2, height, 0, spacing / 2, height), collide=True)

    # Back frame: thin posts at the back and rods along the top and bottom
    for ly, tag in ((-spacing / 2, "right"), (spacing / 2, "left")):
        add(f"back_{tag}_post", mujoco.mjtGeom.mjGEOM_CYLINDER, (-c, ly, height / 2),
            size=[br, height / 2, 0.0])
        for lz, level in ((height, "top"), (br / 2, "bottom")):
            add(f"{tag}_rod_{level}", mujoco.mjtGeom.mjGEOM_CAPSULE, None, size=br,
                fromto=(-c, ly, lz, 0, ly, lz))
    for lz, level in ((height, "top"), (br / 2, "bottom")):
        add(f"back_rod_{level}", mujoco.mjtGeom.mjGEOM_CAPSULE, None, size=br,
            fromto=(-c, -spacing / 2, lz, -c, spacing / 2, lz))

    # Net (visual): back, two sides and top
    depth_net = c - r
    mid_x = -r - depth_net / 2
    thin = 0.0005
    add("net_back", mujoco.mjtGeom.mjGEOM_BOX, (-c, 0, height / 2),
        size=[thin, spacing / 2, height / 2], rgba=NET_RGBA)
    for ly, tag in ((-spacing / 2, "right"), (spacing / 2, "left")):
        add(f"net_{tag}", mujoco.mjtGeom.mjGEOM_BOX, (mid_x, ly, height / 2),
            size=[depth_net / 2, thin, height / 2], rgba=NET_RGBA)
    add("net_top", mujoco.mjtGeom.mjGEOM_BOX, (mid_x, 0, height),
        size=[depth_net / 2, spacing / 2, thin], rgba=NET_RGBA)

    # Collision volumes for the net (hidden): sides, back and top, as in the proto
    side_t, back_t = 2 * br, 0.3
    for ly, tag in ((-spacing / 2, "right"), (spacing / 2, "left")):
        add(f"col_{tag}", mujoco.mjtGeom.mjGEOM_BOX, (mid_x, ly, (height - br) / 2),
            size=[depth_net / 2, side_t / 2, (height - br) / 2], collide=True, hidden=True)
    add("col_back", mujoco.mjtGeom.mjGEOM_BOX, (-c - back_t / 2, 0, (height + br) / 2),
        size=[back_t / 2, (spacing + 2 * br) / 2, (height + br) / 2], collide=True, hidden=True)
    add("col_top", mujoco.mjtGeom.mjGEOM_BOX, (mid_x, 0, height),
        size=[depth_net / 2, (spacing + 2 * br) / 2, side_t / 2], collide=True, hidden=True)

