"""RoboCup soccer field: dimensions from the Webots proto, compiled in MuJoCo (no robot needed)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mujoco_sim.field import FIELD_DIMENSIONS, LINE_WIDTH, add_field, field_dimensions  # noqa: E402


def _build(size):
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
    add_field(spec, size)
    return spec.compile()


def _geom(model, name):
    return model.geom(name)


@pytest.mark.parametrize("size", sorted(FIELD_DIMENSIONS))
def test_field_compiles_and_covers_field_plus_border(size):
    model = _build(size)
    dim = field_dimensions(size)
    np.testing.assert_allclose(
        _geom(model, "floor").size[:2],
        [dim["length"] / 2 + dim["border"], dim["width"] / 2 + dim["border"]])


@pytest.mark.parametrize("size", sorted(FIELD_DIMENSIONS))
def test_goal_posts_on_the_goal_lines(size):
    model = _build(size)
    dim = field_dimensions(size)
    spacing = dim["goal_width"] + 2 * 0.05
    for team, sign in (("home", -1), ("away", +1)):
        # "left"/"right" are in the goal's own frame, so the away goal is mirrored in y
        ys = sorted(_geom(model, f"goal_{team}_{tag}_post").pos[1] for tag in ("left", "right"))
        np.testing.assert_allclose(ys, [-spacing / 2, spacing / 2], atol=1e-9)
        for tag in ("left", "right"):
            assert _geom(model, f"goal_{team}_{tag}_post").pos[0] == pytest.approx(
                sign * dim["length"] / 2)


def test_goals_open_towards_the_field():
    """The net lies behind the goal line: outside the field for both goals."""
    model = _build("kid")
    assert _geom(model, "goal_home_net_back").pos[0] < -4.5
    assert _geom(model, "goal_away_net_back").pos[0] > 4.5


def test_goal_height_matches_kid_size():
    model = _build("kid")
    crossbar = _geom(model, "goal_home_crossbar")
    assert crossbar.pos[2] == pytest.approx(1.2 + 0.05)   # opening + crossbar radius


def test_markings_do_not_collide_but_goal_frame_does():
    model = _build("kid")
    assert _geom(model, "boundary_south").contype == 0
    assert _geom(model, "center_circle_0").contype == 0
    assert _geom(model, "goal_home_left_post").contype == 1
    assert _geom(model, "goal_home_col_back").contype == 1


def test_penalty_mark_distance():
    model = _build("kid")
    # 1.5 m from the goal line; the cross is a pair of boxes centred on that point
    assert _geom(model, "home_penalty_mark_h").pos[0] == pytest.approx(-4.5 + 1.5)
    assert _geom(model, "away_penalty_mark_h").pos[0] == pytest.approx(4.5 - 1.5)


def test_centre_circle_line_spans_the_official_radii():
    model = _build("kid")
    radii = [np.linalg.norm(_geom(model, f"center_circle_{k}").pos[:2]) for k in range(36)]
    assert np.mean(radii) == pytest.approx(0.75 + LINE_WIDTH / 2, abs=5e-3)   # polygon: chord midpoints sit slightly inside the circle


def test_unknown_size_is_rejected():
    with pytest.raises(ValueError):
        field_dimensions("giant")
