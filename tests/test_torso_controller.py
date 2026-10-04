"""Torso orientation controller (pure math, no simulator needed)."""

import pytest

from mujoco_sim.control.torso_controller import TorsoOrientationController


def _imu(pitch=0.0, roll=0.0, pitch_rate=0.0, roll_rate=0.0):
    return {"pitch": pitch, "roll": roll, "pitch_rate": pitch_rate, "roll_rate": roll_rate}


def test_no_tilt_gives_no_correction():
    ctrl = TorsoOrientationController()
    assert all(v == 0.0 for v in ctrl.compute(_imu()).values())


def test_pitch_correction_is_mirrored_between_legs():
    ctrl = TorsoOrientationController(kp_pitch=1.0, kd_pitch=0.0)
    c = ctrl.compute(_imu(pitch=0.1))
    assert c["l_hip_pitch"] == pytest.approx(0.1)
    assert c["r_hip_pitch"] == pytest.approx(-0.1)  # right hip_pitch axis is flipped


def test_corrections_are_clamped():
    ctrl = TorsoOrientationController(max_delta=0.2)
    c = ctrl.compute(_imu(pitch=1.0, roll=-1.0, pitch_rate=5.0, roll_rate=-5.0))
    assert all(abs(v) <= 0.2 + 1e-12 for v in c.values())
