"""
Plot what the analytical walking engine (mujoco_sim) generates, without
running a simulator: CoM, ZMP reference, footsteps and swing-foot height.

    python tools/plot_trajectories.py                    # walk forward
    python tools/plot_trajectories.py --vx 0 --vy 0.03   # sidestep
    python tools/plot_trajectories.py --save walk.png

The ZMP shown is recovered from the CoM through the LIPM equation
zmp = com - com_ddot / lambda^2, with com_ddot from finite differences.
Each step's LIPM solution matches the CoM *position* at the step boundaries but
not its velocity, so the recovered ZMP spikes at step transitions; the axes are
limited to the feet/CoM range so those spikes do not hide the rest.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mujoco_sim.analytical_walking.walking_engine import WalkingEngine  # noqa: E402
from mujoco_sim.config import WALK_PARAMS  # noqa: E402


def simulate(vx: float, vy: float, wz: float, duration: float, dt: float) -> dict:
    engine = WalkingEngine(**WALK_PARAMS)
    engine.set_command(vx=vx, vy=vy, wz=wz)
    rows = [engine.step(dt) for _ in range(int(duration / dt))]
    t = np.arange(len(rows)) * dt
    com = np.array([r.body_pos[:2] for r in rows])
    left = np.array([r.left_sole for r in rows])
    right = np.array([r.right_sole for r in rows])

    lam2 = engine.g / engine.z_com
    com_ddot = np.gradient(np.gradient(com, dt, axis=0), dt, axis=0)
    zmp = com - com_ddot / lam2
    return {"t": t, "com": com, "zmp": zmp, "left": left, "right": right}


def _limits(d: dict, axis: int) -> tuple[float, float]:
    """Axis range covering the feet and CoM (not the ZMP spikes), with margin."""
    values = np.concatenate([d["left"][:, axis], d["right"][:, axis], d["com"][:, axis]])
    lo, hi = values.min(), values.max()
    margin = 0.1 * (hi - lo) + 0.01
    return lo - margin, hi + margin


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot analytical walking trajectories")
    parser.add_argument("--vx", type=float, default=0.05, help="forward speed [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="lateral speed [m/s]")
    parser.add_argument("--wz", type=float, default=0.0, help="turn rate [rad/s]")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds to plot")
    parser.add_argument("--save", metavar="FILE", help="save the figure instead of showing it")
    args = parser.parse_args()

    d = simulate(args.vx, args.vy, args.wz, args.duration, dt=0.002)
    t = d["t"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(f"Analytical walking — vx={args.vx} m/s, vy={args.vy} m/s, wz={args.wz} rad/s")

    for ax, i, label in ((axes[0, 0], 0, "x"), (axes[1, 0], 1, "y")):
        ax.plot(t, d["left"][:, i], color="tab:blue", lw=1, label="left foot")
        ax.plot(t, d["right"][:, i], color="tab:red", lw=1, label="right foot")
        ax.plot(t, d["zmp"][:, i], color="tab:gray", ls="--", lw=1, label="ZMP (LIPM)")
        ax.plot(t, d["com"][:, i], color="black", lw=2, label="CoM")
        ax.set_ylabel(f"{label} [m]")
        ax.set_ylim(*_limits(d, i))
        ax.grid(alpha=0.3)
    axes[0, 0].set_title("Horizontal trajectories over time")
    axes[1, 0].set_xlabel("time [s]")
    axes[0, 0].legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, d["left"][:, 2] * 1000, color="tab:blue", label="left foot")
    ax.plot(t, d["right"][:, 2] * 1000, color="tab:red", label="right foot")
    ax.set_title("Swing-foot height")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [mm]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(d["com"][:, 0], d["com"][:, 1], color="black", lw=2, label="CoM")
    ax.plot(d["zmp"][:, 0], d["zmp"][:, 1], color="tab:gray", ls="--", lw=1, label="ZMP (LIPM)")
    for key, color in (("left", "tab:blue"), ("right", "tab:red")):
        on_ground = d[key][:, 2] < 1e-6
        ax.scatter(d[key][on_ground, 0], d[key][on_ground, 1], s=4, color=color, label=f"{key} foot (stance)")
    ax.set_title("Top view")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(*_limits(d, 0))
    ax.set_ylim(*_limits(d, 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
