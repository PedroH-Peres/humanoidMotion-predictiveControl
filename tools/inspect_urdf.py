"""
Print every joint of the Aurea URDF with its offset from the parent link.

Useful for measuring link lengths when writing analytical IK.

Caveat: PyBullet reports parentFramePos relative to the parent's *inertial*
(centre of mass) frame, not the URDF link frame. For exact link lengths, read
the <joint><origin xyz=...> values in the URDF itself.

    python tools/inspect_urdf.py [path/to/robot.urdf]
"""

import math
import os
import sys

import pybullet as p

DEFAULT_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "pybullet_sim", "assets", "aurea", "aurea.urdf")


def main() -> None:
    urdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URDF

    p.connect(p.DIRECT)
    robot_id = p.loadURDF(urdf, useFixedBase=True)

    print("\n" + "=" * 100)
    print(f"Joints of {os.path.basename(urdf)} (offset from the parent's inertial frame)")
    print("=" * 100)

    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[1].decode("utf-8")
        x, y, z = info[14]  # parentFramePos
        length = math.sqrt(x**2 + y**2 + z**2)
        print(f"ID: {j:02d} | Joint: {name:<14} | Offset (x, y, z): "
              f"{x:+.4f}, {y:+.4f}, {z:+.4f} | Length: {length:.4f} m")

    print("=" * 100 + "\n")
    p.disconnect()


if __name__ == "__main__":
    main()
