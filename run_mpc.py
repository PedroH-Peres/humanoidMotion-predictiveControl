"""Run the MuJoCo simulation with the MPC walking engine:  python run_mpc.py"""

import mujoco_sim.sim as sim
from mujoco_sim.mpc_walking.mpc_walking_engine import MPCWalkingEngine

sim.WalkingEngine = MPCWalkingEngine
sim.main()
