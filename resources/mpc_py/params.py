import numpy as np

# MPC horizon
N = 50     # steps
DT = 0.04  # [sec]

# ---------- ODE model parameters ----------

# Motor constants [s]
MOTOR_LAMBDA = np.tile(40.0*np.array([0.62, 0.12, 0.1, 0.1, 0.1]), 4) # @10Hz control period

# Inertia tensor @ct frame
J_TENSOR = np.array([
    [ 0.386,   -0.0006, -0.0006],
    [-0.0006,    0.386,  0.0006],
    [-0.0006,   0.0006,  0.5318]
], dtype=np.float64)

# Mass
MASS    = 5.09495 # [kg]
G_ACCEL = 9.80665 # [m/s^2]

# link1,2,3,4,5 mass, CoM pos
LINK_MASS = np.array([0.36, 0.12, 0.04, 0.104, 0.36])               # [kg]
LINK_COM_DIST = np.array([-0.6975, -0.0575, -0.055, -0.012, -0.03]) # [m]

# i,0 -> i,5
DH_PARAMS_ARM = np.array([ # [a, alpha]
    [0.1395,np.pi/2.0],  # 0 -> 1
    [0.115,       0.0],  # 1 -> 2
    [0.110,       0.0],  # 2 -> 3
    [0.024, np.pi/2.0],  # 3 -> 4
    [0.068,       0.0],  # 4 -> 5
], dtype=np.float64)

# B -> i,0
DH_PARAMS_BASE = np.array([ # [a, theta]
    [0.120, 0.25*np.pi],  # arm1
    [0.120, 0.75*np.pi],  # arm2
    [0.120,-0.75*np.pi],  # arm3
    [0.120,-0.25*np.pi],  # arm4
], dtype=np.float64)

# ---------- Constraints & Costs ----------

# Constraint
ARM_MIN = np.array([-1.9, -1.9, -1.9, -1.9, -1.5])  # [rad]
ARM_MAX = np.array([1.9, 1.9, 1.9, 1.9, 1.5])       # [rad]
F_MIN = 20.  # [N]
F_MAX = 100. # [N]

# Cost
COST_POS_ERR   = np.array([10., 10., 10.]) # pos err xyz
COST_ANG_ERR   = np.array([0.1])           # attitude err yaw
COST_F_THRUST  = np.array([0.1])           # overall thrust