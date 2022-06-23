from quaternion import *

n = 200
k = 10e3
dt = 0.001
Ixx = 0.00152529
Iyy = 0.00145111
Izz = 0.001476
Ixy = 0.00000437
Iyz = 0.00000408
Ixz = 0.00000418
m_INERTIA = np.array([
    [Ixx, Ixy, Ixz],
    [Ixy, Iyy, Iyz],
    [Ixz, Iyz, Izz]
])
q_initial = get_quaternion_from_euler(0.5, 0.5, 0.5)
w_current_initial = np.array([0.5, 0.5, 0.5])
B_inertial = np.array([1e-5, 1e-5, 1e-5])
torque_initial = np.array([0, 0, 0])
