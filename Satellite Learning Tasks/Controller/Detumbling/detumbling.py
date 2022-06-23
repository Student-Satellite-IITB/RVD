from quaternion import *
import constants


q = constants.q_initial
w_current = constants.w_current_initial
B_current = np.dot(quaternion_rotation_matrix(q), constants.B_inertial)
torque = constants.torque_initial

w0 = list()
w1 = list()
w2 = list()
torque1 = list()
torque2 = list()
torque0 = list()
q0 = list()
q1 = list()
q2 = list()
q3 = list()
qB0 = list()
qB1 = list()
qB2 = list()


def euler_equation(w):
    return np.matmul(np.linalg.inv(constants.m_INERTIA),
                     (torque - np.cross(w, np.matmul(constants.m_INERTIA, w))))


for i in range(constants.n):
    kn1 = euler_equation(w_current)
    ln1 = quaternion_propagation(q, w_current)
    kn2 = euler_equation(w_current + 0.5 * constants.dt * kn1)
    ln2 = quaternion_propagation(q + 0.5 * constants.dt * ln1, w_current + 0.5 * constants.dt * kn1)
    kn3 = euler_equation(w_current + 0.5 * constants.dt * kn2)
    ln3 = quaternion_propagation(q + 0.5 * constants.dt * ln2, w_current + 0.5 * constants.dt * kn2)
    kn4 = euler_equation(w_current + constants.dt * kn3)
    ln4 = quaternion_propagation(q + constants.dt * ln3, w_current + constants.dt * kn3)
    w_current = w_current + (kn1 + 2 * kn2 + 2 * kn3 + kn4) * constants.dt / 6
    q = q + (ln1 + 2 * ln2 + 2 * ln3 + ln4) * constants.dt / 6
    q = q / np.linalg.norm(q)
    B_next = np.matmul(quaternion_rotation_matrix(q), constants.B_inertial)  # matrix cross product
    B_dot = (B_next - B_current) / constants.dt
    mag_moment = (-1) * constants.k * B_dot
    torque = np.cross(mag_moment, B_current)

    print(B_current)
    B_current = B_next

    w0.append(w_current[0])
    w1.append(w_current[1])
    w2.append(w_current[2])
    torque0.append(torque[0])
    torque1.append(torque[1])
    torque2.append(torque[2])
    q0.append(q[0])
    q1.append(q[1])
    q2.append(q[2])
    q3.append(q[3])
    qB0.append(B_dot[0])
    qB1.append(B_dot[1])
    qB2.append(B_dot[2])



