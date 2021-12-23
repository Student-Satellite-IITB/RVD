import numpy as np

m_INERTIA = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # moment of inertia array
v_area = np.array([0.0049, 0.0049, 0.0049])  # area of magnetorquer in sq.m.
n = 450  # number of turns of magnetorquer
PWM_AMPLITUDE = 3.3  # pwm amplitude in volt
PWM_FREQUENCY = 1e3  # pwm frequency in hertz
h = 0.001  # step size of integration in seconds
INDUCTANCE = 68e-3  # Inductance of torquer in Henry
RESISTANCE = 107.0  # Resistance of torquer	in Ohm
v_q0_m = np.array([1., 0., 0., 0.])  # unit quaternion initial condition
v_B_BIB_m = 1e-9 * np.identity(3)  # constant value of magnetic field assumed in body frame
d = 0.6  # duty cycle
CONTROL_TIME = 0.001
PWM_TIME = 1 / PWM_FREQUENCY
t_on = d * PWM_TIME
t_off = (1 - d) * PWM_TIME
time_step = 0.00001
w_initial = 0
