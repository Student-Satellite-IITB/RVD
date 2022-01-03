import math
from const import *
from current_calculation import CURRENT
w_analytical = np.zeros(1)
w_analytical[0] = w_initial
mag_field = v_B_BIB_m[0, 0]
area = v_area[0]
for i in range(1, int(CONTROL_TIME / time_step)):
    int_i = int_i + PWM_AMPLITUDE * d * time_step / RESISTANCE - PWM_AMPLITUDE * INDUCTANCE * (
            math.exp(-RESISTANCE * time_step * (1 - d) / INDUCTANCE) - math.exp(
        -RESISTANCE * time_step / INDUCTANCE)) / math.pow(
        RESISTANCE, 2) + CURRENT[i] * INDUCTANCE * (1 - math.exp(-RESISTANCE * time_step / INDUCTANCE)) / RESISTANCE
    temp = n * area * int_i * mag_field
    w_analytical = np.append(w_analytical, w_analytical[i-1] + temp)
