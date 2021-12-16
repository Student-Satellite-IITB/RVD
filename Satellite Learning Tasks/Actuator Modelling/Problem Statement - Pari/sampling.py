from const import *
import math

t = 5  # time at which we need to find current
d = 0.6  # duty cycle
PWM_TIME = 1 / PWM_FREQUENCY
i_p = np.array
v_p = np.array
v_p[0] = 0
i_p[0] = 0
t_on = d * PWM_TIME
t_off = (1 - d) * PWM_TIME
time_step = 0.001
for i in range(n - 1):
    v_p = np.append(i_p[i] + (PWM_AMPLITUDE - i_p[i] * RESISTANCE) * (1 - math.exp(-RESISTANCE * t_on / INDUCTANCE)))
    i_p = np.append((v_p / RESISTANCE) * math.exp(-RESISTANCE * t_off / INDUCTANCE))

cycle_no = int(t / PWM_TIME) + 1
if t < cycle_no * PWM_TIME + t_on:
    I = i_p[cycle_no - 1] + (PWM_AMPLITUDE - i_p[cycle_no - 1] * RESISTANCE) * (
                1 - math.exp(-RESISTANCE * (t - cycle_no * PWM_TIME) / INDUCTANCE)) / RESISTANCE
else:
    I = (v_p[cycle_no - 1] / RESISTANCE) * math.exp(-RESISTANCE * (t - cycle_no * PWM_TIME - t_on) / INDUCTANCE)
