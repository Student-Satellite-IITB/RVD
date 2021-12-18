from const import *
import math

i_p = np.zeros(1)
v_p = np.zeros(1)

for i in range(int(CONTROL_TIME / PWM_TIME)):
    temp = i_p[i] * RESISTANCE + (PWM_AMPLITUDE - i_p[i] * RESISTANCE) * (1 - math.exp(-RESISTANCE * t_on / INDUCTANCE))
    v_p = np.append(v_p, temp)
    temp = (v_p[i] / RESISTANCE) * math.exp(-RESISTANCE * t_off / INDUCTANCE)
    i_p = np.append(i_p, temp)


def sample(t):
    cycle_no = int(t / PWM_TIME) + 1
    if t < (cycle_no - 1) * PWM_TIME + t_on:
        I = (PWM_AMPLITUDE - i_p[cycle_no - 1] * RESISTANCE) * (
                1 - math.exp(-RESISTANCE * (t - (cycle_no - 1) * PWM_TIME) / INDUCTANCE)) / RESISTANCE
    else:
        I = (v_p[cycle_no - 1] / RESISTANCE) * math.exp(
            -RESISTANCE * (t - (cycle_no - 1) * PWM_TIME - t_on) / INDUCTANCE)
    return I


CURRENT = np.zeros(int(CONTROL_TIME / PWM_TIME)+1)
time = 0
for i in range(int(CONTROL_TIME / PWM_TIME)):
    CURRENT[i] = sample(time)
    time += time_step
    print(CURRENT[i])
