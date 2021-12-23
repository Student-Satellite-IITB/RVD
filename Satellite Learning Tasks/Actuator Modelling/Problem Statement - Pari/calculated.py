import math
from const import *
from analytical import w_analytical
from current_calculation import CURRENT

w_calculated = np.zeros(1)
w_calculated[0] = w_initial
w_difference = np.zeros(1)
w_difference[0] = 0
temp = 0
for i in range(1, int(CONTROL_TIME / time_step)):
    temp = temp + (CURRENT[i] - CURRENT[i - 1]) * time_step * n * v_area[0] * v_B_BIB_m[0,0]
    w_calculated = np.append(w_calculated, temp + w_calculated[i-1])
    w_difference = np.append(w_difference, math.fabs(w_calculated[i]-w_analytical[i])*100/w_analytical[i])

print(w_difference)
