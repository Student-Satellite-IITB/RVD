import sampling
from const import *
CURRENT = np.zeros(int(CONTROL_TIME / time_step) + 1)
time = 0
for i in range(int(CONTROL_TIME / time_step)):
    CURRENT[i] = sampling.sample(time)
    time += time_step
