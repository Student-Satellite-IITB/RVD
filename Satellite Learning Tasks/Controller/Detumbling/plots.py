import numpy as np
import matplotlib.pyplot as plt
import constants
import detumbling

y = np.linspace(0, constants.n * constants.dt, num=constants.n)

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()

ax1 = f1.add_subplot()
ax1.plot(y, detumbling.w0, label='x')
ax1.plot(y, detumbling.w1, label='y')
ax1.plot(y, detumbling.w2, label='z')
ax1.legend()
ax1.set(title="Plot of angular velocity with time",
        xlabel="Time",
        ylabel="angular velocity")

ax2 = f2.add_subplot()
ax2.plot(y, detumbling.torque0, label='x')
ax2.plot(y, detumbling.torque1, label='y')
ax2.plot(y, detumbling.torque2, label='z')
ax2.legend()
ax2.set(title="Plot of torque with time",
        xlabel="Time",
        ylabel="torque")

ax3 = f3.add_subplot()
ax3.plot(y, detumbling.q0, label='x')
ax3.plot(y, detumbling.q1, label='y')
ax3.plot(y, detumbling.q2, label='z')
ax3.plot(y, detumbling.q3, label='k')
ax3.legend()
ax3.set(title="Plot of quaternion with time",
        xlabel="Time",
        ylabel="qt")

ax4 = f4.add_subplot()
ax4.plot(y, detumbling.qB0, label='x')
ax4.plot(y, detumbling.qB1, label='y')
ax4.plot(y, detumbling.qB2, label='z')
ax4.legend()
ax4.set(title="Plot of Bdot with time",
        xlabel="Time",
        ylabel="qt")
plt.show()