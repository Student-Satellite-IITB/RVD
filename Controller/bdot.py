
import numpy as np 
import matplotlib.pyplot as plt 

k = 10
dt = 0.001
Ixx = 0.00152529
Iyy = 0.00145111
Izz = 0.001476
Ixy = 0.00000437
Iyz = - 0.00000408
Ixz = 0.00000118
m_INERTIA = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def quaternion_product(q1,q2):
	return np.concatenate((q2[3]*q1[0:3] + q1[3]*q2[0:3] - np.cross(q1[0:3],q2[0:3]), [q1[3]*q2[3] - np.dot(q1[0:3],q2[0:3])]))


def quaternion_rotation_matrix(Q):
    q0 = Q[3]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[0]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


q=np.array([0.5,0.5,0.5,0.5])
w_current = np.array([0.5,0,0])
B_body = np.array([1e-5,1e-5,1e-5])
B_current = np.array([7.e-5,7.e-5,7.e-5])
torque=np.array([1.0,1.0,1.0])*0.1

w0=list()
w1=list()
w2=list()
torque1=list()
torque2=list()
torque0=list()
q0=list()
q1=list()
q2=list()
q3=list()

for i in range(10000):
	l1=0.5*quaternion_product(np.concatenate(([[0.0]],w_current),axis=None),q)
	l2=0.5*quaternion_product(np.concatenate(([[0.0]],w_current),axis=None),q+l1*dt/2)
	l3=0.5*quaternion_product(np.concatenate(([[0.0]],w_current),axis=None),q+l2*dt/2)
	l4=0.5*quaternion_product(np.concatenate(([[0.0]],w_current),axis=None),q+l3*dt)
	q=q+(l1+2*l2+2*l3+l4)*dt/6
	q=q/np.linalg.norm(q)


	B_next = np.matmul(quaternion_rotation_matrix(q),B_body)  #matrix cross product
	B_dot = (np.subtract(B_next,B_current))*(1/dt)  
	
	mag_moment = (-1)*k*B_dot
	torque = np.cross(mag_moment,B_current)
	B_current = B_next
	k1=np.matmul(np.linalg.inv(m_INERTIA),(torque - np.cross(w_current,np.matmul(m_INERTIA,w_current))))
	k2=np.matmul(np.linalg.inv(m_INERTIA),(torque - np.cross(w_current+k1*dt/2,np.matmul(m_INERTIA,w_current+k1*dt/2))))
	k3=np.matmul(np.linalg.inv(m_INERTIA),(torque - np.cross(w_current+k2*dt/2,np.matmul(m_INERTIA,w_current+k2*dt/2))))
	k4=np.matmul(np.linalg.inv(m_INERTIA),(torque - np.cross(w_current+k3*dt,np.matmul(m_INERTIA,w_current+k3*dt))))
	w_current=w_current+(k1+2*k2+2*k3+k4)*dt/6

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

	
	
y= np.linspace(0, (i+1)*dt, num=i+1)

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()

ax1 = f1.add_subplot()
ax1.plot(y,w0,label='x')
ax1.plot(y,w1,label='y')
ax1.plot(y,w2,label='z')
ax1.legend() 
ax1.set(title = "Plot of angular velocity with time",
          xlabel = "Time",
           ylabel = "angular velocity")
    
ax2 = f2.add_subplot()
ax2.plot(y,torque0,label='x')
ax2.plot(y,torque1,label='y')
ax2.plot(y,torque2,label='z')
ax2.legend() 
ax2.set(title = "Plot of torque with time",
          xlabel = "Time",
           ylabel = "torque")

ax3 = f3.add_subplot()
ax3.plot(y,q0,label='x')
ax3.plot(y,q1,label='y')
ax3.plot(y,q2,label='z')
ax3.plot(y,q3,label='k')
ax3.legend() 
ax3.set(title = "Plot of quaternion with time",
          xlabel = "Time",
           ylabel = "qt")

plt.show()

