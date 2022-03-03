# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from matplotlib import pyplot as plt
I_XX_m = float(input("Enter the value of I_xx:\n"))
I_YY_m = float(input("Enter the value of I_yy:\n"))
I_ZZ_m = float(input("Enter the value of I_zz:\n"))
I_XY_m = float(input("Enter the value of I_xy:\n"))
I_YZ_m = float(input("Enter the value of I_yz:\n"))
I_ZX_m = float(input("Enter the value of I_zx:\n"))

m_I_m = np.array([[I_XX_m, -1*I_XY_m, -1*I_ZX_m], [-1*I_XY_m, I_YY_m, -1*I_YZ_m], [-1*I_ZX_m, -1*I_YZ_m, I_ZZ_m]], order = 'F')

w_x_BIB_m = float(input("Enter the value of w_x:\n"))
w_y_BIB_m = float(input("Enter the value of w_y:\n"))
w_z_BIB_m = float(input("Enter the value of w_z:\n"))

c_w_BIB_m = np.array([w_x_BIB_m, w_y_BIB_m, w_z_BIB_m], order = 'F')
TIME_STEP = float(input("Enter the value of the time step:\n"))
TIME = float(input("Enter the value of the total time:\n"))

angle_st = 0
angle4_st = 0
K = float(input("Enter the value of the coefficient of w in torque:\n"))

i_time = 0

plt.xlabel("Time")
plt.ylabel("Angular Velocity")

axis1_BIB_st = plt.subplot(6, 1, 1)
axis2_BIB_st = plt.subplot(6, 1, 2)
axis3_BIB_st = plt.subplot(6, 1, 3)
axis4_BIB_st = plt.subplot(6, 1, 4)
axis5_BIB_st = plt.subplot(6, 1, 5)
axis6_BIB_st = plt.subplot(6, 1, 6)

while(i_time < TIME):
    c_k1_st = np.matmul(np.linalg.inv(m_I_m), (K*c_w_BIB_m - np.cross(c_w_BIB_m, np.matmul(m_I_m, c_w_BIB_m))))
    c_w1_BIB_m = np.add(c_w_BIB_m, TIME_STEP*c_k1_st/2)
    c_k2_st = np.matmul(np.linalg.inv(m_I_m), (K*c_w1_BIB_m - np.cross(c_w1_BIB_m, np.matmul(m_I_m, c_w1_BIB_m))))
    c_w2_BIB_m = np.add(c_w_BIB_m, TIME_STEP*c_k2_st/2)
    c_k3_st = np.matmul(np.linalg.inv(m_I_m), (K*c_w2_BIB_m - np.cross(c_w2_BIB_m, np.matmul(m_I_m, c_w2_BIB_m))))
    c_w3_BIB_m = np.add(c_w_BIB_m, TIME_STEP*c_k3_st)
    c_k4_st = np.matmul(np.linalg.inv(m_I_m), (K*c_w3_BIB_m - np.cross(c_w3_BIB_m, np.matmul(m_I_m, c_w3_BIB_m))))
    c_w4_BIB_m = np.add(c_w_BIB_m, np.add(TIME_STEP*c_k1_st/6, np.add(TIME_STEP*c_k2_st/3, np.add(TIME_STEP*c_k3_st/3, TIME_STEP*c_k4_st/6))))
    
    c_axis_BIB_m = np.array([c_w_BIB_m[0]/np.linalg.norm(c_w_BIB_m), c_w_BIB_m[1]/np.linalg.norm(c_w_BIB_m), c_w_BIB_m[2]/np.linalg.norm(c_w_BIB_m)], order = 'F') 
    c_axis4_BIB_m = np.array([c_w4_BIB_m[0]/np.linalg.norm(c_w4_BIB_m), c_w4_BIB_m[1]/np.linalg.norm(c_w4_BIB_m), c_w4_BIB_m[2]/np.linalg.norm(c_w4_BIB_m)], order = 'F') 
    angle_st += np.linalg.norm(c_w_BIB_m)*TIME_STEP
    angle4_st += np.linalg.norm(c_w4_BIB_m)*TIME_STEP
    
    m_w_cross_BIB_m = np.array([[0, -1*c_w_BIB_m[2], c_w_BIB_m[1]], [c_w_BIB_m[2], 0, -1*c_w_BIB_m[0]], [-1*c_w_BIB_m[1], c_w_BIB_m[0], 0]], order = 'F')
    m_w4_cross_BIB_m = np.array([[0, -1*c_w4_BIB_m[2], c_w4_BIB_m[1]], [c_w4_BIB_m[2], 0, -1*c_w4_BIB_m[0]], [-1*c_w4_BIB_m[1], c_w4_BIB_m[0], 0]], order = 'F')
    
    m_rotation_BIB_m = np.identity(3) + np.sin(angle_st)*m_w_cross_BIB_m + (1-np.cos(angle_st))*m_w_cross_BIB_m*m_w_cross_BIB_m
    m_rotation4_BIB_m = np.identity(3) + np.sin(angle4_st)*m_w4_cross_BIB_m + (1-np.cos(angle4_st))*m_w4_cross_BIB_m*m_w4_cross_BIB_m
    
    axis1_BIB_st.plot([i_time, i_time + TIME_STEP], [c_w_BIB_m[0], c_w4_BIB_m[0]], 'b')
    axis1_BIB_st.plot([i_time, i_time + TIME_STEP], [c_w_BIB_m[1], c_w4_BIB_m[1]], 'g')
    axis1_BIB_st.plot([i_time, i_time + TIME_STEP], [c_w_BIB_m[2], c_w4_BIB_m[2]], 'r')  
    c_L_BIB_m = np.matmul(m_I_m, c_w_BIB_m)
    c_L4_BIB_m = np.matmul(m_I_m, c_w4_BIB_m)
    
    axis2_BIB_st.plot([i_time, i_time + TIME_STEP], [c_L_BIB_m[0], c_L4_BIB_m[0]], 'b')
    axis2_BIB_st.plot([i_time, i_time + TIME_STEP], [c_L_BIB_m[1], c_L4_BIB_m[1]], 'g')
    axis2_BIB_st.plot([i_time, i_time + TIME_STEP], [c_L_BIB_m[2], c_L4_BIB_m[2]], 'r')
    
    axis3_BIB_st.plot([i_time, i_time + TIME_STEP], [c_w_BIB_m[0]*c_w_BIB_m[0] + c_w_BIB_m[1]*c_w_BIB_m[1] + c_w_BIB_m[2]*c_w_BIB_m[2], c_w4_BIB_m[0]*c_w4_BIB_m[0] + c_w4_BIB_m[1]*c_w4_BIB_m[1] + c_w4_BIB_m[2]*c_w4_BIB_m[2]], 'k')
    axis3_BIB_st.plot([i_time, i_time + TIME_STEP], [c_L_BIB_m[0]*c_L_BIB_m[0] + c_L_BIB_m[1]*c_L_BIB_m[1] + c_L_BIB_m[2]*c_L_BIB_m[2], c_L4_BIB_m[0]*c_L4_BIB_m[0] + c_L4_BIB_m[1]*c_L4_BIB_m[1] + c_L4_BIB_m[2]*c_L4_BIB_m[2]], 'm')
    
    axis4_BIB_st.plot([i_time, i_time + TIME_STEP], [angle_st, angle4_st], 'k' )
    
    axis5_BIB_st.plot([i_time, i_time + TIME_STEP], [c_axis_BIB_m[0], c_axis4_BIB_m[0]], 'b')
    axis5_BIB_st.plot([i_time, i_time + TIME_STEP], [c_axis_BIB_m[1], c_axis4_BIB_m[1]], 'g')
    axis5_BIB_st.plot([i_time, i_time + TIME_STEP], [c_axis_BIB_m[2], c_axis4_BIB_m[2]], 'r')
    
    axis6_BIB_st.plot([i_time, i_time + TIME_STEP], [m_rotation_BIB_m[0, 0], m_rotation4_BIB_m[0, 0]], 'b')
    axis6_BIB_st.plot([i_time, i_time + TIME_STEP], [m_rotation_BIB_m[0, 1], m_rotation4_BIB_m[0, 1]], 'g')
    axis6_BIB_st.plot([i_time, i_time + TIME_STEP], [m_rotation_BIB_m[0, 2], m_rotation4_BIB_m[0, 2]], 'r')
    axis6_BIB_st.plot([i_time, i_time + TIME_STEP], [m_rotation_BIB_m[1, 0], m_rotation4_BIB_m[1, 0]], 'b')
    axis6_BIB_st.plot([i_time, i_time + TIME_STEP], [m_rotation_BIB_m[1, 1], m_rotation4_BIB_m[1, 1]], 'g')
    axis6_BIB_st.plot([i_time, i_time + TIME_STEP], [m_rotation_BIB_m[1, 2], m_rotation4_BIB_m[1, 2]], 'r')
    axis6_BIB_st.plot([i_time, i_time + TIME_STEP], [m_rotation_BIB_m[2, 0], m_rotation4_BIB_m[2, 0]], 'b')
    axis6_BIB_st.plot([i_time, i_time + TIME_STEP], [m_rotation_BIB_m[2, 1], m_rotation4_BIB_m[2, 1]], 'g')
    axis6_BIB_st.plot([i_time, i_time + TIME_STEP], [m_rotation_BIB_m[2, 2], m_rotation4_BIB_m[2, 2]], 'r')
    
    i_time += TIME_STEP
    c_w_BIB_m = c_w4_BIB_m
plt.show()