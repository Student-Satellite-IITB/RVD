# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 14:19:13 2022

@author: Aryan Gupta
"""
import numpy as np
from matplotlib import pyplot as plt

# z-axis in body frame of earth = axis of rotation
# x-axis in body frame of earth = lies in the plane formed by axis of rotation and dipole axis

# angular velocity of magnetotorquer in satellite body frame
w_x_BIB = float(input("Input the value of angular velocity: ")) #Body frame of the satellite
w_y_BIB = float(input("Input the value of angular velocity: "))
w_z_BIB = float(input("Input the value of angular velocity: "))
w_BIB = np.array([w_x_BIB, w_y_BIB, w_z_BIB])
w_BIB_mag = np.sqrt(w_BIB.dot(w_BIB))

# angular velocity of satellite in geocentric inertial frame
w_x_GCI = float(input("Input the value of angular velocity: ")) #Body frame of the satellite
w_y_GCI = float(input("Input the value of angular velocity: "))
w_z_GCI = float(input("Input the value of angular velocity: "))
w_GCI = np.array([w_x_BIB, w_y_BIB, w_z_BIB])
w_GCI_mag = np.sqrt(w_BIB.dot(w_BIB))

t = 0
t_total = float(input("Input the value of total time period: "))
t_step = float(input("Input the value of time step: "))

axis_torquer_x = float(input(""))
axis_torquer_y = float(input(""))
axis_torquer_z = float(input(""))

axis_torquer = np.array([axis_torquer_x, axis_torquer_y, axis_torquer_z, 0])
axis_torquer_mag = np.sqrt(axis_torquer.dot(axis_torquer))
axis_torquer = axis_torquer/axis_torquer_mag

# Code for frame transformation using quaternions

while t <= t_total:
    Q_1 = axis_torquer[1]*np.sin(w_BIB_mag*t*0.5)
    Q_2 = axis_torquer[2]*np.sin(w_BIB_mag*t*0.5)
    Q_3 = axis_torquer[3]*np.sin(w_BIB_mag*t*0.5)
    Q_4 = np.cos(w_BIB_mag*t*0.5)

    Q = np.array([Q_1, Q_2, Q_3, Q_4])
    Q_dot = np.array([Q[4], 0, 0, Q[1]],
                     [0, Q[4], 0, Q[2]],
                     [0, 0, Q[4], Q[3]],
                     [-1*Q[1], -1*Q[2], -1*Q[3], Q[4]])
    + np.array([0, -1*Q[3], Q[2], 0], [Q[3], 0, -1*Q[1], 0], [-1*Q[2], Q[1], 0, 0], [0, 0, 0, 0])
    Q_cross = np.array([Q[4], 0, 0, Q[1]], [0, Q[4], 0, Q[2]], [0, 0, Q[4], Q[3]], [-1*Q[1], -1*Q[2], -1*Q[3], Q[4]]) - np.array([0, -1*Q[3], Q[2], 0], [Q[3], 0, -1*Q[1], 0], [-1*Q[2], Q[1], 0, 0], [0, 0, 0, 0])
    
    axis_torquer_Q = np.matmul(np.matmul(np.transpose(Q_dot), Q_cross), axis_torquer)
    
    t += t_step
    