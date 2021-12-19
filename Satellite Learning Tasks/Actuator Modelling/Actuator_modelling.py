import math
import matplotlib.pyplot as plt
import numpy as np

d = 0.8
V = 3.3
R = 42                                                                                   
N = 450
A = 0.0049
L = 0.068
T = 0.01
B = 1

ip=0.0
vp=0.0
I=[]
t2=[]
a_delta_w=[]
c_delta_w=[]



for n in range (1,10):
  #Charging
  t1=np.linspace((n-1)*T, (n-1+d)*T, num=100,endpoint=False)
  for t in t1:
    I.append((V-(ip*R))*(1- math.exp(-R*(t-((n-1)*T))/L))/R +ip)
    t2 = np.append(t2,t)
          
  ip = I[99*(2*n-1)]
  vp = (ip*R)
  #Discharging
  t1=np.linspace((n-1+d)*T, (n)*T, num=100,endpoint=False)
  for t in t1:
    I.append((vp/R)*(math.exp(-R*(t-((n-1+d)*T))/L)))
    t2 = np.append(t2,t)

  ip = I[99*2*n]
  vp = (ip*R)

# print("I",I)
plt.plot(t2,I)
plt.show()

#Analytical delta_w
for n in range (1,10):
  io = I[99*2*(n-1)]
  a_delta_w.append(N*A*B*((V*d*T/R)-(V*L*(math.exp(-R*T*(1-d)/L)-math.exp(-R*T/L))/(R*R)) + io*L*(1-math.exp(-R*T/L)/R)))
print("Analytical delta_w",a_delta_w)

#Calculated delta_w
for n in range (1,9):
  integration=0.0
  for i in range(200*(n-1),200*n):
    integration += I[i]*B*(t2[i+1]-t2[i])
  c_delta_w.append(N*A*integration)
print("Calculated delta_w",c_delta_w)
