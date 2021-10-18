import numpy as np
from sympy import *
import math 


Ib11, Ib12, Ib13 = symbols('Ib11 Ib12 Ib13')
Ib21, Ib22, Ib23 = symbols('Ib21 Ib22 Ib23')
Ib31, Ib32, Ib33 = symbols('Ib31 Ib32 Ib33')
Ib = Matrix([[Ib11, Ib12, Ib13],[Ib21, Ib22, Ib23],[Ib31, Ib32, Ib33]])  #moment of inertia of body

Tb1 , Tb2, Tb3 = symbols('Tb1 Tb2 Tb3')
Tb = Matrix([[Tb1],[Tb2],[Tb3]])                   #control input torque

wbo1, wbo2, wbo3 = symbols('wbo1 wbo2 wbo3')
wbo = Matrix([[wbo1],[wbo2],[wbo3]]) #angular velocity of body wrt orbital

Abo11, Abo12, Abo13 = symbols('Abo11 Abo12 Abo13')
Abo21, Abo22, Abo23 = symbols('Abo21 Abo22 Abo23')
Abo31, Abo32, Abo33 = symbols('Abo31 Abo32 Abo33')
Abo = Matrix([[Abo11, Abo12, Abo13],[Abo21, Abo22, Abo23],[Abo31, Abo32, Abo33]])  #attitude matrix of body wrt orbital frame

woI1, woI2, woI3 = symbols('woI1 woI2 woI3')
woI = Matrix([[woI1],[woI2],[woI3]]) #angular velocity of inertial wrt orbital

dwoI1, dwoI2, dwoI3 = symbols('dwoI1 dwoI2 dwoI3')
dwoI = Matrix([[dwoI1], [dwoI2], [dwoI3]]) #angular acceleration i.e differentiation of angular vecloity of orbital wrt angular frame

print('Angular acceleration term / angular velocity derivative')
print(Ib.inv()*(Tb - (wbo + Abo*woI).cross(Ib*(Abo*dwoI + wbo))))  

alpha1, alpha2, alpha3 = symbols('alpha1 alpha2 alpha3')  #euler angles describing attitude from body to orbital frame
alpha = Matrix([[alpha1],[alpha2],[alpha3]])

B = Matrix([[sec(alpha2)*cos(alpha3), sec(alpha2)*sin(alpha3), 0],[sin(alpha3), cos(alpha3), 0],[-tan(alpha2)*cos(alpha3), tan(alpha2)*sin(alpha3), 1]])  #pre definied matrix B, looks at theory

print('\n \n \neuler angles derivative:')
print(B*wbo)