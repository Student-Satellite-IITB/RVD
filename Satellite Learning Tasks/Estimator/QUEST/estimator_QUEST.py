import numpy as np
from numpy.core.fromnumeric import trace

vector_B_st = np.array([[]])
vector_I_m = np.array([[]])
I = np.identity(3)

#take inputs here as numpy array from the testing file and append them to vector_B_st and vector_I_m, these will be arrays of array
# remember to use vector_B_st = np.append(vector_B_st, [ *np array from input file* ], axis =0)

vector_B_st = list(filter(None, vector_B_st)) #remove any empty elements 
vector_I_m = list(filter(None, vector_I_m)) #remove any empty elements

#taking all the weights to be 1

lambda_0 = vector_B_st.shape[0]

B = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
for i in vector_B_st.shape[0]:
    B = B + np.outer(vector_B_st[i],vector_I_m[i])

S = B + np.transpose(B) 
kai = np.trace(np.linalg.det(B)*np.linalg.inv(B))
z = np.array([B[1][2]-B[2][1] , B[2][0] - B[0][2] ,B[0][1] - B[1][0]])

def func(lambda0):  #can't use lambda as that is apparently a function
    x = ((lambda0**2 - np.trace(B) + kai)(lambda0**2 - np.trace(B) - np.inner(z,z)) - (lambda0 - np.trace(B))(np.matmul(np.matmul(z,B),np.transpose(z)) + np.linalg.det(S)) - np.matmul(np.matmul(z,np.matmul(B,B)),np.transpose(z)))

def dfunc(lambda1):
    return (2*lambda1(lambda1**2 - np.trace(B) - np.inner(z,z) + lambda1**2 - np.trace(B) + kai) - (lambda1)(np.matmul(np.matmul(z,B),np.transpose(z))))

def newtonRaphson( x ):
    h = func(x) / dfunc(x)
    while abs(h) >= 0.0001:  #our limits
        h = func(x)/dfunc(x)
         
        # x(i+1) = x(i) - f(x) / f'(x)
        x = x - h
     
    return x
 
# Driver program to test above
 # Initial values assumed
lambda_max = newtonRaphson(lambda_0)

rho = lambda_max + np.trace(B)
temp = np.matmul(np.linalg.det(rho*I - S)*np.linalg.inv(rho*I - S), np.transpose(z))

quaternion_estimate = np.array([[temp[0]],[temp[1]],[temp[2]],[np.linalg.det(temp)]])
quaternion_estimate = (1/np.inner(np.transpose(quaternion_estimate),np.transpose(quaternion_estimate)))*quaternion_estimate  #normalisation