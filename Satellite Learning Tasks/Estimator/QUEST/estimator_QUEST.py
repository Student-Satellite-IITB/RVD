import numpy as np
from numpy.core.fromnumeric import trace

vector_B_st = np.array([[-0.82124793099839, 	-0.13064801039884, 	0.555412399222135],[0.927474543469896, 	0.347581781638122,	0.137760213010057]])
vector_I_m = np.array([[-0.82225186031413, 	-0.192660630484306, 	0.535521950690485],[0.636821431291897, 	0.762830374553013, 	-0.112019124736129]])
I = np.identity(3)

#take inputs here as numpy array from the testing file and append them to vector_B_st and vector_I_m, these will be arrays of array
# remember to use vector_B_st = np.append(vector_B_st, [ *np array from input file* ], axis =0)

#vector_B_st = list(filter(None, vector_B_st)) #remove any empty elements 
#vector_I_m = list(filter(None, vector_I_m)) #remove any empty elements

#taking all the weights to be 1


#print(z)

def B_func():
    B = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
    for i in range(vector_B_st.shape[0]):
        B = B + np.outer(vector_B_st[i],vector_I_m[i])
    return B

def S_func():
    B = B_func()
    S = B + np.transpose(B)
    return S 

def z_func():
    B = B_func()
    z = np.array([B[1][2]-B[2][1] , B[2][0] - B[0][2] ,B[0][1] - B[1][0]])
    return z


def func(lambda0, kai):  #can't use lambda as that is apparently a function
    B = B_func()
    S = S_func()
    z = z_func()
    return ((lambda0**2 - np.trace(B)**2 + kai)*(lambda0**2 - np.trace(B)**2 - np.inner(z,z)) - (lambda0 - np.trace(B))*(np.matmul(np.matmul(z,S),np.transpose(z)) + np.linalg.det(S)) - np.matmul(np.matmul(z,np.matmul(S,S)),np.transpose(z)))

def dfunc(lambda1, kai):
    B = B_func()
    S = S_func()
    z = z_func()
    return (2*lambda1*(lambda1**2 - np.trace(B)**2 - np.inner(z,z) + lambda1**2 - np.trace(B)**2 + kai) - (np.matmul(np.matmul(z,S),np.transpose(z)) + np.linalg.det(S)))

def newtonRaphson(x, kai):
    h = func(x, kai) / dfunc(x, kai)
    while abs(h) >= 0.0001:  #our limits
        h = func(x)/dfunc(x)
        # x(i+1) = x(i) - f(x) / f'(x)
        x = x - h
    return x
 
# Driver program to test above
 # Initial values assumed

def main():
    lambda_0 = vector_B_st.shape[0]

    B = B_func()
    S = S_func()

    kai = np.trace(np.linalg.det(S)*np.linalg.inv(S))

    z = z_func()

    lambda_max = newtonRaphson(lambda_0, kai)
    rho = lambda_max + np.trace(B)

    temp = np.matmul(np.linalg.det(rho*I - S)*np.linalg.inv(rho*I - S), np.transpose(z))

    quaternion_estimate = np.array([[temp[0]],[temp[1]],[temp[2]],[np.linalg.det(rho*I - S)]])
    quaternion_estimate = (1/np.inner(np.transpose(quaternion_estimate),np.transpose(quaternion_estimate))**0.5)*quaternion_estimate  #normalisation
    print(quaternion_estimate) #convention is vector first then scalar

main()



