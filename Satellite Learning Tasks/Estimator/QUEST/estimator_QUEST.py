import numpy as np

vector_B_st = np.array([[]])
vector_I_m = np.array([[]])

#take inputs here as numpy array from the testing file and append them to vector_B_st and vector_I_m, these will be arrays of array
# remember to use vector_B_st = np.append(vector_B_st, [ *np array from input file* ], axis =0)

vector_B_st = list(filter(None, vector_B_st)) #remove any empty elements 
vector_I_m = list(filter(None, vector_I_m)) #remove any empty elements

#taking all the weights to be 1

lambda_0 = vector_B_st.shape[0]

B = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
for i in vector_B_st.shape[0]:
    B = B + np.outer(vector_B_st[i],vector_I_m[i])


