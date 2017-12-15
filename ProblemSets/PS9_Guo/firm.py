
# coding: utf-8

# In[1]:

import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import pickle as pk
import pandas as pd





# In[5]:


def firm_sol(alpha,delta, psi,r,z,pi):
    
    betafirm = (1 / (1 + r))
    dens = 5
# put in bounds here for the capital stock space
    kstar = ((1/betafirm-1+delta)/alpha)**(1/(alpha-1))
    kbar = 2*kstar
    lb_k = 0.001
    ub_k = kbar
    krat = np.log(lb_k / ub_k)
    numb = np.ceil(krat / np.log(1 - delta))
    K = np.zeros(int(numb * dens))
# we'll create in a way where we pin down the upper bound - since
# the distance will be small near the lower bound, we'll miss that by little
    for j in range(int(numb * dens)):
        K[j] = ub_k * (1 - delta) ** (j / dens)
    kgrid = K[::-1]
    sizek = kgrid.shape[0]
    op=np.ones([sizek,len(z)])
# operating profits, op

    for i in range(len(z)):
    
        op[:,i] = z[i]*kgrid**alpha 

# firm cash flow, e
    e = np.zeros((len(z),sizek, sizek))
    for k in range(len(z)):
        for i in range(sizek):
            for j in range(sizek):
        
                e[k, i,j] = (op[i,k] - kgrid[j] + ((1 - delta) * kgrid[i]) -
                       ((psi / 2) * ((kgrid[j] - ((1 - delta) * kgrid[i])) ** 2)/ kgrid[i]))
   
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = np.zeros((len(z), sizek))  # initial guess at value function
    Vmat = np.zeros((len(z),sizek, sizek))  # initialize Vmat matrix
    Vstore = np.zeros((len(z),sizek,  VFmaxiter))  # initialize Vstore array
    VFiter = 1
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V
        expected_V=np.dot(pi,V)
        for i in range(sizek):  # loop over k
            Vmat[:,i,: ] = e[:,i, :] + betafirm * expected_V

        Vstore[:,:, VFiter] = V.reshape(len(z),sizek,)  # store value function at each
    # iteration for graphing later
        V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
        PF = np.argmax(Vmat, axis=2)  # find the index of the optimal k'
        Vstore[:, :,i] = V  # store V at each iteration of VFI
        VFdist = (np.absolute(V - TV)).max()  # check distance between value
    # function for this iteration and value function from past iteration
        VFiter += 1
    Firm_Solution={'PF': PF, 'pi':pi,'z':z,'kvec':kgrid}
    filehandler = open("firm_solution.obj","wb")
    pk.dump(Firm_Solution,filehandler)
    filehandler.close()
    return Firm_Solution,e, V


# In[ ]:



