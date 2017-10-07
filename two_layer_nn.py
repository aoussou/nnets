import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

''' 
A 2-layer network, to model the XOR and XNOR operator.

This model is actually not good, so you should just do this as an exercise.
'''

# create the input vector X
X = np.zeros( (4,2) )
X[0,0] = 0; X[0,1] = 0
X[1,0] = 0; X[1,1] = 1
X[2,0] = 1; X[2,1] = 0
X[3,0] = 1; X[3,1] = 1
X = np.concatenate(  (np.ones((X.shape[0],1) ),X) ,axis=1)
print('X is \n',X,'\n')

# create the true solution vector Z
Z = np.zeros( (4,1) )
Z[0,0] = 0
Z[1,0] = 1
Z[2,0] = 1
Z[3,0] = 0
print('Z is \n',Z,'\n') 

# chose random values for the weights
scale = 1       

U = np.random.rand(3,2)*2*scale - scale
v = np.random.rand(3,1)*2*scale - scale


print('U initial is \n',U,'\n')
print('v initial is \n',v,'\n')
 
# gradient descent step
alpha = .2;

nIt = 100000;
    
# create an error matrix to plot the gradient descent results
E = np.zeros((nIt,1))

for i in range(0,nIt):
    
    Y = np.dot(X,U)
    A0 = sp.special.expit(Y)
    A = np.concatenate(  (np.ones((A0.shape[0],1) ),A0) ,axis=1)
    #print('Y is \n',Y,'\n')
  
    h = sp.special.expit(np.dot(A,v))

    #print('Zp is \n',Zp,'\n')
    d = Z - h
    
    D = np.diag(d[:,0])
    
    U = U + alpha*np.dot(np.dot(X.T,np.dot(D,np.multiply(A0,1-A0))), np.diag(v[1:,0]))

    v = v + alpha*np.dot(A.T,d)

    E[i] = np.dot(d.T,d)
    
print('U final initial is \n',U,'\n')
print('v final initial is \n',v,'\n')

xplot = list(range(1,nIt+1))
plt.semilogx(np.log(xplot),E, "-o", color="r")
plt.tight_layout()
plt.xlabel('# iterations')
plt.ylabel('error')
print('the final result with a .5 threshold is \n',np.round(h))