import numpy as np
import random
import matplotlib.pyplot as plt

''' 
A 2-layer network, to model the XOR and XNOR operator.

This model is actually not good, so you should just this as an exercise.

A better model uses a sigmoid activation function.
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
 
print('Z is \n',Z,'\n') 

# chose random values for the weights
scale = .5          

U = np.zeros( (3,2) )
U[0,0] = random.uniform(-1,1)*scale; U[0,1] = random.uniform(-1,1)*scale
U[1,0] = random.uniform(-1,1)*scale; U[1,1] = random.uniform(-1,1)*scale
U[2,0] = random.uniform(-1,1)*scale; U[2,1] = random.uniform(-1,1)*scale

v = np.zeros( (3,1) )
v[0,0] = random.uniform(-1,1)*scale
v[1,0] = random.uniform(-1,1)*scale
v[2,0] = random.uniform(-1,1)*scale 

print('U initial is \n',U,'\n')
print('v initial is \n',v,'\n')
 
# gradient descent step
alpha = .02;

nIt = 10000;
    
# create an error matrix to plot the gradient decent results
E = np.zeros((nIt,1))

for i in range(0,nIt):
    
    
    Y0 = np.dot(X,U)    
    Y = np.concatenate(  (np.ones((Y0.shape[0],1) ),Y0) ,axis=1) 
    Zp = np.dot(Y,v)
    d = Z - Zp
    U = U + alpha*np.dot(np.dot(X.T,d),v[1:].T)
    v = v + alpha*np.dot(Y.T,d)

    E[i] = np.dot(d.T,d)
    
print('U final initial is \n',U,'\n')
print('v final initial is \n',v,'\n')

print('the final result is Zpred =',Zp)
xplot = list(range(1,nIt+1))
plt.semilogx(np.log(xplot),np.log(E), "-o", color="r")
plt.tight_layout()
plt.xlabel('# iterations')
plt.ylabel('error')
