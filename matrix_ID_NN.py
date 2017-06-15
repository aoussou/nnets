import numpy as np
import random

''' 
A Neural Network for the IDENTITY and NOT operator.

The output is the same as the input

input x | output y

  0     |   0
  1     |   1
'''

# initialize and populate X, the input vector
x0 = np.zeros( (2,1) )
x0[0,0] = 0
x0[1,0] = 1

# initialize y
# for the IDENTITY operator, it is  the same a x0
yId = x0 

# for the NOT operator
yNot = np.zeros( (2,1) )
yNot[0,0] = 1
yNot[1,0] = 0

# initialize w and populate it with random values  
w = np.zeros( (2,1) )
w[0,0] = random.uniform(-1,1)*10
w[1,0] = random.uniform(-1,1)*10
 
# choose a gradient descent step
alpha = .01;

# chose the IDENTITY (Y = Yid) or NOT operator (Y = Ynot)
y = yId

# 
X = np.concatenate(  (np.ones((x0.shape[0],1) ),x0) ,axis=1)

print('X is \n',X,'\n')
print('y is \n',y,'\n')
print('w is \n',w,'\n')

# GRADIENT DESCENT    
for i in range(0,10000):
    
    w = w + alpha*np.dot(X.T, y - np.dot(X,w))
    

print('the solution is: \n',w)    
    
beta = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,y))
print('Using LS normal equations we obtain beta = \n',beta)
