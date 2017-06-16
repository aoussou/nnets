#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:52:17 2017

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt

''' 
1)Get the results of a Least Square Linear Regression to model a binary 
operator using the normal equations.

2) Check whether the operator can actually be modelled with a linear neural
network
'''


X1 = np.zeros( (4,1) )
X1[0,0] = 0
X1[1,0] = 0
X1[2,0] = 1
X1[3,0] = 1

X2 = np.zeros( (4,1) )
X2[0,0] = 0
X2[1,0] = 1
X2[2,0] = 0
X2[3,0] = 1

y = np.zeros( (4,1) )
y[0,0] = 0
y[1,0] = 1
y[2,0] = 1
y[3,0] = 1

X = np.concatenate( (X1,X2,), axis=1)
X = np.concatenate(  (np.ones((X.shape[0],1) ),X) ,axis=1)

wLS = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,y))
print('Using LS normal equations we obtain wLS = \n',wLS)


fig = plt.figure()
for i in range(0,len(y)):
    
    if y[i]:
        plt.plot(X1[i],X2[i],marker='o',color="r")     
    else:
        plt.plot(X1[i],X2[i],marker='o',color="b")
        
simArtist = plt.Line2D((0,1),(0,0), color='b', marker='o', linestyle='')
anyArtist = plt.Line2D((0,1),(0,0), color='r',marker='o', linestyle='')
plt.legend([simArtist,anyArtist],['y = 0', 'y = 1'])

plt.gca().set_aspect('equal', adjustable='box')         
plt.tight_layout()
plt.xlabel('x1')
plt.ylabel('x2')




        
