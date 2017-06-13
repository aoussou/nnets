#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:45:52 2017

@author: john
"""

''' 
A Neural Network for the IDENTITY and NOT operator.

The output is the same as the input

input x | output y

  0     |   0
  1     |   1
'''

import random

# initialize weights with random values
a = random.uniform(-1, 1)*10
b = random.uniform(-1, 1)*10
 
# choose a gradient descent step
alpha = .01;

# choose a number of iterations
nIt = 100000

# GRADIENT DESCENT    
for i in range(0,nIt):
    
    aOld = a    
    bOld = b
    

    a = aOld - alpha*(aOld + bOld - 1)
    b = bOld - alpha*(aOld + 2*bOld - 1)    
    
    # the NOT operator 
    '''
    a = aOld - alpha*(aOld + bOld)    
    b = bOld - alpha*(aOld + 2*bOld - 1) 
    '''
    
    
print('a = ',a)
print('b = ',b)

