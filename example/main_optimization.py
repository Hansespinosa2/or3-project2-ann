#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:05:26 2022

@author: alphago
"""
import numpy as np
import time
import matplotlib.pyplot as plt

def f(x):
    return 1/2*x.T.dot(x)

def g(x):
    return x

initial_0 = np.array([1,2,3,4,5,6,7])
K = 100

def probablistic_search(f,g,initial_0,K):
    step_size = 0.1
    x = initial_0
    x_seq = []
    obj_seq = []
    
    for i in range(K):
        x_new = x-step_size*(np.random.normal(g(x),0.1))
        x = x_new
        obj = f(x)
        
        x_seq.append(x)
        obj_seq.append(obj)
    
    return x,obj,x_seq,obj_seq

start = time.time()
x,obj,x_seq,obj_seq = probablistic_search(f,g,initial_0,K)
end = time.time()

plt.plot(obj_seq)
plt.xlabel('iterations')
plt.ylabel('objective values')
plt.title('Sequence of objective values')

print('Solution time: ',end-start)
print('Output objective function value: ',obj)
print('Output solution:',x)

