#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[5]:


import pandas as pd


# In[6]:


import sympy as sym


# In[7]:


import matplotlib as pyplot


# In[8]:


from matplotlib import pyplot 


# In[9]:


def objective(x):
    return(x+3)**2


# In[10]:


def derivative(x):
    return 2*(x+3)


# In[12]:


def gradient(alpha,start,max_iter):
    x_list=list()
    x=start
    x_list.append(x)
    for i in range(max_iter):
        gradi=derivative(x)
        x=x-(alpha*gradi)
        x_list.append(x)
    return x_list
x=sym.symbols('x')
expr=(x+3)**2.0
grad=sym.Derivative(expr,x)
print("{}".format(grad.doit()))
grad.doit().subs(x,2)


# In[18]:


alpha=0.1
start=2
max_iter=30
x=sym.symbols('x')
expr=(x+3)**2
x_cor=np.linspace(-15,15,100)
pyplot.plot(x_cor,objective(x_cor))
pyplot.plot(2,objective(2),'ro')


# In[22]:


x=gradient(alpha,start,max_iter)
x_cor=np.linspace(-5,5,100)
pyplot.plot(x_cor,objective(x_cor))

x_arr=np.array(x)
pyplot.plot(x_arr,objective(x_arr),'.-',color='red')
pyplot.show()


# In[27]:


# Solve for the critical points by setting the derivative equal to zero
critical_points = sym.solve(derivative, x)

# Filter out real solutions to find local minima
local_minima = [point for point in critical_points if point.is_real]

print("Critical Points:", critical_points)
print("Local Minima:", local_minima)


# In[ ]:




