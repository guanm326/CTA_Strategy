# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:39:49 2016

@author: zhaoyong
"""


import datetime
import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
import seaborn as sns
sns.set_style('whitegrid')


#a, b, tc, m, c, w, phi
def fnon0(x):
    tc = x[0]
    m = x[1]
    w = x[2]
    phi = x[3]
    if m != 0 or w != 0:
        f = [np.power(tc - t, m) for t in DataSeries[0]]
        g = [np.power(tc - t, m)*np.cos((w *np.log(tc-t)))+phi for t in DataSeries[0]]
    return f,g

def fnon(x):
    #a = x[0]
    #b = x[1]
    #c = x[4]
    f = np.array(fnon0(x)[0])
    g = np.array(fnon0(x)[1])
    n = len(DataSeries[0])
    sigma_f = np.sum(f)
    sigma_g = np.sum(g)
    sigma_fg = np.sum(f*g)
    sigma_f2 = np.sum(f**2)
    sigma_g2 = np.sum(g**2)

    matrixA = np.array([ [n, sigma_f, sigma_g],
                         [sigma_f, sigma_f2, sigma_fg],
                         [sigma_g, sigma_fg, sigma_g2] ])
    
    sigma_y = np.sum(DataSeries[1][:t+1])  # yi的取值
    sigma_yf = np.sum(DataSeries[1]*f)
    sigma_yg = np.sum(DataSeries[1]*g)

    matrixB = np.array([ [sigma_y], [sigma_yf], [sigma_yg]])

    lu, piv = scipy.linalg.lu_factor(matrixA)
    Y = scipy.linalg.lu_solve((lu,piv), matrixB)

    a = Y[0]
    b = Y[1]
    c = Y[2]
    return a,b,c

def lppl(t,x):
    tc = x[0]
    m = x[1]
    w = x[2]
    phi = x[3]
    a,b,c = fnon(x)

    obj = a + (b*np.power(tc - t, m))*(1 + (c*np.cos((w *np.log(tc-t))+phi)))
    return obj

def func(x):
    delta = [lppl(t,x) for t in DataSeries[0]]
    delta = np.subtract(delta, DataSeries[1])
    delta = np.power(delta, 2)
    return np.average(delta)


HS = pd.read_csv('HS300.csv', index_col=0)
HS['datetime'] = HS.index.to_datetime()

time = np.linspace(0, len(HS)-1, len(HS))
close = [np.log(HS.CLOSE[i]) for i in range(len(HS.CLOSE))]
#close = SP.Close.values
global DataSeries
DataSeries = [time[:-160], close[:-160]]
global date_close
date_close = [HS['datetime'].values, HS.CLOSE.values]

#print str(func([1,1,600,1.5,.5,1,.5]))
cofs, nfeval, rc = fmin_tnc(func, [500, 0.1, 12, 2*np.pi], fprime=None,approx_grad=True)
print cofs
print nfeval
print "value of func: " + str(func(cofs))
values = [np.exp(lppl(t,cofs)) for t in DataSeries[0]]
plt.figure(figsize=(16,8))


#真实线
plt.plot(date_close[0], date_close[1])
#预测线
plt.plot(date_close()[0][:-160], values)

plt.show()
