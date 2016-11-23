# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:05:39 2016

@author: zhaoyong
"""

import time
import random
import numpy as np
import pandas as pd
import scipy.linalg
from numba import jit
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import seaborn as sns
sns.set_style('whitegrid')

#a, b, tc, m, c, w, phi

def lppl(t,x):
    """方程式,tc,m,w为参数"""
    n = len(DataSeries[0])

    tc = x[0]
    m = x[1]
    w = x[2]
    # 中间变量发，f,g,h
    f[t] = np.power(tc - t, m)
    g[t] = f[t]*np.cos((w * np.log(tc-t)))
    h[t] = f[t]*np.sin((w * np.log(tc-t)))
    # 矩阵当中所需变量
    sigma_f = np.sum(f)
    sigma_g = np.sum(g)
    sigma_h = np.sum(h)
    sigma_fg = np.sum(f*g)
    sigma_fh = np.sum(f*h)
    sigma_gh = np.sum(g*h)
    sigma_f2 = np.sum(f**2)
    sigma_g2 = np.sum(g**2)
    sigma_h2 = np.sum(h**2)

    matrixA = np.array([ [n, sigma_f, sigma_g,sigma_h],
                         [sigma_f, sigma_f2, sigma_fg,sigma_fh],
                         [sigma_g, sigma_fg, sigma_g2,sigma_gh],
                         [sigma_h, sigma_fh, sigma_gh, sigma_h2] ])
    y[t] = DataSeries[1][t]
    sigma_y = np.sum(y)  # yi的取值
    sigma_yf = np.sum(y*f)
    sigma_yg = np.sum(y*g)
    sigma_yh = np.sum(y*h)


    matrixB = np.array([ [sigma_y], [sigma_yf], [sigma_yg], [sigma_yh]])

    #lu, piv = scipy.linalg.lu_factor(matrixA)
    #Y = scipy.linalg.lu_solve((lu,piv), matrixB)
    Y = np.dot(np.matrix(matrixA).I,matrixB)

    a = Y[0]
    b = Y[1]
    c = Y[2]
    d = Y[3]
    obj = a + b*f[t] + c*g[t] +d*h[t]

    #print 'obj:',np.array(obj)[0][0]
    return np.array(obj)[0][0]


def func(x):
    n = len(DataSeries[0])
    delta = np.zeros(n)
    global f,g,y,h
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    y = np.zeros(n)
    print "x0:",x
    if 1 > x[1] > 0 and x[0] > n and x[2] > 0:
        for t in DataSeries[0]:
            delta[t] = lppl(t,x)
        delta = np.subtract(delta, DataSeries[1])
        delta = np.power(delta, 2)
        print "x:",x
        print "delta:", np.average(delta)
        return np.average(delta)

#SP = pd.io.data.get_data_yahoo('^GSPC', start=datetime.datetime(2012, 5, 1),end=datetime.datetime(2015, 5, 23))
HS = pd.read_csv('HS300.csv', index_col=0)
HS['datetime'] = HS.index.to_datetime()

date = np.linspace(0, len(HS)-1, len(HS),dtype='int')
close = [np.log(HS.CLOSE[i]) for i in range(len(HS.CLOSE))]
#close = SP.Close.values
global DataSeries
DataSeries = [date[:-160], close[:-160]]
global date_close
date_close = [HS['datetime'].values, HS.CLOSE.values]
# 基于牛顿梯度算法的最小化

"""TODO:可以尝试其他优化算法来求解，对比最后的结果"""
result = minimize(func, [360, 0.1, 12])
cofs = result.x
print "cofs:",cofs
#print "nfeval:",nfeval
#print "value of func: " + str(func(cofs))
values = [np.exp(lppl(t,cofs)) for t in DataSeries[0]]
plt.figure(figsize=(16,8))


#真实线
plt.plot(date_close[0], date_close[1])
#预测线
plt.plot(date_close[0][:-160], values)

plt.show()