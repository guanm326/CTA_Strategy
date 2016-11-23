# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:20:21 2016

@author: zhaoyong
"""
import time
import random
import numpy as np
import pandas as pd
import scipy.linalg
from numba import jit
from scipy.optimize import fmin_tnc
import seaborn as sns
sns.set_style('whitegrid')

#a, b, tc, m, c, w, phi

def lppl(t,x):
    tc = x[0]
    m = x[1]
    w = x[2]
    phi = x[3]
    f[t] = np.power(tc - t, m)
    g[t] = f[t]*np.cos((w *np.log(tc-t)))+phi

    sigma_f = np.sum(f)
    sigma_g = np.sum(g)
    sigma_fg = np.sum(f*g)
    sigma_f2 = np.sum(f**2)
    sigma_g2 = np.sum(g**2)
    n = len(DataSeries[0])

    matrixA = np.array([ [n, sigma_f, sigma_g],
                         [sigma_f, sigma_f2, sigma_fg],
                         [sigma_g, sigma_fg, sigma_g2] ])
    y[t] = DataSeries[1][int(t)]
    sigma_y = np.sum(y)  # yi的取值
    sigma_yf = np.sum(y*f)
    sigma_yg = np.sum(y*g)

    matrixB = np.array([ [sigma_y], [sigma_yf], [sigma_yg]])

    #lu, piv = scipy.linalg.lu_factor(matrixA)
    #Y = scipy.linalg.lu_solve((lu,piv), matrixB)
    Y = np.dot(np.matrix(matrixA).I,matrixB)

    a = Y[0]
    b = Y[1]
    c = Y[2]
    obj = a + b*f[t] + c*g[t]

    #print 'obj:',np.array(obj)[0][0]
    return np.array(obj)[0][0]

def func(x):
    n = len(DataSeries[0])
    delta = np.zeros(n)
    global f,g,y
    f = np.zeros(n)
    g = np.zeros(n)
    y = np.zeros(n)
    for t in DataSeries[0]:
        delta[t] = lppl(t,x)
    delta = np.subtract(delta, DataSeries[1])
    delta = np.power(delta, 2)
    print np.average(delta)
    return np.average(delta)



class Individual:
    'base class for individuals'

    def __init__ (self, InitValues):
        self.fit = 0
        self.cof = InitValues

    def fitness(self): #
        try:
            cofs, nfeval, rc = fmin_tnc(func, self.cof, fprime=None,approx_grad=True, messages=0) #基于牛顿梯度下山的寻找函数最小值
            self.fit = func(cofs)
            self.cof = cofs
        except:
            #does not converge
            return False
    def mate(self, partner): #交配
        reply = []
        for i in range(0, len(self.cof)): # 遍历所以的输入参数
            if (random.randint(0,1) == 1): # 交配，0.5的概率自身的参数保留，0.5的概率留下partner的参数，即基因交换
                reply.append(self.cof[i])
            else:
                reply.append(partner.cof[i])

        return Individual(reply)
    def mutate(self): #突变
        for i in range(0, len(self.cof)-1):
            if (random.randint(0,len(self.cof)) <= 2):
                #print "Mutate" + str(i)
                self.cof[i] += random.choice([-1,1]) * .05 * i #突变

    def PrintIndividual(self): #打印结果
        #t, a, b, tc, m, c, w, phi

        cofs = "Critical Time: " + str(round(self.cof[0], 3))
        cofs += "m: " + str(round(self.cof[1], 3))

        cofs += "omega: " + str(round(self.cof[2], 3))
        cofs += "phi: " + str(round(self.cof[3], 3))

        return "fitness: " + str(self.fit) +"\n" + cofs
        #return str(self.cof) + " fitness: " + str(self.fit)
    def getDataSeries(self):
        return DataSeries
    def getExpData(self):
        return [np.exp(lppl(t,self.cof)) for t in DataSeries[0]]
    def getActSerise(self):
        return date_close


def fitFunc(t, a, b, tc, m, c, w, phi):
    return a + (b*np.power(tc - t, m))*(1 + (c*np.cos((w *np.log(tc-t))+phi)))

#SP = pd.io.data.get_data_yahoo('^GSPC', start=datetime.datetime(2012, 5, 1),end=datetime.datetime(2015, 5, 23))
HS = pd.read_csv('HS300.csv', index_col=0)
HS['datetime'] = HS.index.to_datetime()

date = np.linspace(0, len(HS)-1, len(HS))
close = [np.log(HS.CLOSE[i]) for i in range(len(HS.CLOSE))]
#close = SP.Close.values
global DataSeries
DataSeries = [date[:-160], close[:-160]]
global date_close
date_close = [HS['datetime'].values, HS.CLOSE.values]
