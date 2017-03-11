#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ch05: Logistic Regression with gradient ascent
in Machine Learning in Action
"""
from __future__ import division
import random
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(file):
    dataMat=[]
    labelMat=[]
    with open(file) as f:
        for line in f:
            (x,y,label) = line.strip().split('\t')
            dataMat.append([1.0,float(x),float(y)])
            labelMat.append(int(label))
    return dataMat,labelMat

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def gradientAscent(dataMat,labelMat,alpha=0.01,maxIter=500):
    """ gradient ascent"""
    data=np.mat(dataMat)
    label=np.mat(labelMat).transpose()
    m,n=data.shape
    weight=np.ones((n,1))
    for k in xrange(maxIter):
        h=sigmoid(data*weight)
        error=label-h
        weight+=alpha*data.transpose()*error
    return weight

def stocGradientAscent(dataMat,labelMat,alpha=0.01,maxIter=200):
    """Stochastic gradient ascent algorithm """
    data=np.mat(dataMat)
    label=np.mat(labelMat).transpose()
    m,n=data.shape
    weight=np.ones((n,1))
    for k in xrange(maxIter):
        for j in xrange(m):
            h=sigmoid(data[j]*weight)
            error=label[j]-h
            weight+=alpha*data[j].transpose()*error #挑选（伪随机）第i个实例来更新权值向量
    return weight

def betterAscent(dataMat,labelMat,alpha=0.01,maxIter=200):
    """ change the alpha dynamically"""
    data=np.mat(dataMat)
    label=np.mat(labelMat).transpose()
    m,n=data.shape
    weight=np.ones((n,1))
    for k in xrange(maxIter):
        dataIndex=range(m)
        for j in xrange(m):
            alpha = 4 / (1 + j + k) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))  # 总算是随机了
            h=sigmoid(data[j]*weight)
            error=label[j]-h
            weight+=alpha*data[j].transpose()*error
            del dataIndex[randIndex]
    return weight

def plotBestFigure(weight,file):
    weights=weight.transpose()
    dataMat,labelMat=loadDataSet(file)
    dataArr=np.array(dataMat)
    xmin = min(dataArr[:, 1])
    xmax = max(dataArr[:, 1])
    n=dataArr.shape[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in xrange(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i][1])
            ycord1.append(dataArr[i][2])
        else:
            xcord2.append(dataArr[i][1])
            ycord2.append(dataArr[i][2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s',label='class 1')
    ax.scatter(xcord2,ycord2,s=30,c='green',label='class 0')
    x=np.arange(xmin,xmax,1)
    y=(-weights[0][0]-weights[0][1]*x)/weights[0][2]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc = 'upper right')
    plt.show()

if __name__ == "__main__":
    file='testSet.txt'
    data,label=loadDataSet(file)
    weight=gradientAscent(data,label)
    print weight
    plotBestFigure(weight,file)