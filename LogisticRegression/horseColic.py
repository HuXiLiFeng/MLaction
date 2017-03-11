#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import random
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(file):
    dataMat=[]
    labelMat=[]
    with open(file) as f:
        for line in f:
            line = line.strip().split('\t')
            dataMat.append([float(j) for j in line[0:-1]])
            labelMat.append(float(line[-1]))
    print len(dataMat),len(dataMat[0]),len(labelMat)
    print dataMat[0]
    return dataMat,labelMat

def sigmoid(x):
    return 1/(1+np.exp(-x))

def betterAscent(dataMat,labelMat,alpha=0.01,maxIter=500):
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

def evaluate(testFile,weight):
    testData,testLabel=loadDataSet(testFile)
    data = np.array(testData)
    weight=np.array(weight)
    temp=np.dot(data,weight)
    predict=[]
    right=0
    err=0
    for i in xrange(temp.shape[0]):
        if sigmoid(temp[i][0])>0.5:
            predict.append(1)
        else:
            predict.append(0)
    for j in xrange(len(predict)):
        if predict[j]==testLabel[j]:
            right+=1
        else:
            err+=1
    print right,err
    print right/len(testData),err/len(testData)



dataMat,labelMat=loadDataSet('horseColicTraining.txt')
weight=betterAscent(dataMat,labelMat)
evaluate('horseColicTest.txt',weight)
