#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 调用sklearn库实现PCA降维

import numpy as np
from sklearn import datasets,decomposition,manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadData():
    iris=datasets.load_iris()
    return iris.data,iris.target

def testPCA(*data):
    X,y=data
    pca=decomposition.PCA(n_components=2, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
    newData=pca.fit_transform(X)
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for label, color in zip(np.unique(y), colors):
        position = y == label
        ax.scatter(newData[position, 0], newData[position, 1], label="class%d" % label, color=color)
    ax.set_title('PCA')
    ax.set_xlabel("the first components")
    ax.set_ylabel("the second components")
    ax.legend(loc='best')
    plt.show()

def incrementalPCA(*data):
    X, y = data
    fig = plt.figure()
    # IncrementalPCA用于超大规模数据，将数据分批加载进内存
    ax1 = fig.add_subplot(1, 1, 1)
    incPCA = decomposition.IncrementalPCA(n_components=2,batch_size=10)
    incPCA.fit(X)
    newData1=incPCA.transform(X)
    color = ['blue', 'black', 'red']
    for i in range(newData1.shape[0]):
        ax1.scatter(newData1[i, 0], newData1[i, 1], color=color[y[i]])
    ax1.set_title('3 class after dimensionality reduction using IncrementalPCA')
    ax1.legend(loc='best')
    plt.show()

def kernelPCA(*data):
    # 核化PCA
    X, y = data
    fig = plt.figure()
    kernels=['linear','poly','rbf','sigmoid']
    for i,kernel in enumerate(kernels):
        # 此处仅考虑kernel和n_components参数，还有degree,gamma,coef0等参数可自行设置
        kernelPCA = decomposition.KernelPCA(kernel=kernel,n_components=2)
        kernelPCA.fit(X)
        newData2 = kernelPCA.transform(X)
        colors=((1,0,0),(0,1,0),(0,0,1))
        ax2 = fig.add_subplot(2, 2, i+1)
        for label,color in zip(np.unique(y),colors):
            position=y==label
            ax2.scatter(newData2[position,0],newData2[position,1],label="class%d"%label,color=color)
        ax2.set_title('kernelPCA:%s'%kernel)
        ax2.set_xlabel("the first components")
        ax2.set_ylabel("the second components")
        ax2.legend(loc='best')
    plt.suptitle("kernel PCA")
    plt.show()



if __name__ == '__main__':
    X,y=loadData()
    testPCA(X,y)
    incrementalPCA(X,y)
    kernelPCA(X,y)
