#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 利用流行学习降维

import numpy as np
from sklearn import datasets,manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def testMDS():
    # 测试多维缩放降维(multiple dimensional scaling, MDS)
    X, color = datasets.samples_generator.make_s_curve(n_samples=1000, random_state=11)
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)
    plt.show()


    mds = manifold.MDS(n_components=2, max_iter=100, n_init=1)
    Y = mds.fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title("MDS")
    ax.set_xlabel("first component")
    ax.set_ylabel("second component")
    plt.show()
    print mds.stress_

def loadData():
    iris = datasets.load_iris()
    return iris.data,iris.target

def testIsomap(*data):
    # 等度量映射降维，Isomap
    X,y=data
    isomap=manifold.Isomap(n_neighbors=7,n_components=2,eigen_solver='arpack',tol=0.00001,max_iter=100,
                           path_method='auto')
    newData=isomap.fit_transform(X)
    label_color=((0,'blue'),(1,'yellow'),(2,'red'))
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for label,color in label_color:
        position=y==label
        ax.scatter(X[position,0],X[position,1],color=color,label="class%s"%label)
    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')
    ax.set_title('isomap')
    ax.legend(loc='best')
    plt.show()

def testLLE(*data):
    # 等度量映射降维，Isomap
    X,y=data
    LLE=manifold.LocallyLinearEmbedding(n_neighbors=7,n_components=2,eigen_solver='arpack',tol=0.00001,max_iter=100,
                                        method='standard',neighbors_algorithm='kd_tree')
    newData=LLE.fit_transform(X)
    label_color=((0,'blue'),(1,'yellow'),(2,'red'))
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for label,color in label_color:
        position=y==label
        ax.scatter(newData[position,0],newData[position,1],color=color,label="class%s"%label)
    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')
    ax.set_title('LLE')
    ax.legend(loc='best')
    plt.show()



if __name__=="__main__":
    #testMDS()
    X,y=loadData()
    testIsomap(X,y)

