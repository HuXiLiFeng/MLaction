#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, cross_validation, linear_model, svm


def load_data():
    # 这里用来分类，用的是鸢尾花数据集
    iris = datasets.load_iris()
    return cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=0)


def test_LinearSVC(X_train, X_test, y_train, y_test):
    clf = svm.LinearSVC(penalty='l2',loss='squared_hinge',dual=True,C=1.0)
    clf.fit(X_train, y_train)
    print "Coefficients:%s, intercept:%s" % (clf.coef_, clf.intercept_)
    print "Score:%.2f" % clf.score(X_test, y_test)

def test_LinearSVC_C(X_train, X_test, y_train, y_test):
    # 对惩罚参数C进行实验
    train_scores=[]
    test_scores=[]
    Cs=np.logspace(-2,2)
    for c in Cs:
        clf = svm.LinearSVC(penalty='l2',loss='squared_hinge',dual=True,C=c)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,train_scores,label="Train_Score",marker='*')
    ax.plot(Cs,test_scores,label="Test_Score",marker='s')
    ax.set_xlabel("C")
    ax.set_ylabel("score")
    # 对数值横坐标
    ax.set_xscale('log')
    ax.set_title("linearSVC")
    ax.legend(loc='best')
    plt.show()

def testSVC(X_train, X_test, y_train, y_test):
    #测试多项式核函数
    fig=plt.figure()
    degree=range(1,20)
    train_scores=[]
    test_scores=[]
    for i in degree:
        clf = svm.SVC(kernel='poly',degree=i)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
    ax=fig.add_subplot(1,1,1)
    ax.plot(degree,train_scores,label="Train_Score",marker='+')
    ax.plot(degree,test_scores,label="Test_Score",marker='o')
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_title("SVC_poly_degree")
    ax.set_ylim(0,1.05)
    ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    test_LinearSVC(X_train, X_test, y_train, y_test)
    test_LinearSVC_C(X_train, X_test, y_train, y_test)
    testSVC(X_train, X_test, y_train, y_test)
