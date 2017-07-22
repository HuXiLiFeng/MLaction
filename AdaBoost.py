#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本文件会调用sklearn库中的关于AdaBoost的一些方法
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,cross_validation,ensemble,model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVR


def loadData():
    # 在分类问题中，用的是手写识别数据集Digit dataset
    digits = datasets.load_digits()
    # 最后一个参数stratify表示分层抽样
    return model_selection.train_test_split(digits.data, digits.target, test_size=0.25, random_state=1,
                                             stratify = digits.target)
def loadRegressorData():
    # 糖尿病人数据集，用于回归
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=1)

def testAdaBoostClassifier(X_train,X_test,y_train,y_test):
    # 测试不同基分类器的效果
    bdt = ensemble.AdaBoostClassifier(learning_rate=0.1, n_estimators=50,algorithm='SAMME.R', random_state=0)
    bdt.fit(X_train, y_train)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    estimatorNum=len(bdt.estimators_)
    X=range(1,estimatorNum+1)
    ax.plot(list(X),list(bdt.staged_score(X_train, y_train)),label="DecisionTreeClassifier--Training Score")
    ax.plot(list(X), list(bdt.staged_score(X_test,y_test)), label="DecisionTreeClassifier--Testing Score")

    # 朴素贝叶斯分类器
    clf = ensemble.AdaBoostClassifier(base_estimator=GaussianNB(),learning_rate=0.1,n_estimators=50,
                                      algorithm='SAMME.R',random_state=0)
    clf.fit(X_train, y_train)
    ax.plot(list(X),list(clf.staged_score(X_train, y_train)),label="Navie Bayes--Training Score")
    ax.plot(list(X), list(clf.staged_score(X_test,y_test)), label="Navie Bayes--Testing Score")

    ax.set_xlabel("estimators num")
    ax.set_ylabel("Score/Accuracy")
    ax.legend(loc='best')
    ax.set_title("AdaBoostClassifier")
    plt.show()
    print "基分类器：决策树分类器（默认），精度：%.4f"%bdt.score(X_test,y_test)
    print "基分类器：贝叶斯分类器（默认），精度：%.4f" % clf.score(X_test, y_test)

def testClassifyLearningRate(X_train,X_test,y_train,y_test):
    learningRates=np.linspace(0.01,1,num=40)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    trainingScore=[]
    testingScore=[]
    alforithms=['SAMME.R','SAMME']
    for learning_rate in learningRates:
        bdt = ensemble.AdaBoostClassifier(learning_rate=learning_rate, n_estimators=100, algorithm=alforithms[1], random_state=0)
        bdt.fit(X_train, y_train)
        trainingScore.append(bdt.score(X_train,y_train))
        testingScore.append(bdt.score(X_test,y_test))
    ax.plot(learningRates, trainingScore, label="Training Score")
    ax.plot(learningRates, testingScore, label="Testing Score")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("Score/Accuracy")
    ax.legend(loc='best')
    ax.set_title("AdaBoostClassifier: learning rate VS. Score")
    plt.show()
    print "训练集中得到最大的训练精度为：%.3f，是第%s次迭代得到的"%(max(trainingScore),trainingScore.index(max(trainingScore)))
    print "测试集中得到最大的训练精度为：%.3f，是第%s次迭代得到的" % (max(testingScore), testingScore.index(max(testingScore)))

def testAdaBoostRegressor(X_train, X_test, y_train, y_test):
    # 测试AdaBoost回归
    fig = plt.figure()
    loss=['linear','square','exponential']
    ax = fig.add_subplot(1, 1, 1)
    # 测试不同损失函数的影响
    for i in range(len(loss)):
        bdt = ensemble.AdaBoostRegressor(learning_rate=0.1, n_estimators=50, random_state=0,loss=loss[i])
        bdt.fit(X_train, y_train)
        estimatorNum = len(bdt.estimators_)
        X = range(1, estimatorNum + 1)
        ax.plot(list(X), list(bdt.staged_score(X_train, y_train)), label="Training Score:loss=%s"%loss[i])
        ax.plot(list(X), list(bdt.staged_score(X_test, y_test)), label="Testing Score:loss=%s"%loss[i])
        ax.set_xlabel("estimators num")
        ax.set_ylabel("Score")
        ax.legend(loc='best')
        print "损失函数：%s"%loss[i],"Train, Score：%.4f" % bdt.score(X_train, y_train)
        print "损失函数：%s"%loss[i],"Test, Score：%.4f" % bdt.score(X_test, y_test)
    plt.suptitle("AdaBoostRegressor==different loss")
    plt.show()

def testAdaBoostRegressorEstimators(X_train, X_test, y_train, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    estimators=[ensemble.AdaBoostRegressor(LinearSVR(epsilon=0.01, C=100)),ensemble.AdaBoostRegressor()]
    for i in range(len(estimators)):
        bdt = estimators[i]
        bdt.fit(X_train, y_train)
        estimatorNum = len(bdt.estimators_)
        X = range(1, estimatorNum + 1)
        ax.plot(list(X), list(bdt.staged_score(X_train, y_train)), label="Training Score:estimator%s"%i)
        ax.plot(list(X), list(bdt.staged_score(X_test, y_test)), label="Testing Score:estimator%s"%i)
        ax.set_xlabel("estimators num")
        ax.set_ylabel("Score")
        ax.legend(loc='best')
        print "基训练器：%s"%i,"Train, Score：%.4f" % bdt.score(X_train, y_train)
        print "基训练器：%s"%i,"Test, Score：%.4f" % bdt.score(X_test, y_test)
    plt.suptitle("AdaBoostRegressor==different base estimator")
    plt.show()

def testGradientBoostingClassifier(*data):
    X_train, X_test, y_train, y_test=data
    trainScore=[]
    testScore=[]
    # 可以调参：基训练器的个数，学习率等参数
    n_estimators=np.arange(1,100,step=5)
    learning=np.linspace(0.01,1.0)
    for i in learning:
        GBC=ensemble.GradientBoostingClassifier(loss='deviance',learning_rate=i,n_estimators=100,verbose=0)
        GBC.fit(X_train, y_train)
        trainScore.append(GBC.score(X_train,y_train))
        testScore.append(GBC.score(X_test, y_test))
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(learning,trainScore,label='trainScore')
    ax.plot(learning, testScore, label='testScore')
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("score/accuracy")
    ax.set_title('GBDT Classifier: score VS. learning_rate')
    ax.legend(loc='best')
    plt.show()

def testGradientBoostingRegressor(*data):
    X_train, X_test, y_train, y_test=data
    trainScore=[]
    testScore=[]
    # 可以调参：基训练器的个数，学习率,最大树深，基训练器最大特征个数等参数
    n_estimators=np.arange(1,100,step=5)
    learning=np.linspace(0.01,1.0,num=10)
    for i in learning:
        GBR=ensemble.GradientBoostingRegressor(loss='huber',learning_rate=i,n_estimators=100,verbose=0,
                                               max_features="auto",max_depth=3)
        GBR.fit(X_train, y_train)
        trainScore.append(GBR.score(X_train,y_train))
        testScore.append(GBR.score(X_test, y_test))
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(learning,trainScore,label='trainScore')
    ax.plot(learning, testScore, label='testScore')
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("score")
    ax.set_title('GBDT Regressor: score VS. learning_rate')
    ax.legend(loc='best')
    plt.show()

def testRandomForestClassifier(*data):
    estimators=np.arange(1,100,step=2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    trainingScore=[]
    testingScore=[]
    # 还可以调参
    # featuresNum=["auto","sqrt","log2",5,0.6]
    for estimator in estimators:
        bdt = ensemble.RandomForestClassifier(n_estimators=estimator,max_features="log2",max_depth=6)
        bdt.fit(X_train, y_train)
        trainingScore.append(bdt.score(X_train,y_train))
        testingScore.append(bdt.score(X_test,y_test))
    ax.plot(estimators, trainingScore, label="Training Score")
    ax.plot(estimators, testingScore, label="Testing Score")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Score")
    ax.legend(loc='best')
    ax.set_title("RandomForestClassifier: n_estimators VS. Score")
    plt.show()

def testRandomForestRegressor(*data):
    estimators = np.arange(1, 100, step=2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    trainingScore = []
    testingScore = []
    # 还可以调参
    # featuresNum=["auto","sqrt","log2",5,0.6]
    for estimator in estimators:
        bdt = ensemble.RandomForestRegressor(n_estimators=estimator, max_features="log2", max_depth=6)
        bdt.fit(X_train, y_train)
        trainingScore.append(bdt.score(X_train, y_train))
        testingScore.append(bdt.score(X_test, y_test))
    ax.plot(estimators, trainingScore, label="Training Score")
    ax.plot(estimators, testingScore, label="Testing Score")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Score")
    ax.legend(loc='best')
    ax.set_title("RandomForestRegressor: n_estimators VS. Score")
    plt.show()




if __name__ == '__main__':
    # 用于分类的数据
    # X_train, X_test, y_train, y_test=loadData()
    # testAdaBoostClassifier(X_train,X_test,y_train,y_test)
    # testClassifyLearningRate(X_train, X_test, y_train, y_test)
    # testGradientBoostingClassifier(X_train, X_test, y_train, y_test)
    # testRandomForestClassifier(X_train, X_test, y_train, y_test)



    # 用于回归的数据
    X_train, X_test, y_train, y_test = loadRegressorData()
    # testAdaBoostRegressor(X_train, X_test, y_train, y_test)
    # testAdaBoostRegressorEstimators(X_train, X_test, y_train, y_test)
    # testGradientBoostingRegressor(X_train, X_test, y_train, y_test)
    testRandomForestRegressor(X_train, X_test, y_train, y_test)







