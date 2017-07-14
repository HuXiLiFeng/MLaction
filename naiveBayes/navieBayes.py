#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from numpy import *
import math

def loadDataSet():
    #载入数据及对应标签
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    #记录语料中出现过的单词列表
    vocabSet=set()
    for i in dataSet:
        vocabSet=vocabSet|set(i)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    #根据createVocabList()函数生成的列表，为inputSet构建0-1向量
    vec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vec[vocabList.index(word)]=1
        else:
            print "the word: %s is not in my " % word
    return vec

def train(trainMatrix,trainCategory):
    numTrainExamples=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/numTrainExamples
    p0Num=[0]*numWords
    p1Num=[0]*numWords
    p0Demo=0
    p1Demo=0
    for i in xrange(numTrainExamples):
        if trainCategory[i]==1:   #标签为1，表示侮辱性文字；标签为0，表示正常言论
            p1Num=list(map(lambda x: x[0] + x[1], zip(p1Num, trainMatrix[i])))
            #p1Num+=trainMatrix[i]
            p1Demo+=sum(trainMatrix[i])
        else:
            p0Num = list(map(lambda x: x[0] + x[1], zip(p0Num, trainMatrix[i])))
            #p0Num+=trainMatrix[i]
            p0Demo+=sum(trainMatrix[i])
    print p1Demo,p0Demo
    p1Vect =[log(i/p1Demo) for i in p1Num]
    p0Vect =[log(i/p0Demo) for i in p0Num]
    # p1Vect=log(p1Num/p1Demo)
    # p0Vect=log(p0Num/p0Demo)
    return pAbusive,p1Vect,p0Vect

def classify(testVec,p0Vec,p1Vec,pClass1):
    #p(y|x)=p(x|y)*p(y)/p(x)
    #这里为了防止小数连乘出错，所以取对数
    p1=dot(testVec,p1Vec)+log(pClass1)
    p0=dot(testVec,p0Vec)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testBN(data,vabList,labels):
    trainMat = []
    for doc in data:
        trainMat.append(setOfWords2Vec(vabList, doc))
    pAbusive, p1Vect, p0Vect = train(trainMat, labels)
    print pAbusive, p1Vect, p0Vect
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(vabList,testEntry))
    print "the test entry is classified:%d " % classify(thisDoc,p0Vect,p1Vect,pAbusive)


if __name__ == '__main__':
    data,labels=loadDataSet()
    myVabList=createVocabList(data)
    print myVabList
    testBN(data,myVabList,labels)

