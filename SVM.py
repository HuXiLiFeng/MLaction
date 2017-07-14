#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

iris=datasets.load_iris()
X=iris.data[:,[0,1]]
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3,random_state=2)
print X_train.shape,X_test.shape
# 为了追求机器学习和最优化算法的最佳性能，将特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # 估算每个特征的平均值和标准差
print sc.mean_ # 查看特征的平均值，由于Iris只用了两个特征，所以结果是array([ 3.82857143,  1.22666667])
print sc.scale_ # 查看特征的标准差，这个结果是array([ 1.79595918,  0.77769705])
X_train_std = sc.transform(X_train)
# 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))    #垂直方向合并数组
y_combined = np.hstack((y_train, y_test))                #水平方向合并数组
print type(X_combined_std),X_combined_std.shape
print X_combined_std[0:5]

print type(y_combined),y_combined.shape
print y_combined[0:5]

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0) # 用线性核，你也可以通过kernel参数指定其它的核。
svm.fit(X_train_std, y_train)
# x1_min, x1_max = X_combined_std[:, 0].min() - 1, X_combined_std[:, 0].max() + 1
# x2_min, x2_max = X_combined_std[:, 1].min() - 1, X_combined_std[:, 1].max() + 1
# xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
# Z = svm.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
# print Z
# print svm.predict(np.c_[xx1.ravel(), xx2.ravel()])
print svm.predict(X_test)
print y_test
print svm.score(X_test,y_test)



