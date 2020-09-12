# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:57:50 2020

@author: yy
"""

import numpy as np
import pandas as pd


"""数据加载"""
train_data = pd.read_csv(r'C:\Users\yy\Desktop\BI\L3\L3-2\L3-code\titanic\train.csv')
test_data = pd.read_csv(r'C:\Users\yy\Desktop\BI\L3\L3-2\L3-code\titanic\test.csv')

"""数据探索"""
print(train_data.info())   # 数据集基本信息，包括数据集行列数、列数据类型、空值数等
print(train_data.describe())   # 连续型数据的分布描述
print(train_data.describe(include=['O']))   # 离散型数据的分布描述
print(train_data.head())   # 查看前10条数据
print(train_data.tail())   # 查看后5条数据

"""缺失值处理"""
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)   # 使用平均年龄来填充nan值
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)   # 使用票价的均值填充nan值
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].value_counts().index[0], inplace=True)   # 使用登录最多的港口来填充nan值
test_data['Embarked'].fillna(train_data['Embarked'].value_counts().index[0], inplace=True)

"""特征选择"""
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

"""特征预处理：离散数据数值化"""
from sklearn.feature_extraction import DictVectorizer
dvec = DictVectorizer(sparse=False)
X_train = dvec.fit_transform(train_features.to_dict(orient='record'))
y_train = train_labels 
X_test = dvec.transform(test_features.to_dict(orient='record'))
print(dvec.feature_names_)

"""训练模型-TPOT"""
from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=10, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
y_test = tpot.predict(X_test)
print(u'模型使用训练集数据的准确率为 %.4lf' % tpot.score(X_train, y_train))

"""训练模型-RandomForest"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(criterion='gini')
rf.fit(X_train, y_train)
y_test = rf.predict(X_test)

accuracy_rf = round(rf.score(X_train, y_train), 6)   # 模型准确率
print(u'RF的准确率为 %.4lf' % accuracy_rf)
print(u'RF的cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(rf, X_train, y_train, cv=10)))   # 模型K折平均准确率
