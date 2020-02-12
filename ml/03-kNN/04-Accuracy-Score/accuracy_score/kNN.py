#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/2/11 18:49
# @Author  : duanhaobin
# @File    : kNN.py
# @Software: PyCharm
# @Desc    : 模拟sklearn,封装自己的kNN模型

import numpy as np
from math import sqrt
from collections import Counter

class kNNClassifier:

    def __init__(self, k):
        """
        初始化kNN分类器
        """
        assert k >=1 ,"k must be valid"
        self.k = k
        self.__X_train = None
        self.__y_train = None

    "fit-> predict->result"
    def fit(self, X_train, y_train):
        """
        拟合数据
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of the X_train must be equal to the size of the y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be at least k"
        self.__X_train = X_train
        self.__y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的结果向量"""
        assert self.__X_train is not None and self.__y_train is not None,\
            "must fit before predict"
        assert X_predict.shape[1] == self.__X_train.shape[1],\
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """
        给定单个训练数据,返回预测结果值
        """
        assert x.shape[0] == self.__X_train.shape[1], \
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self.__X_train]
        nearest = np.argsort(distances)  # 返回距离最近的index集合
        topK_y = [self.__y_train[near] for near in nearest[:self.k]]  # 注意取前k个最近的
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """
        根据测试集 X_test 和 y_test来确定模型的准确度
        """
        y_predict = self.predict(X_test)
        return sum(y_predict == y_test) / len(y_test)

