#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/1/17 11:00
# @Author  : duanhaobin
# @File    : kNN.py
# @Software: PyCharm
# @Desc    : 模拟sklearn,封装自己的kNN模型

import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:

    def __init__(self, k):
        """
        初始化kNN分类器
        """
        assert k >= 1, "k must be valid"

        self.k = k
        self.__X_train = None  # 私有变量
        self.__y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练分类器"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be at least k."

        self.__X_train = X_train
        self.__y_train = y_train
        return self

    def predict(self, X_predict):
        "给定待预测数据集X_predict,返回表示X_predict的结果向量"
        assert self.__X_train is not None and self.__y_train is not None,\
            "must fit before predict"
        assert X_predict.shape[1] == self.__X_train.shape[1],\
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x,返回x预测结果值"""
        assert x.shape[0] == self.__X_train.shape[1], \
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self.__X_train]
        neraest = np.argsort(distances)
        topK_y = [self.__y_train[near] for near in neraest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]


    def __repr__(self):
        return f"the value of k is:{self.k}"