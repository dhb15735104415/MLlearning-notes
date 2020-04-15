#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 22:43
# @Author  : duanhaobin
# @File    : LinearRegression.py
# @Software: PyCharm
# @Desc    : 实现多元线性回归算法

import numpy as np
from accuracy_score.metrics import r2_score


class LinearRegression:

    def __init__(self):
        """初始化分类器"""
        self.coefficient_ = None  # 系数
        self.interception_ = None  # 截距
        self.__theta = None  # 根据公式直接就能计算出

    def fit_normal(self, X_train, y_train):
        """
            根据训练数据计算出正规方程解,训练LinearRegression模型
        :X_train 样本训练数据
        :y_train 标记训练数据
        :return self
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 给X_train的第一列增加1列数据,且每一项都为1
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])

        # 根据公式计算theta
        self.__theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        # 截距和系数都分别赋值
        self.coefficient_ = self.__theta[1:]
        self.interception_ = self.__theta[0]
        return self

    def predict(self, X_predict):
        """
            给定待预测数据集X_predict,返回表示X_predict的结果向量
        :X_train 样本待预测数据
        :return 预测的结果向量
        """
        assert self.interception_ is not None and self.coefficient_ is not None, \
            "must fit before predict!"

        assert X_predict.shape[1] == len(self.coefficient_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])

        return X_b.dot(self.__theta)

    def score(self, X_test, y_test):
        """
            根据测试数据集测试模型得分
        :X_test 测试数据集
        :y_test 测试标记集
        :return float 得分
        """
        y_predict = self.predict(X_test)

        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
